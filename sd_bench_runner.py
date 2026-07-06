#!/usr/bin/env python3
"""
sd_bench_runner.py

Orchestrates performance benchmarks for stable-diffusion.cpp models.
Evaluates end-to-end generation latency, steps per second (it/s),
and peak RAM/VRAM utilization under standardized workloads.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sd_bench_runner")

PROXY_URL = os.getenv("PROXY_URL", "http://localhost:11434")
ROUTER_MODELS_DIR = Path(os.getenv("ROUTER_MODELS_DIR", ".alpaca-router"))
SD_CONTAINER_NAME = os.getenv("SD_CONTAINER_NAME", "sd-server")


class ResourceSampler:
    """Asynchronous context manager to sample peak CPU, RAM, and VRAM utilization during a query."""

    def __init__(self, container_name: str = SD_CONTAINER_NAME):
        self.container_name = container_name
        self.peak_ram_pct: float = 0.0
        self.peak_vram_mb: int = 0
        self.vram_total_mb: int = 0
        self.active: bool = False
        self.loop_task: Optional[asyncio.Task[None]] = None

    async def _sample_loop(self) -> None:
        while self.active:
            # Sample system RAM percentage
            ram = psutil.virtual_memory().percent
            if ram > self.peak_ram_pct:
                self.peak_ram_pct = ram

            # Sample GPU memory via local nvidia-smi
            try:
                proc = await asyncio.create_subprocess_exec(
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                if proc.returncode == 0:
                    lines = stdout.decode().strip().split("\n")
                    if lines:
                        total, used = [int(x.strip()) for x in lines[0].split(",")]
                        self.vram_total_mb = total
                        if used > self.peak_vram_mb:
                            self.peak_vram_mb = used
            except Exception:
                # Fallback to Docker exec query
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "docker",
                        "exec",
                        self.container_name,
                        "nvidia-smi",
                        "--query-gpu=memory.total,memory.used",
                        "--format=csv,noheader,nounits",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, _ = await proc.communicate()
                    if proc.returncode == 0:
                        lines = stdout.decode().strip().split("\n")
                        if lines:
                            total, used = [int(x.strip()) for x in lines[0].split(",")]
                            self.vram_total_mb = total
                            if used > self.peak_vram_mb:
                                self.peak_vram_mb = used
                except Exception:
                    pass
            await asyncio.sleep(0.1)

    def start(self) -> None:
        self.active = True
        self.loop_task = asyncio.create_task(self._sample_loop())

    async def stop(self) -> None:
        self.active = False
        if self.loop_task:
            self.loop_task.cancel()
            try:
                await self.loop_task
            except (asyncio.CancelledError, Exception):
                pass


def locate_model_file(model_alias: str) -> Path:
    """Finds the stable-diffusion model file in the router models directory."""
    # Try direct matches
    for ext in [".safetensors", ".gguf"]:
        direct_path = ROUTER_MODELS_DIR / f"{model_alias}{ext}"
        if direct_path.exists():
            return direct_path.resolve()

    # Try substring matches
    for path in ROUTER_MODELS_DIR.iterdir():
        if path.suffix in (".safetensors", ".gguf"):
            if model_alias.lower() in path.name.lower():
                return path.resolve()

    raise FileNotFoundError(
        f"Could not find model file for alias: {model_alias} in {ROUTER_MODELS_DIR}"
    )


async def run_benchmark_payload(
    model_alias: str, prompt: str, size: str, steps: int, runs: int = 3
) -> Dict[str, Any]:
    """Runs a series of generation queries against the proxy and profiles resource usage."""
    logger.info(
        f"Initiating benchmark sweep for model '{model_alias}' ({size}, {steps} steps, {runs} runs)..."
    )

    url = f"{PROXY_URL}/v1/images/generations"
    payload = {"model": model_alias, "prompt": prompt, "size": size, "n": 1, "steps": steps}

    run_results = []

    # Warm-up run to ensure model loading/compilation is excluded from direct performance latency
    logger.info("Executing warm-up run (ensuring model is resident)...")
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                logger.error(f"Warmup failed with status {resp.status_code}: {resp.text}")
                raise RuntimeError(f"Warmup generation failed: {resp.text}")
    except Exception as e:
        logger.error(f"Warm-up query failed: {e}")
        raise

    # Sweep profiling runs
    for run_idx in range(1, runs + 1):
        logger.info(f"Executing profiling run {run_idx}/{runs}...")
        sampler = ResourceSampler()
        sampler.start()

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
        except Exception as e:
            logger.error(f"Run {run_idx} failed: {e}")
            await sampler.stop()
            raise RuntimeError(f"Benchmark run {run_idx} failed: {str(e)}")

        end_time = time.perf_counter()
        await sampler.stop()

        latency = end_time - start_time
        it_s = steps / latency if latency > 0 else 0.0

        logger.info(
            f"Run {run_idx} complete: Latency = {latency:.2f}s, Speed = {it_s:.2f} it/s, Peak VRAM = {sampler.peak_vram_mb} MB"
        )

        run_results.append(
            {
                "run": run_idx,
                "latency_sec": latency,
                "iterations_per_sec": it_s,
                "peak_vram_mb": sampler.peak_vram_mb,
                "peak_ram_pct": sampler.peak_ram_pct,
            }
        )

    # Compile aggregate results
    avg_latency = sum(r["latency_sec"] for r in run_results) / runs
    avg_it_s = sum(r["iterations_per_sec"] for r in run_results) / runs
    peak_vram = max(r["peak_vram_mb"] for r in run_results)
    peak_ram = max(r["peak_ram_pct"] for r in run_results)

    summary = {
        "model": model_alias,
        "prompt": prompt,
        "resolution": size,
        "steps": steps,
        "runs": runs,
        "average_latency_sec": round(avg_latency, 3),
        "average_iterations_per_sec": round(avg_it_s, 2),
        "peak_vram_mb": peak_vram,
        "peak_ram_pct": round(peak_ram, 2),
        "runs_detailed": run_results,
    }

    return summary


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 sd_bench_runner.py <model_alias> [steps] [resolution] [runs]")
        print("Example: python3 sd_bench_runner.py stable-diffusion-xl-base-1.0 20 512x512 3")
        sys.exit(1)

    model_alias = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    resolution = sys.argv[3] if len(sys.argv) > 3 else "512x512"
    runs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    prompt = "A high-resolution photograph of a beautiful valley with a clear stream at sunset, cinematic lighting."

    try:
        # Verify model exists on host
        model_path = locate_model_file(model_alias)
        logger.info(f"Verified model file exists: {model_path}")

        # Run benchmark
        results = asyncio.run(
            run_benchmark_payload(
                model_alias=model_alias, prompt=prompt, size=resolution, steps=steps, runs=runs
            )
        )

        # Save results
        out_dir = Path("data/sd_benchmarks")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"sd_bench_{model_alias}_profile.json"

        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

        print("\n=== STABLE DIFFUSION BENCHMARK RESULTS ===")
        print(f"Model:                     {results['model']}")
        print(f"Resolution:                {results['resolution']}")
        print(f"Steps:                     {results['steps']}")
        print(f"Average Latency:           {results['average_latency_sec']:.2f}s")
        print(f"Average Speed (it/s):      {results['average_iterations_per_sec']:.2f} it/s")
        print(f"Peak VRAM Consumption:     {results['peak_vram_mb']} MB")
        print(f"Peak RAM Consumption:      {results['peak_ram_pct']}%")
        print(f"Profile saved to:          {out_file}\n")

    except Exception:
        logger.exception("Benchmark run failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

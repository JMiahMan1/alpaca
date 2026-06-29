#!/usr/bin/env python3
"""
telemetry_monitor.py

Lightweight daemon to poll and record running metrics (System RAM, VRAM,
context window growth, CPU/GPU utilization) over time on a per-model basis.
This tracks the memory 'creep' that leads to self-healing triggers.
"""

import asyncio
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path

import httpx
import psutil

# Configuration from environment variables
POLL_INTERVAL = float(os.getenv("TELEMETRY_POLL_INTERVAL", "5.0"))
TELEMETRY_DIR = Path(os.getenv("TELEMETRY_DIR", "data/telemetry"))
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://llama-server:8080")
DOCKER_CONTAINER = os.getenv("LLAMA_DOCKER_CONTAINER", "llama-server")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("telemetry_monitor")

# Global asyncio stop event — set by signal handler to unblock the main loop
# Using asyncio.Event instead of a plain bool so that signal delivery is picked
# up promptly even while the event loop is blocked inside asyncio.gather().
_stop_event: asyncio.Event | None = None


def handle_signals(signum, frame):
    logger.info(f"Signal {signum} received. Shutting down daemon gracefully...")
    if _stop_event is not None:
        # call_soon_threadsafe is required because signal handlers run outside
        # the asyncio event loop thread.
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(_stop_event.set)


# Register shutdown signals
signal.signal(signal.SIGINT, handle_signals)
signal.signal(signal.SIGTERM, handle_signals)


async def get_system_metrics():
    """Retrieve system CPU and RAM usage metrics."""
    try:
        mem = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=None)
        return {
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "ram_used_gb": round(mem.used / (1024**3), 2),
            "ram_free_gb": round(mem.available / (1024**3), 2),
            "ram_used_pct": mem.percent,
            "cpu_util_pct": cpu_pct,
        }
    except Exception as e:
        logger.error(f"Error querying system metrics: {e}")
        return {
            "ram_total_gb": 0.0,
            "ram_used_gb": 0.0,
            "ram_free_gb": 0.0,
            "ram_used_pct": 0.0,
            "cpu_util_pct": 0.0,
        }


async def get_gpu_metrics():
    """Retrieve GPU memory and utilization via nvidia-smi.
    Tries docker container execution first, then falls back to local execution.
    """
    cmd_docker = [
        "docker",
        "exec",
        DOCKER_CONTAINER,
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    cmd_host = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]

    # Try docker exec first
    stdout = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd_docker, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await proc.communicate()
        if proc.returncode == 0:
            stdout = out.decode().strip()
    except Exception:
        pass

    # Fall back to host execution if docker exec failed or returned non-zero
    if not stdout:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_host, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            out, err = await proc.communicate()
            if proc.returncode == 0:
                stdout = out.decode().strip()
        except Exception as e:
            logger.debug(f"Host nvidia-smi execution failed: {e}")

    gpus = []
    if stdout:
        try:
            for line in stdout.split("\n"):
                if line:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        idx_str, name, total_str, used_str, free_str, util_str = parts[:6]
                        try:
                            idx = int(idx_str)
                            total = int(total_str) if "n/a" not in total_str.lower() else 0
                            used = int(used_str) if "n/a" not in used_str.lower() else 0
                            free = int(free_str) if "n/a" not in free_str.lower() else 0
                            util = float(util_str) if "n/a" not in util_str.lower() else 0.0

                            gpus.append(
                                {
                                    "index": idx,
                                    "name": name,
                                    "vram_total_mb": total,
                                    "vram_used_mb": used,
                                    "vram_free_mb": free,
                                    "vram_used_pct": round(used / total * 100, 1)
                                    if total > 0
                                    else 0.0,
                                    "gpu_util_pct": util,
                                }
                            )
                        except ValueError as val_err:
                            logger.debug(
                                f"Failed to parse numeric value from nvidia-smi parts: {parts} - Error: {val_err}"
                            )
        except Exception as e:
            logger.error(f"Error parsing nvidia-smi output: {e}")

    # Fallback/mock structure if no GPU metrics could be retrieved
    if not gpus:
        gpus.append(
            {
                "index": 0,
                "name": "Unknown GPU (Unavailable)",
                "vram_total_mb": 0,
                "vram_used_mb": 0,
                "vram_free_mb": 0,
                "vram_used_pct": 0.0,
                "gpu_util_pct": 0.0,
            }
        )

    return gpus


async def get_llama_server_metrics(client: httpx.AsyncClient):
    """Retrieve runtime settings and slot utilization from llama-server."""
    props = {}
    slots = []
    model_alias = "unknown_model"
    backend_model = None

    # Detect the active model name and backend model from the proxy runtime status
    proxy_url = os.getenv("PROXY_URL", "http://alpaca-proxy:11434")
    try:
        resp = await client.get(f"{proxy_url}/admin/runtime", timeout=1.0)
        if resp.status_code == 200:
            data = resp.json()
            loaded = data.get("loaded_models", [])
            if loaded:
                public_name = loaded[0]["name"]
                backend_model = loaded[0]["backend_model"]
                # Sanitize the public name (e.g. replacing '/' with '_') to match the web app's expectation
                model_alias = re.sub(r"[^\w\-.\.]", "_", public_name)
    except Exception as e:
        logger.debug(f"Could not reach alpaca-proxy for runtime info: {e}")

    # 1. Fetch props
    try:
        resp = await client.get(f"{LLAMA_SERVER_URL}/props", timeout=2.0)
        if resp.status_code == 200:
            props = resp.json()
    except Exception as e:
        logger.debug(f"Could not reach llama-server /props: {e}")

    # Fallback to model_path stem if we couldn't get the name from the proxy
    if model_alias == "unknown_model" or model_alias == "system_idle":
        model_path = props.get("model_path")
        if model_path and model_path != "none":
            raw_stem = Path(model_path).stem
            model_alias = re.sub(r"[^\w\-.\.]", "_", raw_stem)
        else:
            model_alias = "system_idle"

    # 2. Fetch slots (for context usage tracking)
    try:
        slots_url = f"{LLAMA_SERVER_URL}/slots"
        if backend_model:
            slots_url += f"?model={backend_model}"
        resp = await client.get(slots_url, timeout=2.0)
        if resp.status_code == 200:
            slots = resp.json()
    except Exception as e:
        logger.debug(f"Could not reach llama-server /slots: {e}")

    # Context window stats
    n_ctx = props.get("n_ctx", 0)

    # Analyze slots
    total_slots = len(slots)
    active_slots = 0
    total_tokens_cached = 0

    for slot in slots:
        state = slot.get("state", 0)  # 0 = IDLE, 1 = PROCESSING/BUSY in typical llama.cpp
        # Sometimes slot state is string or dict, check for active keys
        is_processing = False
        if isinstance(slot.get("is_processing"), bool):
            is_processing = slot["is_processing"]
        elif state != 0:
            is_processing = True

        if is_processing:
            active_slots += 1

        # Track tokens cached (in llama.cpp this is n_past)
        n_past = slot.get("n_past", 0)
        total_tokens_cached += n_past

    # Calculate average utilization
    kv_cache_used_pct = (
        round(total_tokens_cached / (n_ctx * total_slots) * 100, 1)
        if (n_ctx > 0 and total_slots > 0)
        else 0.0
    )

    return {
        "model_alias": model_alias,
        "model_path": props.get("model_path"),
        "n_ctx": n_ctx,
        "n_gpu_layers": props.get("n_gpu_layers"),
        "flash_attn": props.get("flash_attn"),
        "total_slots": total_slots,
        "active_slots": active_slots,
        "total_tokens_cached": total_tokens_cached,
        "kv_cache_used_pct": kv_cache_used_pct,
    }


def write_telemetry_log(model_alias: str, data: dict):
    """Write telemetry data point into model's JSONL file."""
    TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    log_file = TELEMETRY_DIR / f"{model_alias}.jsonl"

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        logger.error(f"Failed to write telemetry for {model_alias}: {e}")


async def main():
    global _stop_event
    _stop_event = asyncio.Event()

    logger.info("Initializing Telemetry Monitor Daemon...")
    logger.info(f"Poll Interval: {POLL_INTERVAL} seconds")
    logger.info(f"Telemetry Dir: {TELEMETRY_DIR.resolve()}")
    logger.info(f"Llama-Server URL: {LLAMA_SERVER_URL}")

    # Build HTTP Client with pool settings
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    async with httpx.AsyncClient(limits=limits, timeout=5.0) as client:
        while not _stop_event.is_set():
            loop_start = time.time()

            # Fetch all categories of metrics concurrently;
            # wrap in a cancellable task so SIGTERM can interrupt a slow gather.
            gather_task = asyncio.ensure_future(
                asyncio.gather(
                    get_system_metrics(),
                    get_gpu_metrics(),
                    get_llama_server_metrics(client),
                )
            )
            stop_task = asyncio.ensure_future(_stop_event.wait())

            done, pending = await asyncio.wait(
                {gather_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel whichever task is still running
            for t in pending:
                t.cancel()

            if _stop_event.is_set():
                break

            sys_metrics, gpu_metrics, llama_metrics = await gather_task

            # Get active model alias or default
            model_alias = llama_metrics.get("model_alias", "system_idle")

            # Aggregate payload
            payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "epoch_time": time.time(),
                "model_alias": model_alias,
                "system": sys_metrics,
                "gpus": gpu_metrics,
                "llama_server": {
                    "model_path": llama_metrics.get("model_path"),
                    "n_ctx": llama_metrics.get("n_ctx"),
                    "n_gpu_layers": llama_metrics.get("n_gpu_layers"),
                    "flash_attn": llama_metrics.get("flash_attn"),
                    "slots": {
                        "total": llama_metrics.get("total_slots"),
                        "active": llama_metrics.get("active_slots"),
                        "tokens_cached": llama_metrics.get("total_tokens_cached"),
                        "kv_cache_used_pct": llama_metrics.get("kv_cache_used_pct"),
                    },
                },
            }

            # Save telemetry to disk
            write_telemetry_log(model_alias, payload)
            logger.debug(
                f"Recorded telemetry point for {model_alias} (Sys RAM: {sys_metrics['ram_used_pct']}%, VRAM: {gpu_metrics[0]['vram_used_pct'] if gpu_metrics else 0.0}%)"
            )

            # Sleep the remainder of the interval, but wake early on stop
            elapsed = time.time() - loop_start
            sleep_time = max(0.1, POLL_INTERVAL - elapsed)
            try:
                await asyncio.wait_for(_stop_event.wait(), timeout=sleep_time)
            except asyncio.TimeoutError:
                pass  # Normal path — timeout means we keep looping

    logger.info("Telemetry Monitor Daemon stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Daemon interrupted by keyboard. Exiting.")

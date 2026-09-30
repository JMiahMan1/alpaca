#!/usr/bin/env python3
"""
telemetry_monitor.py

Lightweight daemon to poll and record running metrics (System RAM, VRAM,
context window growth, CPU/GPU utilization) over time on a per-model basis.
This tracks the memory 'creep' that leads to self-healing triggers.
"""

import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import psutil

# Configuration from environment variables
POLL_INTERVAL = float(os.getenv("TELEMETRY_POLL_INTERVAL", "5.0"))
TELEMETRY_DIR = Path(os.getenv("TELEMETRY_DIR", "data/telemetry"))
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://llama-server:8080")
DOCKER_CONTAINER = os.getenv("LLAMA_DOCKER_CONTAINER", "llama-server")

# Retention. Without this the directory grew without bound: 33 files totalling
# 1.2 GB over 42 days, because a per-model file is only ever appended to and
# nothing ever rolled it or aged it out.
TELEMETRY_MAX_BYTES = int(os.getenv("TELEMETRY_MAX_BYTES", str(16 * 1024 * 1024)))
TELEMETRY_MAX_GENERATIONS = int(os.getenv("TELEMETRY_MAX_GENERATIONS", "3"))
TELEMETRY_RETENTION_DAYS = float(os.getenv("TELEMETRY_RETENTION_DAYS", "14"))
# How often to age files out. Hourly is far more often than a daily retention
# window needs, and costs one stat per file.
PRUNE_INTERVAL_S = float(os.getenv("TELEMETRY_PRUNE_INTERVAL_S", "3600"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("telemetry_monitor")

# Global asyncio stop event - set by signal handler to unblock the main loop
# Using asyncio.Event instead of a plain bool so that signal delivery is picked
# up promptly even while the event loop is blocked inside asyncio.gather().
_stop_event: asyncio.Event | None = None
# The loop that owns _stop_event. Signal handlers run outside the loop thread, so
# they cannot call get_running_loop() and must not call get_event_loop() either:
# since 3.12 asyncio.run() does not install a thread-local "current" loop, so
# get_event_loop() there raises RuntimeError and the graceful shutdown is lost.
_stop_loop: asyncio.AbstractEventLoop | None = None
# Counts polls since the last retention sweep. A module global rather than a
# local of main() so the counter survives the loop being re-entered.
_sweeps = 0


def handle_signals(signum, frame):
    logger.info(f"Signal {signum} received. Shutting down daemon gracefully...")
    if _stop_event is None or _stop_loop is None:
        return
    # call_soon_threadsafe is required because signal handlers run outside
    # the asyncio event loop thread.
    _stop_loop.call_soon_threadsafe(_stop_event.set)


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
        out, _err = await proc.communicate()
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
            out, _err = await proc.communicate()
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
                                    "vram_used_pct": round(used / total * 100, 1) if total > 0 else 0.0,
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
    proxy_url = os.getenv("PROXY_URL", "http://host.docker.internal:11434")
    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    proxy_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = await client.get(f"{proxy_url}/admin/runtime", headers=proxy_headers, timeout=1.0)
        if resp.status_code == 200:
            data = resp.json()
            loaded = data.get("loaded_models", [])
            if loaded:
                public_name = loaded[0]["name"]
                backend_model = loaded[0]["backend_model"]
                # Sanitize the public name to match the web app expectation
                model_alias = re.sub(r"[/:.]", "_", public_name)
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
            model_alias = re.sub(r"[/:.]", "_", raw_stem)
        else:
            model_alias = "system_idle"

    # 2. Fetch slots (for context usage tracking)
    if backend_model:
        try:
            slots_url = f"{LLAMA_SERVER_URL}/slots"
            slots_resp = await client.get(slots_url, params={"model": backend_model}, timeout=2.0)
            if slots_resp.status_code == 200:
                slots = slots_resp.json()
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
        round(total_tokens_cached / (n_ctx * total_slots) * 100, 1) if (n_ctx > 0 and total_slots > 0) else 0.0
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


def _rotate_if_oversized(log_file: Path) -> None:
    """Roll ``log_file`` -> ``.1`` -> ``.2`` ... once it passes the size cap.

    Shifted rather than clobbered so a file already being read keeps existing
    for the length of the shift; the oldest generation is dropped. Nothing here
    raises: telemetry is an observation log and must never be the thing that
    takes the daemon down.
    """
    try:
        if not log_file.exists() or log_file.stat().st_size < TELEMETRY_MAX_BYTES:
            return
        oldest = log_file.with_suffix(f"{log_file.suffix}.{TELEMETRY_MAX_GENERATIONS}")
        if oldest.exists():
            oldest.unlink()
        for gen in range(TELEMETRY_MAX_GENERATIONS - 1, 0, -1):
            src = log_file.with_suffix(f"{log_file.suffix}.{gen}")
            if src.exists():
                src.replace(log_file.with_suffix(f"{log_file.suffix}.{gen + 1}"))
        log_file.replace(log_file.with_suffix(f"{log_file.suffix}.1"))
    except OSError as e:
        logger.warning(f"Could not rotate {log_file.name}: {e}")


def prune_old_telemetry(max_age_days: float | None = None) -> int:
    """Delete telemetry files untouched for longer than the retention window.

    Two independent bounds keep the directory finite. Age is this function: a
    model you have not run in two weeks should not still be costing disk, and
    its history is no longer read by anything (the analyzer only looks at the
    last hour). Size is ``_rotate_if_oversized``: an actively-written file is
    always kept here and instead rolled into a bounded number of generations, so
    there is no path where a hot file is deleted out from under the daemon.
    """
    days = TELEMETRY_RETENTION_DAYS if max_age_days is None else max_age_days
    cutoff = time.time() - (days * 86400)
    removed = 0
    try:
        entries = list(TELEMETRY_DIR.glob("*.jsonl*"))
    except OSError:
        return 0
    for path in entries:
        try:
            if path.stat().st_mtime >= cutoff:
                continue
            path.unlink()
            removed += 1
        except OSError as e:
            logger.warning(f"Could not prune {path.name}: {e}")
    if removed:
        logger.info(f"Pruned {removed} telemetry file(s) older than {days:g} days")
    return removed


def write_telemetry_log(model_alias: str, data: dict):
    """Append one telemetry record, in a single write.

    Two things this deliberately does not do the obvious way.

    ``open(..., "ab", buffering=0)`` rather than ``open(..., "a")``: the buffered
    text writer accumulates the record in user space and emits it on close, so a
    process killed in between leaves whatever the flush had managed to push --
    which is how ``system_idle.jsonl`` ended up with a record cut off mid-key
    ("...n_ctx") at line 172 of 365,326. One unbuffered write of one complete
    line to an O_APPEND descriptor is the smallest unit the OS will reorder or
    interleave, so a record is either wholly there or wholly absent.

    It also repairs a tail left by an earlier version: if the file does not end
    in a newline the next record would otherwise concatenate onto the damaged
    one, turning one bad record into two. The newline costs nothing and keeps
    the damage to the line that caused it. Readers already skip unparseable
    lines rather than failing the whole request.
    """
    try:
        TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
        log_file = TELEMETRY_DIR / f"{model_alias}.jsonl"
        # Rotation is housekeeping; the measurement is the product. A rotation
        # that fails (read-only dir, a racing prune, ENOSPC mid-shift) must not
        # cost us the point we just took, so it is contained here and the write
        # goes ahead regardless.
        try:
            _rotate_if_oversized(log_file)
        except OSError as e:
            logger.warning(f"Rotation skipped for {log_file.name}: {e}")
        line = json.dumps(data) + "\n"
        # A previous record may have been cut off without its newline. Decided
        # from the size, not from f.tell(): on an O_APPEND descriptor Linux
        # leaves the offset at 0 until the first write, so tell() says "empty"
        # for a file with 365,000 lines in it.
        try:
            has_content = log_file.stat().st_size > 0
        except OSError:
            has_content = False
        if has_content:
            with open(log_file, "rb") as probe:
                probe.seek(-1, os.SEEK_END)
                if probe.read(1) != b"\n":
                    line = "\n" + line
        with open(log_file, "ab", buffering=0) as f:
            f.write(line.encode("utf-8"))
    except Exception as e:
        logger.error(f"Failed to write telemetry for {model_alias}: {e}")


async def main():
    global _stop_event, _stop_loop, _sweeps
    _stop_event = asyncio.Event()
    _stop_loop = asyncio.get_running_loop()
    # One sweep shortly after startup reclaims whatever accumulated while the
    # daemon was down, rather than waiting a full interval.
    _sweeps = int(PRUNE_INTERVAL_S // POLL_INTERVAL) if POLL_INTERVAL > 0 else 0

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

            _done, pending = await asyncio.wait({gather_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)

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
                "timestamp": datetime.now(UTC).isoformat(),
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

            # Retention sweep, hourly rather than every poll: it stats the whole
            # directory and the point is to reclaim disk, not to be exact about
            # it. Deleting is idempotent, so a restart mid-sweep is harmless.
            _sweeps += 1
            if _sweeps * POLL_INTERVAL >= PRUNE_INTERVAL_S:
                _sweeps = 0
                prune_old_telemetry()

            # Sleep the remainder of the interval, but wake early on stop
            elapsed = time.time() - loop_start
            sleep_time = max(0.1, POLL_INTERVAL - elapsed)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(_stop_event.wait(), timeout=sleep_time)

    logger.info("Telemetry Monitor Daemon stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Daemon interrupted by keyboard. Exiting.")

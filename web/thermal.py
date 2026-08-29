"""UI-configurable thermal watchdog for the web benchmark path.

Config lives in data/thermal_watchdog.json (written via /api/thermal/watchdog).
All values are user-adjustable from the dashboard; the watchdog can be turned
off entirely (enabled=false). Environment overrides (THERMAL_WATCHDOG,
THERMAL_ABORT_C, ...) take precedence for headless deployments.

Temperature probes reuse settings_scan's helpers (nvidia-smi for GPU,
lm-sensors Tctl for CPU). A watchdog instance tracks per-test peaks so
benchmark results can carry temp/performance statistics per model.
"""

import asyncio
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

# Standalone defaults; every value is overridable via the UI (data file) or env.
DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "abort_c": 93.0,
    "throttle_c": 85.0,
    "resume_c": 78.0,
    "poll_s": 5.0,
    "pretest_max_wait_s": 600.0,
}
CONFIG_PATH = Path("data/thermal_watchdog.json")

_ABSENT = object()


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


def load_config() -> dict[str, Any]:
    """Resolve watchdog config: env override > data/thermal_watchdog.json > defaults."""
    cfg = dict(DEFAULT_CONFIG)
    path = Path(os.environ.get("THERMAL_WATCHDOG_CONFIG", "data/thermal_watchdog.json"))
    try:
        if path.exists():
            stored = json.loads(path.read_text())
            if isinstance(stored, dict):
                for key in DEFAULT_CONFIG:
                    if key in stored:
                        cfg[key] = stored[key]
    except Exception as e:  # unreadable file -> defaults
        print(f"[thermal] config read failed ({e}); using defaults")
    env_map = {
        "enabled": _env_bool("THERMAL_WATCHDOG"),
        "abort_c": _env_float("THERMAL_ABORT_C"),
        "throttle_c": _env_float("THERMAL_THROTTLE_C"),
        "resume_c": _env_float("THERMAL_RESUME_C"),
        "poll_s": _env_float("THERMAL_POLL_S"),
        "pretest_max_wait_s": _env_float("THERMAL_PRETEST_MAX_WAIT_S"),
    }
    for key, val in env_map.items():
        if val is not None:
            cfg[key] = val
    if cfg["throttle_c"] >= cfg["abort_c"]:
        cfg["throttle_c"] = cfg["abort_c"] - 8.0
    if cfg["resume_c"] >= cfg["throttle_c"]:
        cfg["resume_c"] = cfg["throttle_c"] - 7.0
    return cfg


def save_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate + persist user-facing keys; returns the normalized config."""
    clean: dict[str, Any] = {}
    clean["enabled"] = bool(cfg.get("enabled", DEFAULT_CONFIG["enabled"]))
    for key, lo, hi in (
        ("abort_c", 60.0, 100.0),
        ("throttle_c", 50.0, 99.0),
        ("resume_c", 40.0, 95.0),
        ("poll_s", 1.0, 60.0),
        ("pretest_max_wait_s", 0.0, 3600.0),
    ):
        val = float(cfg.get(key, DEFAULT_CONFIG[key]))
        clean[key] = max(lo, min(hi, val))
    if clean["throttle_c"] >= clean["abort_c"]:
        clean["throttle_c"] = clean["abort_c"] - 8.0
    if clean["resume_c"] >= clean["throttle_c"]:
        clean["resume_c"] = clean["throttle_c"] - 7.0
    path = Path(os.environ.get("THERMAL_WATCHDOG_CONFIG", "data/thermal_watchdog.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2))
    return clean


def _gpu_temp_c() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _cpu_tctl_c() -> float | None:
    try:
        out = subprocess.run(["sensors"], capture_output=True, text=True, timeout=5, check=False)
        m = re.search(r"Tctl:\s+\+?([\d.]+)", out.stdout)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def _read_local_temps() -> dict[str, Any]:
    """Direct local probes (host-run dev; containers usually lack nvidia-smi/sensors)."""
    return {"gpu": _gpu_temp_c(), "cpu": _cpu_tctl_c()}


def _proxy_base_urls() -> list[str]:
    """Proxy base URLs from the same env the benchmark suite uses."""
    raw = os.environ.get("PROXY_SERVER_URLS", "") or os.environ.get("PROXY_URL", "")
    urls = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    if not urls:
        raise RuntimeError(
            "PROXY_SERVER_URLS is not set - the web container cannot reach the proxy for "
            "temperature probes. Set PROXY_SERVER_URLS (e.g. http://host.docker.internal:11434) "
            "in the alpaca-web environment."
        )
    return urls


def _read_proxy_temps() -> dict[str, Any] | None:
    """Fetch temps from the proxy's /admin/temps (docker exec nvidia-smi + hwmon Tctl)."""
    import httpx

    last_error: Exception | None = None
    for base in _proxy_base_urls():
        try:
            resp = httpx.get(f"{base}/admin/temps", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            return {"gpu": data.get("gpu"), "cpu": data.get("cpu")}
        except Exception as e:
            last_error = e
    raise RuntimeError(f"no proxy /admin/temps reachable: {last_error}")


def _read_live_temps() -> dict[str, Any]:
    """Live temperature probe: proxy /admin/temps first, local probes as fallback."""
    try:
        temps = _read_proxy_temps()
        if temps and (temps.get("cpu") is not None or temps.get("gpu") is not None):
            return temps
    except Exception as e:
        print(f"[thermal] proxy temp probe failed ({e}); falling back to local probes")
    return _read_local_temps()


class ThermalAbortError(RuntimeError):
    """Raised mid-generation when the abort temperature is reached."""


class ThermalWatchdog:
    """Tracks temperatures during a benchmark test and enforces throttle/abort."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = {**load_config(), **(config or {})}
        self.enabled = bool(self.config["enabled"])
        self.peak_cpu_c: float | None = None
        self.peak_gpu_c: float | None = None
        self.throttled_s: float = 0.0
        self.aborted: bool = False
        self.abort_reason: str | None = None
        self._last_read_t: float = 0.0
        self._last_temps: dict[str, float | None] = {}

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "peak_cpu_c": self.peak_cpu_c,
            "peak_gpu_c": self.peak_gpu_c,
            "throttled_s": round(self.throttled_s, 1),
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
        }

    def _read_temps(self) -> dict[str, float | None]:
        return _read_live_temps()

    async def heartbeat(self) -> str:
        """Poll temps (rate-limited). Returns 'ok' | 'throttle' | 'abort'.

        'throttle' means: keep calling heartbeat (sleep between calls) until
        the CPU cools below resume_c. Raises ThermalAbortError on abort.
        """
        if not self.enabled or self.aborted:
            return "ok" if not self.aborted else "abort"
        now = time.time()
        if now - self._last_read_t < float(self.config["poll_s"]):
            return "ok"
        self._last_read_t = now
        temps = await asyncio.to_thread(self._read_temps)
        self._last_temps = temps
        cpu, gpu = temps.get("cpu"), temps.get("gpu")
        if cpu is not None:
            self.peak_cpu_c = cpu if self.peak_cpu_c is None else max(self.peak_cpu_c, cpu)
        if gpu is not None:
            self.peak_gpu_c = gpu if self.peak_gpu_c is None else max(self.peak_gpu_c, gpu)
        abort_c = float(self.config["abort_c"])
        if (cpu is not None and cpu >= abort_c) or (gpu is not None and gpu >= abort_c):
            self.aborted = True
            self.abort_reason = f"cpu={cpu}C gpu={gpu}C >= abort {abort_c}C"
            print(f"[thermal] ABORT: {self.abort_reason}")
            raise ThermalAbortError(self.abort_reason)
        throttle_c = float(self.config["throttle_c"])
        resume_c = float(self.config["resume_c"])
        if (cpu is not None and cpu >= throttle_c) or (gpu is not None and gpu >= throttle_c):
            pause_start = time.time()
            print(f"[thermal] THROTTLE pause: cpu={cpu}C gpu={gpu}C (resume at {resume_c}C)")
            while True:
                await asyncio.sleep(3.0)
                temps = await asyncio.to_thread(self._read_temps)
                cpu, gpu = temps.get("cpu"), temps.get("gpu")
                if cpu is not None:
                    self.peak_cpu_c = cpu if self.peak_cpu_c is None else max(self.peak_cpu_c, cpu)
                if gpu is not None:
                    self.peak_gpu_c = gpu if self.peak_gpu_c is None else max(self.peak_gpu_c, gpu)
                if (cpu is not None and cpu >= abort_c) or (gpu is not None and gpu >= abort_c):
                    self.aborted = True
                    self.abort_reason = f"cpu={cpu}C gpu={gpu}C >= abort {abort_c}C during throttle"
                    print(f"[thermal] ABORT during throttle: {self.abort_reason}")
                    raise ThermalAbortError(self.abort_reason)
                if (cpu is None or cpu < resume_c) and (gpu is None or gpu < resume_c):
                    self.throttled_s += time.time() - pause_start
                    print(f"[thermal] cooled to cpu={cpu}C gpu={gpu}C - resuming")
                    return "ok"
        return "ok"

    async def pre_test_wait(self) -> None:
        """Wait for the box to cool below resume_c before starting a test (bounded)."""
        if not self.enabled:
            return
        deadline = time.time() + float(self.config.get("pretest_max_wait_s", 600.0))
        while True:
            temps = await asyncio.to_thread(self._read_temps)
            cpu, gpu = temps.get("cpu"), temps.get("gpu")
            if cpu is not None:
                self.peak_cpu_c = cpu if self.peak_cpu_c is None else max(self.peak_cpu_c, cpu)
            if gpu is not None:
                self.peak_gpu_c = gpu if self.peak_gpu_c is None else max(self.peak_gpu_c, gpu)
            resume_c = float(self.config["resume_c"])
            if (cpu is None or cpu < resume_c) and (gpu is None or gpu < resume_c):
                return
            if time.time() >= deadline:
                print(f"[thermal] pre-test wait timed out at cpu={cpu}C gpu={gpu}C - proceeding")
                return
            print(f"[thermal] pre-test wait: cpu={cpu}C gpu={gpu}C (resume at {resume_c}C)")
            await asyncio.sleep(5.0)

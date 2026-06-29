#!/usr/bin/env python3
"""
Flask server for LLM benchmark dashboard with SocketIO
"""

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO

# Add project root to path if needed to import llm_benchmark_suite
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_benchmark_suite import LLMModelBenchmark
from web.shared_llm_benchmark import SharedLLMModelBenchmark

app = Flask(__name__)
CORS(app)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "alpaca-secret-key-12984")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

benchmark = LLMModelBenchmark()
shared_llm_benchmark = SharedLLMModelBenchmark()

# Active run status tracking
active_run = {
    "status": "idle",  # "idle", "running", "cancelled", "completed", "failed"
    "type": None,  # "general" or "shared_llm"
    "current_model": None,
    "current_test": None,
    "current_category": None,
    "tests_completed": 0,
    "total_tests": 0,
    "models": [],
    "use_proxy": True,
    "results": [],
    "start_time": None,
    "saved_as": None,
}

active_run_lock = threading.Lock()
cancel_event = None
benchmark_thread = None


# Callback for progress reporting from inside the benchmark threads
def get_progress_callback(run_type):
    def callback(event, data):
        global active_run
        with active_run_lock:
            if event == "benchmark_start":
                active_run["status"] = "running"
                active_run["type"] = run_type
                active_run["models"] = data["models"]
                active_run["use_proxy"] = data["use_proxy"]
                active_run["total_tests"] = data["total_tests"]
                active_run["tests_completed"] = 0
                active_run["results"] = []
                active_run["start_time"] = data["timestamp"]
                active_run["saved_as"] = None

                socketio.emit(
                    "benchmark_start",
                    {
                        "type": run_type,
                        "models": data["models"],
                        "use_proxy": data["use_proxy"],
                        "total_tests": data["total_tests"],
                        "timestamp": data["timestamp"],
                    },
                )

            elif event == "model_start":
                active_run["current_model"] = data["model"]
                socketio.emit("model_start", {"model": data["model"]})

            elif event == "test_start":
                active_run["current_test"] = data["test_label"]
                active_run["current_category"] = data["category"]
                socketio.emit(
                    "test_start",
                    {
                        "model": data["model"],
                        "category": data["category"],
                        "test_id": data["test_id"],
                        "test_label": data["test_label"],
                    },
                )

            elif event == "test_complete":
                active_run["tests_completed"] += 1
                socketio.emit(
                    "test_complete",
                    {
                        "model": data["model"],
                        "category": data["category"],
                        "test_id": data["test_id"],
                        "test_label": data["test_label"],
                        "result": data["result"],
                        "progress": {
                            "completed": active_run["tests_completed"],
                            "total": active_run["total_tests"],
                            "percentage": round(
                                (active_run["tests_completed"] / active_run["total_tests"]) * 100
                            ),
                        },
                    },
                )

            elif event == "model_complete":
                active_run["results"].append(data["results"])
                socketio.emit(
                    "model_complete", {"model": data["model"], "results": data["results"]}
                )

            elif event == "benchmark_complete":
                active_run["status"] = data.get("status", "completed")
                active_run["current_model"] = None
                active_run["current_test"] = None
                active_run["current_category"] = None
                active_run["saved_as"] = data.get("saved_as")
                socketio.emit("benchmark_complete", data)

            elif event == "benchmark_cancelled":
                active_run["status"] = "cancelled"
                active_run["current_model"] = None
                active_run["current_test"] = None
                active_run["current_category"] = None
                socketio.emit("benchmark_cancelled", {"message": "Benchmark cancelled by user"})

    return callback


def run_general_in_thread(models, use_proxy, run_cancel_event, callback):
    global active_run
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run():
        try:
            await benchmark.run_model_benchmarks(
                models=models,
                use_proxy=use_proxy,
                progress_callback=callback,
                cancel_event=run_cancel_event,
            )
        except Exception as e:
            print(f"Error in benchmark execution: {e}")
            socketio.emit("benchmark_error", {"error": str(e)})
            with active_run_lock:
                active_run["status"] = "failed"
                active_run["current_model"] = None

    loop.run_until_complete(run())
    loop.close()


def run_shared_llm_in_thread(models, use_proxy, run_cancel_event, callback):
    global active_run
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run():
        try:
            await shared_llm_benchmark.run_shared_llm_benchmarks(
                models=models,
                use_proxy=use_proxy,
                progress_callback=callback,
                cancel_event=run_cancel_event,
            )
        except Exception as e:
            print(f"Error in SharedLLM execution: {e}")
            socketio.emit("benchmark_error", {"error": str(e)})
            with active_run_lock:
                active_run["status"] = "failed"
                active_run["current_model"] = None

    loop.run_until_complete(run())
    loop.close()


# Routes
@app.route("/")
def index():
    """Serve the dashboard HTML"""
    return render_template("index.html")


@app.route("/api/status")
def get_status():
    """Return the current active benchmark status"""
    with active_run_lock:
        return jsonify(dict(active_run))


@app.route("/api/models")
def get_models():
    """Return available models from proxies and direct ollama instances"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        direct_models = loop.run_until_complete(benchmark.discover_all_models())
        proxy_models = loop.run_until_complete(benchmark.discover_all_proxy_models())
        loop.close()

        combined = list(dict.fromkeys(direct_models + proxy_models))
        return jsonify(
            {"models": combined, "direct_models": direct_models, "proxy_models": proxy_models}
        )
    except Exception as e:
        fallback = benchmark._get_fallback_models()
        return jsonify(
            {
                "models": fallback,
                "direct_models": fallback,
                "proxy_models": fallback,
                "warning": str(e),
            }
        )


def get_host_system_metrics():
    """Fetch CPU and RAM metrics from the host via psutil.
    GPU/VRAM is intentionally omitted here — the web container has no GPU device
    access. GPU data comes from the proxy via `docker exec llama-server nvidia-smi`.
    """
    import platform

    import psutil

    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }

    # RAM usage
    try:
        vm = psutil.virtual_memory()
        info["ram_usage"] = {
            "total_gb": round(vm.total / 1e9, 2),
            "available_gb": round(vm.available / 1e9, 2),
            "used_gb": round(vm.used / 1e9, 2),
            "used_pct": vm.percent,
        }
    except Exception as e:
        print(f"Error fetching host RAM metrics: {e}")

    # CPU usage
    try:
        info["cpu_usage"] = {
            "percent": psutil.cpu_percent(interval=None),
            "load_avg": [round(x, 2) for x in os.getloadavg()] if hasattr(os, "getloadavg") else [],
        }
    except Exception as e:
        print(f"Error fetching host CPU metrics: {e}")

    return info


@app.route("/api/proxy/status")
def get_proxy_status():
    """Fetch real-time model serving, slots, client connections, and hardware metrics from proxy"""
    import httpx

    # Dynamically find the first reachable proxy URL
    proxy_url = None
    errors = []
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception as e:
            errors.append(f"{url}: {str(e)}")
            continue

    if not proxy_url:
        return jsonify(
            {
                "online": False,
                "error": f"Could not connect to any proxy endpoints. Tried: {', '.join(errors)}",
            }
        )

    try:
        with httpx.Client(timeout=3.0) as client:
            system_resp = client.get(f"{proxy_url}/admin/system")
            metrics_resp = client.get(f"{proxy_url}/admin/metrics")
            runtime_resp = client.get(f"{proxy_url}/admin/runtime")
            slots_resp = client.get(f"{proxy_url}/admin/slots")
            logs_resp = client.get(f"{proxy_url}/api/logs?limit=100")

            system_data = system_resp.json() if system_resp.status_code == 200 else {}

            # Overlay host CPU/RAM metrics (psutil in the web container).
            # GPU is NOT fetched here — the proxy already has it via docker exec.
            try:
                host_metrics = get_host_system_metrics()
                for k, v in host_metrics.items():
                    # Only overwrite if the proxy didn't supply the key or returned nothing
                    if k not in system_data or not system_data[k]:
                        system_data[k] = v
            except Exception as he:
                print(f"Failed to overlay host hardware metrics: {he}")

            # Normalize gpu_info → gpus so the dashboard JS always gets `data.system.gpus`
            gpu_raw = system_data.get("gpu_info", [])
            if isinstance(gpu_raw, list):
                system_data["gpus"] = gpu_raw
            else:
                # gpu_info is an error dict — leave gpus as empty list
                system_data["gpus"] = []

            return jsonify(
                {
                    "online": True,
                    "system": system_data,
                    "metrics": metrics_resp.json() if metrics_resp.status_code == 200 else {},
                    "runtime": runtime_resp.json() if runtime_resp.status_code == 200 else {},
                    "slots": slots_resp.json() if slots_resp.status_code == 200 else {},
                    "logs": logs_resp.json().get("logs", [])
                    if logs_resp.status_code == 200
                    else [],
                }
            )
    except Exception as e:
        return jsonify(
            {
                "online": False,
                "error": f"Failed to fetch metrics from active proxy {proxy_url}: {str(e)}",
            }
        )


@app.route("/api/run", methods=["POST"])
def start_benchmark():
    """Start standard benchmarking process"""
    global cancel_event, benchmark_thread, active_run
    with active_run_lock:
        if active_run["status"] == "running":
            return jsonify({"error": "Benchmark is already running"}), 409

    data = request.get_json() or {}
    models = data.get("models", [])
    use_proxy = data.get("use_proxy", True)

    if not models:
        return jsonify({"error": "No models specified"}), 400

    with active_run_lock:
        cancel_event = threading.Event()
        callback = get_progress_callback("general")

        active_run["status"] = "running"
        active_run["type"] = "general"
        active_run["models"] = models
        active_run["use_proxy"] = use_proxy
        active_run["tests_completed"] = 0
        active_run["total_tests"] = 0
        active_run["results"] = []
        active_run["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        active_run["saved_as"] = None

        benchmark_thread = threading.Thread(
            target=run_general_in_thread,
            args=(models, use_proxy, cancel_event, callback),
            daemon=True,
        )
        benchmark_thread.start()

    return jsonify({"status": "Benchmark started", "active_run": dict(active_run)})


@app.route("/api/run/shared_llm", methods=["POST"])
def start_shared_llm_benchmark():
    """Start SharedLLM task validation benchmarking process"""
    global cancel_event, benchmark_thread, active_run
    with active_run_lock:
        if active_run["status"] == "running":
            return jsonify({"error": "Benchmark is already running"}), 409

    data = request.get_json() or {}
    models = data.get("models", [])
    use_proxy = data.get("use_proxy", True)

    if not models:
        return jsonify({"error": "No models specified"}), 400

    with active_run_lock:
        cancel_event = threading.Event()
        callback = get_progress_callback("shared_llm")

        active_run["status"] = "running"
        active_run["type"] = "shared_llm"
        active_run["models"] = models
        active_run["use_proxy"] = use_proxy
        active_run["tests_completed"] = 0
        active_run["total_tests"] = 0
        active_run["results"] = []
        active_run["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        active_run["saved_as"] = None

        benchmark_thread = threading.Thread(
            target=run_shared_llm_in_thread,
            args=(models, use_proxy, cancel_event, callback),
            daemon=True,
        )
        benchmark_thread.start()

    return jsonify({"status": "SharedLLM Benchmark started", "active_run": dict(active_run)})


@app.route("/api/cancel", methods=["POST"])
def cancel_benchmark():
    """Cancel currently running benchmark"""
    global cancel_event, active_run
    with active_run_lock:
        if active_run["status"] != "running" or cancel_event is None:
            return jsonify({"error": "No active benchmark is running"}), 400

        cancel_event.set()
        active_run["status"] = "cancelled"
        active_run["current_model"] = None
        active_run["current_test"] = None
        active_run["current_category"] = None

        socketio.emit("benchmark_cancelled", {"message": "Cancellation requested by user"})
    return jsonify({"status": "Cancellation requested"})


@app.route("/api/results")
def get_results_list():
    """List all saved benchmark result files (both general and SharedLLM types)"""
    try:
        # Load General results
        general_dir = benchmark.RESULTS_DIR
        general_files = list(general_dir.glob("benchmarks_*.json")) if general_dir.exists() else []

        # Load SharedLLM results
        shared_dir = shared_llm_benchmark.RESULTS_DIR
        shared_files = (
            list(shared_dir.glob("shared_llm_benchmarks_*.json")) if shared_dir.exists() else []
        )

        results_list = []

        # Process general files
        for file_path in general_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                results_list.append(
                    {
                        "filename": file_path.name,
                        "type": "general",
                        "generated_at": data.get("generated_at"),
                        "benchmark_type": data.get("benchmark_type"),
                        "models_tested": data.get("models_tested"),
                        "status": data.get("status", "completed"),
                        "models": [
                            r.get("model") for r in data.get("results", []) if r.get("model")
                        ],
                        "saved_as": str(file_path),
                    }
                )
            except Exception as fe:
                print(f"Error reading file {file_path.name}: {fe}")

        # Process SharedLLM files
        for file_path in shared_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                results_list.append(
                    {
                        "filename": file_path.name,
                        "type": "shared_llm",
                        "generated_at": data.get("generated_at"),
                        "benchmark_type": data.get("benchmark_type"),
                        "models_tested": data.get("models_tested"),
                        "status": data.get("status", "completed"),
                        "models": [
                            r.get("model") for r in data.get("results", []) if r.get("model")
                        ],
                        "saved_as": str(file_path),
                    }
                )
            except Exception as fe:
                print(f"Error reading file {file_path.name}: {fe}")

        # Sort files by timestamp (newest first)
        results_list.sort(key=lambda x: x["filename"], reverse=True)
        return jsonify({"results": results_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/results/<filename>")
def get_result_detail(filename):
    """Get the detail of a specific benchmark result file"""
    try:
        filename = os.path.basename(filename)

        # Determine directory based on name prefix
        if filename.startswith("shared_llm_"):
            file_path = shared_llm_benchmark.RESULTS_DIR / filename
        else:
            file_path = benchmark.RESULTS_DIR / filename

        if not file_path.exists():
            return jsonify({"error": "Result file not found"}), 404

        with open(file_path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_models_ini_path():
    # 1. Check ROUTER_MODELS_DIR env variable
    env_dir = os.environ.get("ROUTER_MODELS_DIR")
    if env_dir:
        p = Path(env_dir) / "models.ini"
        if p.exists():
            return p
    # 2. Check for container mount /router-models/models.ini
    container_path = Path("/router-models/models.ini")
    if container_path.exists():
        return container_path
    # 3. Fallback to host workspace directory .alpaca-router/models.ini
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / ".alpaca-router" / "models.ini"


@app.route("/api/profiles")
def get_profiles():
    """Retrieve all model profile sections from models.ini"""
    ini_path = get_models_ini_path()
    if not ini_path.exists():
        return jsonify({"error": f"models.ini not found at {ini_path}"}), 404
    try:
        import configparser

        config = configparser.ConfigParser(delimiters=("=",))
        config.read(str(ini_path))

        profiles = {}
        # Include [*] defaults section if present
        if config.has_section("*"):
            profiles["*"] = dict(config["*"])

        for section in config.sections():
            if section != "*":
                profiles[section] = dict(config[section])

        return jsonify({"profiles": profiles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/profiles/save", methods=["POST"])
def save_profile():
    """Save/update a specific model profile section in models.ini"""
    data = request.get_json() or {}
    section = data.get("section")
    settings = data.get("settings")
    if not section or not isinstance(settings, dict):
        return jsonify({"error": "Invalid payload: 'section' and 'settings' required"}), 400

    ini_path = get_models_ini_path()
    try:
        import configparser

        config = configparser.ConfigParser(delimiters=("=",))
        config.read(str(ini_path))

        if not config.has_section(section):
            config.add_section(section)

        for k, v in settings.items():
            if v is True or v == "true":
                config[section][k] = "true"
            elif v is False or v == "false":
                config[section][k] = "false"
            elif v is None or v == "":
                config[section].pop(k, None)
            else:
                config[section][k] = str(v)

        # Write back to file
        with open(ini_path, "w") as f:
            config.write(f)
        try:
            os.chmod(ini_path, 0o666)
        except Exception:
            pass

        return jsonify(
            {"status": "success", "message": f"Successfully updated section [{section}]"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/profiles/delete", methods=["POST"])
def delete_profile():
    """Delete a specific model profile section from models.ini"""
    data = request.get_json() or {}
    section = data.get("section")
    if not section:
        return jsonify({"error": "Invalid payload: 'section' required"}), 400
    if section == "*":
        return jsonify({"error": "Cannot delete global defaults section [*]"}), 400

    ini_path = get_models_ini_path()
    try:
        import configparser

        config = configparser.ConfigParser(delimiters=("=",))
        config.read(str(ini_path))

        if config.has_section(section):
            config.remove_section(section)
            with open(ini_path, "w") as f:
                config.write(f)
            try:
                os.chmod(ini_path, 0o666)
            except Exception:
                pass
            return jsonify(
                {"status": "success", "message": f"Successfully deleted section [{section}]"}
            )
        else:
            return jsonify({"error": f"Section [{section}] not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/logs/download")
def download_logs():
    """Download historical proxy logs as a text file"""
    import httpx

    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version")
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if not proxy_url:
        return "Proxy server offline", 503

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{proxy_url}/api/logs?limit=5000")
            if resp.status_code == 200:
                log_data = resp.json().get("logs", [])
                text = "\n".join(log_data)
                return (
                    text,
                    200,
                    {
                        "Content-Type": "text/plain",
                        "Content-Disposition": "attachment; filename=alpaca_proxy_system.log",
                    },
                )
            return "Failed to fetch logs from proxy", 500
    except Exception as e:
        return f"Failed to retrieve logs: {str(e)}", 500


@app.route("/api/proxy/restart", methods=["POST"])
def restart_proxy_services():
    """Trigger background restart of llama-server and alpaca-proxy docker containers"""
    import httpx

    # 1. Try to find the active proxy and call /admin/restart
    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if proxy_url:
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(f"{proxy_url}/admin/restart", params={"restart_proxy": "true"})
                if resp.status_code == 200:
                    return jsonify(
                        {
                            "status": "success",
                            "message": "Backend restart sequence initiated via proxy API.",
                        }
                    )
        except Exception as e:
            print(f"Proxy restart request failed: {e}")

    # 2. Fallback to local subprocess execution (e.g. host development mode)
    def run_restart_subprocess():
        time.sleep(0.5)
        try:
            import subprocess

            subprocess.run(["docker", "restart", "llama-server"], capture_output=True, text=True)
            subprocess.run(["docker", "restart", "alpaca-proxy"], capture_output=True, text=True)
        except Exception as e:
            print(f"Subprocess restart failed: {e}")

    threading.Thread(target=run_restart_subprocess, daemon=True).start()
    return jsonify(
        {"status": "success", "message": "Backend restart sequence initiated via fallback."}
    )


@app.route("/api/requests")
def get_active_requests():
    """Fetch active and recently completed requests from the proxy"""
    import httpx

    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if not proxy_url:
        return jsonify({"error": "Could not connect to any proxy endpoints."}), 503

    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{proxy_url}/admin/requests")
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                return jsonify(
                    {"error": f"Proxy returned status {resp.status_code}"}
                ), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to fetch requests telemetry from proxy: {str(e)}"}), 500


@app.route("/api/requests/clear", methods=["POST"])
def clear_completed_requests():
    """Clear completed requests buffer in the proxy"""
    import httpx

    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if not proxy_url:
        return jsonify({"error": "Could not connect to any proxy endpoints."}), 503

    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.post(f"{proxy_url}/admin/requests/clear")
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                return jsonify(
                    {"error": f"Proxy returned status {resp.status_code}"}
                ), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to clear requests history in proxy: {str(e)}"}), 500


@app.route("/api/requests/cancel", methods=["POST"])
def cancel_stuck_request():
    """Cancel a stuck/active request in the proxy (searches all proxies)"""
    import httpx

    data = request.get_json() or {}
    request_id = data.get("request_id")

    if not request_id:
        return jsonify({"error": "request_id is required"}), 400

    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code != 200:
                    continue

                resp = client.delete(f"{url}/admin/requests/{request_id}")
                if resp.status_code == 200:
                    return jsonify(resp.json())
        except Exception:
            continue

    return jsonify({"error": "Could not connect to any proxy endpoints."}), 503


@app.route("/api/requests/resubmit/<string:request_id>", methods=["POST"])
def resubmit_stuck_request(request_id):
    """Resubmit a stuck request by extracting its prompt and sending to the model (searches all proxies)"""
    import httpx

    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code != 200:
                    continue

                # Try persistent resubmit storage first (never rotates out)
                resp = client.get(f"{url}/admin/resubmit/{request_id}")
                if resp.status_code == 200:
                    req = resp.json()
                else:
                    # Fall back to active + completed requests
                    resp = client.get(f"{url}/admin/requests")
                    if resp.status_code != 200:
                        continue
                    all_data = resp.json()
                    all_requests = all_data.get("active_requests", []) + all_data.get(
                        "completed_requests", []
                    )
                    req = None
                    for r in all_requests:
                        if r.get("request_id") == request_id:
                            req = r
                            break

                if not req:
                    continue

                req_type = req.get("type", "unknown")
                model = req.get("model", "")
                prompt = req.get("prompt", "")

                if not prompt:
                    continue

                if req_type in ("ollama_chat", "openai_chat"):
                    import re

                    role_pattern = re.compile(r"^([A-Z_]+):\s*(.*)", re.MULTILINE)
                    matches = role_pattern.findall(prompt)

                    messages = []
                    for role, content in matches:
                        r = role.lower()
                        if r in ("system", "user", "assistant"):
                            messages.append({"role": r, "content": content})
                        else:
                            messages.append({"role": "user", "content": f"{role}: {content}"})

                    # Strip trailing assistant messages (model response stored in prompt)
                    while len(messages) > 1 and messages[-1]["role"] == "assistant":
                        messages.pop()

                    if messages:
                        body = {
                            "model": model,
                            "messages": messages,
                            "stream": False,
                            "keep_alive": -1,
                        }
                        endpoint = f"{url}/api/chat"
                    else:
                        body = {
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "keep_alive": -1,
                        }
                        endpoint = f"{url}/api/chat"
                elif req_type == "ollama_generate":
                    body = {"model": model, "prompt": prompt, "stream": False, "keep_alive": -1}
                    endpoint = f"{url}/api/generate"
                elif req_type == "openai_generate":
                    body = {"model": model, "prompt": prompt, "stream": False, "keep_alive": -1}
                    endpoint = f"{url}/v1/completions"
                else:
                    body = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "keep_alive": -1,
                    }
                    endpoint = f"{url}/api/chat"

                resp = client.post(endpoint, json=body, timeout=60.0)
                if resp.status_code != 200:
                    return jsonify(
                        {"error": f"Proxy returned status {resp.status_code}"}
                    ), resp.status_code

                result = resp.json()

                try:
                    client.delete(f"{url}/admin/requests/{request_id}")
                except Exception:
                    pass

                return jsonify({"status": "resubmitted", "result": result})
        except Exception:
            continue

    return jsonify({"error": "Request not found in any proxy"}), 404


@app.route("/api/usage")
def get_model_usage():
    """Get model usage statistics from the proxy"""
    import httpx

    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if not proxy_url:
        return jsonify({"error": "Could not connect to any proxy endpoints."}), 503

    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{proxy_url}/admin/usage")
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                return jsonify(
                    {"error": f"Proxy returned status {resp.status_code}"}
                ), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to fetch model usage stats: {str(e)}"}), 500


@app.route("/api/models/switch", methods=["POST"])
def switch_model():
    """Switch to a different model via the proxy"""
    import httpx

    data = request.get_json() or {}
    model = data.get("model")
    if not model:
        return jsonify({"error": "model is required"}), 400

    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if not proxy_url:
        return jsonify({"error": "Could not connect to any proxy endpoints."}), 503

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{proxy_url}/admin/models/switch", json={"model": model})
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                return jsonify({"error": resp.text}), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to switch model: {str(e)}"}), 500


@app.route("/api/models/unload", methods=["POST"])
def unload_model():
    """Unload a model via the proxy"""
    import httpx

    data = request.get_json() or {}
    model = data.get("model")
    if not model:
        return jsonify({"error": "model is required"}), 400

    proxy_url = None
    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version", timeout=1.0)
                if resp.status_code == 200:
                    proxy_url = url
                    break
        except Exception:
            continue

    if not proxy_url:
        return jsonify({"error": "Could not connect to any proxy endpoints."}), 503

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{proxy_url}/admin/models/unload", json={"model": model})
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                return jsonify({"error": resp.text}), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to unload model: {str(e)}"}), 500


@socketio.on("connect")
def handle_connect():
    print("Client connected")
    with active_run_lock:
        socketio.emit("sync_status", dict(active_run))


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)

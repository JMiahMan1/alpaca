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


# Load .env file if present
def load_dotenv_custom():
    base_dir = Path(__file__).resolve().parent.parent
    dotenv_path = base_dir / ".env"
    if dotenv_path.exists():
        with open(dotenv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    os.environ.setdefault(key, value)

load_dotenv_custom()

import logging

DEBUG_LOGGING = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes") or os.getenv("DEBUG_LOGGING", "0").lower() in ("1", "true", "yes")
if not DEBUG_LOGGING:
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

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

puller_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "alpaca-puller.py"
)
active_pulls = {}
active_pulls_lock = threading.Lock()


def run_puller_thread(model_name, source, local_name, no_resume=False):
    global active_pulls
    import subprocess

    cmd = [sys.executable, puller_path, "pull", model_name]
    if source and source != "auto":
        cmd += ["--source", source]
    if local_name:
        cmd += ["--name", local_name]
    if no_resume:
        cmd += ["--no-resume"]

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            # Check for stop/cancel events
            with active_pulls_lock:
                if model_name in active_pulls:
                    status = active_pulls[model_name].get("status", "running")
                    if status == "cancelled":
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        socketio.emit("pull_status", {
                            "model": model_name,
                            "status": "cancelled"
                        })
                        return
                    elif status == "stopping":
                        active_pulls[model_name]["status"] = "stopping"
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        active_pulls[model_name]["status"] = "stopped"
                        socketio.emit("pull_status", {
                            "model": model_name,
                            "status": "stopped"
                        })
                        # Clean up stop marker so it doesn't interfere with future pulls
                        stop_dir = Path(os.getenv("ROUTER_MODELS_DIR", ".alpaca-router")) / ".alpaca-stop"
                        stop_file = stop_dir / f"{model_name.replace('/', '_').replace(':', '_')}"
                        stop_file.unlink(missing_ok=True)
                        return
            if line:
                line_str = line.rstrip()
                with active_pulls_lock:
                    if model_name in active_pulls:
                        active_pulls[model_name]["logs"].append(line_str)
                        if len(active_pulls[model_name]["logs"]) > 1000:
                            active_pulls[model_name]["logs"].pop(0)
                socketio.emit("pull_log", {"model": model_name, "line": line_str})

        rc = process.poll()
        with active_pulls_lock:
            if model_name in active_pulls:
                status = active_pulls[model_name].get("status", "running")
                if status == "stopping":
                    active_pulls[model_name]["status"] = "stopped"
                    socketio.emit("pull_status", {"model": model_name, "status": "stopped"})
                    stop_dir = Path(os.getenv("ROUTER_MODELS_DIR", ".alpaca-router")) / ".alpaca-stop"
                    stop_file = stop_dir / f"{model_name.replace('/', '_').replace(':', '_')}"
                    stop_file.unlink(missing_ok=True)
                    return

        if rc == 0:
            socketio.emit("pull_status", {"model": model_name, "status": "success"})
        else:
            socketio.emit(
                "pull_status",
                {"model": model_name, "status": "failed", "error": f"Exit code {rc}"},
            )

    except Exception as e:
        socketio.emit(
            "pull_status", {"model": model_name, "status": "failed", "error": str(e)}
        )
    finally:
        with active_pulls_lock:
            if model_name in active_pulls:
                del active_pulls[model_name]


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
                active_run["current_test"] = None
                active_run["current_category"] = None
        finally:
            # Guarantee we never stay stuck in "running" state
            with active_run_lock:
                if active_run["status"] == "running":
                    print("[benchmark] Thread exiting with status still 'running' — forcing to 'completed'")
                    active_run["status"] = "completed"
                    active_run["current_model"] = None
                    active_run["current_test"] = None
                    active_run["current_category"] = None
                    socketio.emit("benchmark_complete", {"status": "completed", "saved_as": active_run.get("saved_as")})

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
                active_run["current_test"] = None
                active_run["current_category"] = None
        finally:
            # Guarantee we never stay stuck in "running" state
            with active_run_lock:
                if active_run["status"] == "running":
                    print("[benchmark] SharedLLM thread exiting with status still 'running' — forcing to 'completed'")
                    active_run["status"] = "completed"
                    active_run["current_model"] = None
                    active_run["current_test"] = None
                    active_run["current_category"] = None
                    socketio.emit("benchmark_complete", {"status": "completed", "saved_as": active_run.get("saved_as")})

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

        # Write to profile.json as well so reindexing doesn't discard overrides
        if section != "*":
            try:
                profile_path = ini_path.parent / f"{section}.profile.json"
                existing_profile = {}
                if profile_path.exists():
                    try:
                        with open(profile_path, "r") as pf:
                            existing_profile = json.load(pf)
                    except Exception:
                        pass
                
                for k, v in settings.items():
                    if v is None or v == "":
                        existing_profile.pop(k, None)
                    else:
                        existing_profile[k] = v
                
                with open(profile_path, "w") as pf:
                    json.dump(existing_profile, pf, indent=4)
                try:
                    os.chmod(profile_path, 0o666)
                except Exception:
                    pass
            except Exception as pe:
                print(f"Failed to save profile json overlay: {pe}")

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
            # Remove profile.json if it exists
            try:
                profile_path = ini_path.parent / f"{section}.profile.json"
                if profile_path.exists():
                    os.remove(profile_path)
            except Exception as pe:
                print(f"Failed to remove profile json file: {pe}")

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



@app.route("/api/telemetry/history")
def get_telemetry_history():
    """Fetch telemetry history for a specific model"""
    model = request.args.get("model")
    limit = request.args.get("limit", 100, type=int)
    
    if not model:
        # Try to find the currently active model from the proxy
        try:
            import httpx
            proxy_url = None
            for url in benchmark.PROXY_SERVER_URLS:
                with httpx.Client(timeout=0.5) as client:
                    resp = client.get(f"{url}/api/version")
                    if resp.status_code == 200:
                        proxy_url = url
                        break
            if proxy_url:
                with httpx.Client(timeout=1.0) as client:
                    resp = client.get(f"{proxy_url}/admin/runtime")
                    if resp.status_code == 200:
                        model = resp.json().get("active_model")
        except Exception:
            pass

    if not model:
        model = "system_idle"

    # sanitize model name just like telemetry_monitor.py
    import re
    telemetry_dir = Path(os.getenv("TELEMETRY_DIR", "data/telemetry"))
    sanitized_model = re.sub(r"[^\w\-.\.]", "_", model)
    sanitized_model_lower = sanitized_model.lower()
    
    def find_telemetry_file(filename):
        """Try multiple directories and flexible matching for telemetry files."""
        candidates = [
            telemetry_dir / filename,
            Path("/app/data/telemetry") / filename,
            Path("web").parent / "data" / "telemetry" / filename,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    # 1. Try exact match
    log_file = find_telemetry_file(f"{sanitized_model}.jsonl")
    
    # 2. Try case-insensitive match
    if log_file is None:
        log_file = find_telemetry_file(f"{sanitized_model_lower}.jsonl")
    
    # 3. Search all telemetry files for one containing the model name
    if log_file is None:
        for search_dir in [telemetry_dir, Path("/app/data/telemetry"), Path("web").parent / "data" / "telemetry"]:
            if search_dir.exists():
                for f in search_dir.glob("*.jsonl"):
                    if sanitized_model_lower in f.stem.lower():
                        log_file = f
                        break
            if log_file:
                break

    if log_file is None or not log_file.exists():
        return jsonify({"model": model, "history": []})
        
    points = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    points.append(json.loads(line))
        return jsonify({"model": model, "history": points[-limit:]})
    except Exception as e:
        return jsonify({"error": f"Failed to read telemetry: {str(e)}"}), 500


@app.route("/api/telemetry/recommendations")
def get_telemetry_recommendations():
    """Run configuration analyzer and fetch recommendations"""
    model = request.args.get("model")
    strategy = request.args.get("strategy", "performance")
    
    if not model:
        # Default to currently loaded model
        try:
            import httpx
            proxy_url = None
            for url in benchmark.PROXY_SERVER_URLS:
                with httpx.Client(timeout=0.5) as client:
                    resp = client.get(f"{url}/api/version")
                    if resp.status_code == 200:
                        proxy_url = url
                        break
            if proxy_url:
                with httpx.Client(timeout=1.0) as client:
                    resp = client.get(f"{proxy_url}/admin/runtime")
                    if resp.status_code == 200:
                        model = resp.json().get("active_model")
        except Exception:
            pass
            
    if not model:
        return jsonify({
            "status": "insufficient_data",
            "model_alias": "None",
            "detected_issues": ["No active model running, and no model specified."],
            "recommendations": {},
            "explanation": "Please load a model or specify one via '?model=name'."
        })
        
    import re
    sanitized_model = re.sub(r"[^\w\-.\.]", "_", model)
    try:
        from analyzer import analyze_telemetry
        perf_first = (strategy == "performance")
        analysis = analyze_telemetry(sanitized_model, performance_first=perf_first)
        return jsonify(analysis)
    except Exception as e:
        # Fallback analysis if importer/analyzer fails
        return jsonify({
            "status": "error",
            "model_alias": model,
            "detected_issues": [f"Tuning analyzer engine failed: {str(e)}"],
            "recommendations": {},
            "explanation": "Ensure analyzer.py is mounted correctly in the web container path."
        })


@app.route("/api/telemetry/recommendations/apply", methods=["POST"])
def apply_telemetry_recommendations():
    """Apply recommendations to the model profile overlay"""
    data = request.get_json() or {}
    model = data.get("model")
    recommendations = data.get("recommendations", {})
    
    if not model:
        return jsonify({"error": "model is required"}), 400
    if not recommendations:
        return jsonify({"error": "no recommendations provided"}), 400
        
    import re
    sanitized_model = re.sub(r"[^\w\-.\.]", "_", model)
    router_models_dir = Path(os.getenv("ROUTER_MODELS_DIR", "/router-models"))
    
    # 1. Delimiter-agnostic scan to find the correct GGUF file stem
    profile_stem = sanitized_model
    
    def clean_str(s):
        return s.replace("/", "").replace("_", "").replace("-", "").lower()
    clean_target = clean_str(model)
    
    target_dir = router_models_dir
    if not target_dir.exists():
        target_dir = Path("data")
        if not target_dir.exists():
            target_dir = Path("web").parent / ".alpaca-router"
            
    if target_dir.exists():
        for entry in target_dir.iterdir():
            if entry.suffix == ".gguf":
                clean_sec = clean_str(entry.stem)
                if clean_target in clean_sec or clean_sec in clean_target or clean_target.replace("latest", "") in clean_sec:
                    profile_stem = entry.stem
                    break
                    
    profile_path = target_dir / f"{profile_stem}.profile.json"
            
    try:
        profile_data = {}
        if profile_path.exists():
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json.load(f)
                
        profile_data.update(recommendations)
        
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)
            
        # Try to regenerate models.ini via puller
        try:
            from alpaca_puller import update_models_ini
            update_models_ini()
            ini_msg = "and regenerated models.ini"
        except Exception:
            ini_msg = "but could not regenerate models.ini automatically"
            
        return jsonify({
            "status": "success",
            "message": f"Applied tuning properties for {model} {ini_msg}.",
            "applied": recommendations
        })
    except Exception as e:
        return jsonify({"error": f"Failed to apply tuning properties: {str(e)}"}), 500


@app.route("/api/routing/matrix", methods=["GET", "POST"])
def get_or_post_routing_matrix():
    """GET current model capability routing matrix, or POST modifications to it"""
    matrix_file = Path("data/routing_matrix.json")
    if not matrix_file.parent.exists():
        matrix_file = Path("web").parent / "data" / "routing_matrix.json"
        
    # Default routing matrix template
    default_matrix = {
        "fast_chat": {
            "model": "qwen3--8b",
            "description": "Sub-second latency chat for voice assistant or general conversation.",
            "min_tps": 40.0,
            "max_ttft_ms": 250,
            "reasoning_required": False
        },
        "complex_coding": {
            "model": "qwen2.5-coder--7b",
            "description": "Accurate syntax completions, code editing, and structural debugging.",
            "min_tps": 20.0,
            "max_ttft_ms": 500,
            "reasoning_required": False
        },
        "reasoning": {
            "model": "gemma-4-E2B-it-uncensored-GGUF--gemma-4-E2B-it-uncensored-Q4_K_M",
            "description": "Deep thinking, logical reasoning, multi-step problem solving, math/science.",
            "min_tps": 15.0,
            "max_ttft_ms": 800,
            "reasoning_required": True
        },
        "summarization": {
            "model": "gemma-4-12b-fable5",
            "description": "Document parsing, entity extraction, context summaries, and long context tasks.",
            "min_tps": 30.0,
            "max_ttft_ms": 300,
            "reasoning_required": False
        }
    }
    
    if request.method == "POST":
        data = request.get_json() or {}
        try:
            matrix_file.parent.mkdir(parents=True, exist_ok=True)
            with open(matrix_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return jsonify({"status": "success", "message": "Routing matrix updated successfully.", "matrix": data})
        except Exception as e:
            return jsonify({"error": f"Failed to save routing matrix: {str(e)}"}), 500
            
    # GET method
    if matrix_file.exists():
        try:
            with open(matrix_file, "r", encoding="utf-8") as f:
                matrix = json.load(f)
                return jsonify(matrix)
        except Exception:
            return jsonify(default_matrix)
    else:
        return jsonify(default_matrix)


@app.route("/api/routing/optimal")
def get_optimal_model():
    """Query endpoint for SharedLLM routing decision"""
    task = request.args.get("task", "fast_chat")
    min_tps = request.args.get("min_tps", type=float)
    max_ttft_ms = request.args.get("max_ttft_ms", type=int)
    reasoning_required = request.args.get("reasoning_required")
    if reasoning_required is not None:
        reasoning_required = reasoning_required.lower() in ("true", "1", "yes")

    # Load matrix config
    matrix_file = Path("data/routing_matrix.json")
    if not matrix_file.parent.exists():
        matrix_file = Path("web").parent / "data" / "routing_matrix.json"
        
    matrix = {}
    if matrix_file.exists():
        try:
            with open(matrix_file, "r", encoding="utf-8") as f:
                matrix = json.load(f)
        except Exception:
            pass
            
    # Match criteria
    task_config = matrix.get(task, {})
    
    # Fallbacks/overrides based on request params
    optimal_model = task_config.get("model", "qwen3--8b")
    explanation = f"Matched to configured model for task type '{task}'."
    
    # If the caller requests specific speed overrides, validate models based on benchmarks
    try:
        from analyzer import load_latest_benchmark
        # Try loading benchmark for current optimal
        bench = load_latest_benchmark(optimal_model)
        if bench:
            tps = bench.get("avg_tokens_per_sec", 0.0)
            ttft = bench.get("avg_ttft_ms", 0)
            
            # If configured model fails constraints, search alternatives
            if (min_tps and tps < min_tps) or (max_ttft_ms and ttft > max_ttft_ms):
                explanation = f"Configured model '{optimal_model}' did not meet constraints (Benchmarked: {tps} TPS, {ttft}ms TTFT). Searching fallbacks..."
                
                best_model = optimal_model
                best_score = -9999.0
                
                # Check other models in benchmark directory
                from analyzer import BENCHMARK_DIR
                if BENCHMARK_DIR.exists():
                    for path in BENCHMARK_DIR.glob("*.json"):
                        try:
                            with open(path, "r") as f:
                                data = json.load(f)
                                for res in data.get("results", []):
                                    m_tps = res.get("avg_tokens_per_sec", 0.0)
                                    m_ttft = res.get("avg_ttft_ms", 9999)
                                    m_model = res.get("model")
                                    
                                    meets_tps = (not min_tps) or (m_tps >= min_tps)
                                    meets_ttft = (not max_ttft_ms) or (m_ttft <= max_ttft_ms)
                                    
                                    score = m_tps - (m_ttft / 10.0)
                                    if meets_tps and meets_ttft and score > best_score:
                                        best_score = score
                                        best_model = m_model
                        except Exception:
                            continue
                
                if best_model != optimal_model:
                    explanation += f" Routed to '{best_model}' as optimal alternative."
                    optimal_model = best_model
    except Exception as e:
        explanation += f" (Benchmark validation skipped: {str(e)})"

    return jsonify({
        "optimal_model": optimal_model,
        "task": task,
        "explanation": explanation,
        "fallback_model": "qwen3--8b" if optimal_model != "qwen3--8b" else "qwen2.5-coder--7b"
    })


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


@app.route("/api/vram/clear", methods=["POST"])
def clear_vram():
    """Clear VRAM via the proxy"""
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
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(f"{proxy_url}/admin/vram/clear")
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                return jsonify({"error": resp.text}), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to clear VRAM: {str(e)}"}), 500


@app.route("/api/models/delete", methods=["POST"])
def delete_model():
    """Delete a model and clean up blobs via the proxy"""
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
            resp = client.post(f"{proxy_url}/admin/models/delete", json={"model": model})
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                try:
                    err_msg = resp.json().get("detail", resp.text)
                except Exception:
                    err_msg = resp.text
                return jsonify({"error": err_msg}), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to delete model: {str(e)}"}), 500


@app.route("/api/models/search", methods=["POST"])
def search_models():
    """Search for models on Ollama Registry and Hugging Face"""
    import re
    from html import unescape

    import httpx

    data = request.get_json() or {}
    query = data.get("query")
    source = data.get("source", "all")  # "ollama", "huggingface", or "all"

    if not query:
        return jsonify({"error": "query is required"}), 400

    results = []

    # 1. Ollama Search
    if source in ("ollama", "all"):
        try:
            url = f"https://ollama.com/search?q={query}"
            resp = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10.0)
            if resp.status_code == 200:
                html_content = resp.text
                pattern = r'href="/library/([^"]+)"[^>]*>.*?<span[^>]*>([^<]+)</span>.*?<p class="[^"]*break-words[^"]*">([^<]+)</p>'
                matches = re.findall(pattern, html_content, re.DOTALL)
                for library_name, span_name, desc in matches:
                    clean_desc = unescape(desc.strip().replace("\n", " "))
                    results.append(
                        {
                            "name": library_name.strip(),
                            "description": clean_desc,
                            "source": "ollama",
                        }
                    )
        except Exception as e:
            print(f"Ollama search error: {e}")

    # 2. Hugging Face Search
    if source in ("huggingface", "all"):
        try:
            url = f"https://huggingface.co/api/models?search={query}&filter=gguf&limit=20"
            resp = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10.0)
            if resp.status_code == 200:
                hf_data = resp.json()
                for model_item in hf_data:
                    model_id = model_item.get("id")
                    downloads = model_item.get("downloads", 0)
                    likes = model_item.get("likes", 0)
                    tags = model_item.get("tags", [])
                    qwen_tags = [t for t in tags if t != "gguf"][:3]
                    tag_str = ", ".join(qwen_tags) if qwen_tags else "GGUF"

                    desc = f"GGUF repository by {model_item.get('author', 'HF author')}. Downloads: {downloads:,} | Likes: {likes:,} | Tags: {tag_str}"

                    results.append(
                        {
                            "name": model_id,
                            "description": desc,
                            "source": "huggingface",
                        }
                    )
        except Exception as e:
            print(f"Hugging Face search error: {e}")

    # 3. Precise HF Repo Lookup if query has a slash
    if source in ("huggingface", "all") and "/" in query:
        try:
            precise_url = f"https://huggingface.co/api/models/{query}"
            token = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")
            headers = {"User-Agent": "Mozilla/5.0"}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            resp = httpx.get(precise_url, headers=headers, timeout=5.0)
            if resp.status_code == 200:
                model_item = resp.json()
                model_id = model_item.get("id")
                # Avoid duplicate entries in list
                if not any(r["name"] == model_id for r in results):
                    downloads = model_item.get("downloads", 0)
                    likes = model_item.get("likes", 0)
                    tags = model_item.get("tags", [])
                    qwen_tags = [t for t in tags if t != "gguf"][:3]
                    tag_str = ", ".join(qwen_tags) if qwen_tags else "GGUF"
                    desc = f"[Direct Match] GGUF repository by {model_item.get('author', 'HF author')}. Downloads: {downloads:,} | Likes: {likes:,} | Tags: {tag_str}"
                    results.insert(0, {
                        "name": model_id,
                        "description": desc,
                        "source": "huggingface"
                    })
        except Exception as e:
            print(f"Precise HF lookup error: {e}")

    return jsonify({"results": results})


@app.route("/api/models/huggingface/files", methods=["GET"])
def get_hf_files():
    """List .gguf files in a Hugging Face repository"""
    repo = request.args.get("repo")
    if not repo:
        return jsonify({"error": "repo is required"}), 400

    import httpx

    token = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")
    headers = {"User-Agent": "Mozilla/5.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://huggingface.co/api/models/{repo}"
    try:
        resp = httpx.get(url, headers=headers, timeout=15.0)
        if resp.status_code != 200:
            return (
                jsonify(
                    {
                        "error": f"Failed to fetch model info from Hugging Face: {resp.text}"
                    }
                ),
                resp.status_code,
            )

        model_info = resp.json()
        siblings = model_info.get("siblings", [])
        gguf_files = []
        for s in siblings:
            fname = s.get("rfilename", "")
            if fname.endswith(".gguf"):
                size = s.get("size")
                size_str = ""
                if size:
                    if size > 1024**3:
                        size_str = f"{size / (1024**3):.2f} GB"
                    else:
                        size_str = f"{size / (1024**2):.1f} MB"
                gguf_files.append({"filename": fname, "size": size_str})
        return jsonify({"files": gguf_files})
    except Exception as e:
        return jsonify({"error": f"Error fetching Hugging Face files: {str(e)}"}), 500


@app.route("/api/models/ollama/tags", methods=["GET"])
def get_ollama_model_tags():
    """List available tags/sizes for a model in the Ollama library"""
    model = request.args.get("model")
    if not model:
        return jsonify({"error": "model is required"}), 400

    import re

    import httpx

    url = f"https://ollama.com/library/{model}/tags"
    try:
        resp = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15.0)
        if resp.status_code != 200:
            return (
                jsonify(
                    {
                        "error": f"Failed to fetch tags from Ollama Library: {resp.text}"
                    }
                ),
                resp.status_code,
            )

        html = resp.text
        # Matches href="/library/model:tag"
        pattern = rf'href="/library/{re.escape(model)}:([^"]+)"'
        tags = re.findall(pattern, html)

        # Deduplicate and keep order
        unique_tags = []
        seen = set()
        for t in tags:
            t_clean = t.strip()
            if t_clean not in seen:
                seen.add(t_clean)
                unique_tags.append(t_clean)

        # Fallback to general tag parsing if specific model name prefix did not match
        if not unique_tags:
            fallback_pattern = r'href="/library/[^"]+:([^"]+)"'
            generic_tags = re.findall(fallback_pattern, html)
            for t in generic_tags:
                t_clean = t.strip()
                if t_clean not in seen:
                    seen.add(t_clean)
                    unique_tags.append(t_clean)

        return jsonify({"tags": unique_tags})
    except Exception as e:
        return jsonify({"error": f"Error fetching Ollama model tags: {str(e)}"}), 500



@app.route("/api/models/pull", methods=["POST"])
def trigger_model_pull():
    """Trigger a background pull of a model via alpaca-puller.py"""
    data = request.get_json() or {}
    model = data.get("model")
    source = data.get("source", "auto")
    local_name = data.get("local_name")
    no_resume = data.get("no_resume", False)

    if not model:
        return jsonify({"error": "model is required"}), 400

    with active_pulls_lock:
        if model in active_pulls:
            return (
                jsonify({"error": f"Model {model} is already being downloaded."}),
                409,
            )
        active_pulls[model] = {
            "model": model,
            "source": source,
            "local_name": local_name or "",
            "status": "running",
            "logs": []
        }

    t = threading.Thread(
        target=run_puller_thread, args=(model, source, local_name, no_resume), daemon=True
    )
    t.start()

    return jsonify(
        {
            "status": "pulling_started",
            "message": f"Started pulling model {model} in the background.",
        }
    )


@app.route("/api/models/pulls/active", methods=["GET"])
def get_active_pulls():
    """Retrieve currently active downloads and their logs"""
    with active_pulls_lock:
        return jsonify({
            "active_pulls": {
                k: {
                    "model": v["model"],
                    "source": v["source"],
                    "local_name": v.get("local_name", ""),
                    "status": v.get("status", "running"),
                    "logs": v["logs"]
                }
                for k, v in active_pulls.items()
            }
        })


@app.route("/api/models/pulls/<model_id>/stop", methods=["POST"])
def stop_pull(model_id):
    """Stop/pause a running pull by creating a stop marker file."""
    with active_pulls_lock:
        if model_id not in active_pulls:
            return jsonify({"error": "Pull not found"}), 404
        pull = active_pulls[model_id]
        if pull.get("status") not in ("running", "paused"):
            return jsonify({"error": f"Pull is {pull.get('status', 'unknown')}, cannot stop"}), 400
        pull["status"] = "stopping"

    # Create stop marker file that alpaca-puller checks
    stop_dir = Path(os.getenv("ROUTER_MODELS_DIR", ".alpaca-router")) / ".alpaca-stop"
    stop_dir.mkdir(parents=True, exist_ok=True)
    stop_file = stop_dir / f"{model_id.replace('/', '_').replace(':', '_')}"
    stop_file.write_text(str(time.time()))
    print(f"Stop marker created: {stop_file}")

    return jsonify({
        "status": "stopping",
        "message": f"Stopping pull for {pull['model']}..."
    })


@app.route("/api/models/pulls/<model_id>/cancel", methods=["POST"])
def cancel_pull(model_id):
    """Cancel a pull: stop it and remove any partial downloads."""
    with active_pulls_lock:
        if model_id not in active_pulls:
            return jsonify({"error": "Pull not found"}), 404
        pull = active_pulls[model_id]
        if pull.get("status") == "cancelled":
            return jsonify({"error": "Pull already cancelled"}), 400
        pull["status"] = "cancelled"

    # Remove stop marker for this specific model to prevent interfering with future pulls
    stop_dir = Path(os.getenv("ROUTER_MODELS_DIR", ".alpaca-router")) / ".alpaca-stop"
    stop_file = stop_dir / f"{model_id.replace('/', '_').replace(':', '_')}"
    stop_file.unlink(missing_ok=True)
    print(f"Cancelled pull for {pull['model']}, partial downloads will be cleaned up on next restart.")

    return jsonify({
        "status": "cancelled",
        "message": f"Pull for {pull['model']} cancelled."
    })


@socketio.on("connect")
def handle_connect():
    print("Client connected")
    with active_run_lock:
        socketio.emit("sync_status", dict(active_run))


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)

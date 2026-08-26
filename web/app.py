#!/usr/bin/env python3
"""
Flask server for LLM benchmark dashboard with SocketIO
"""

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import re
import secrets
import select
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, session, url_for
from flask_cors import CORS
from flask_socketio import SocketIO

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import contextlib
import io
import tarfile
import uuid

import docker

from llm_benchmark_suite import LLMModelBenchmark
from multistep_benchmark import MultiStepBenchmark
from sandbox_exec import serve_app, serve_ui, stop_serve, ui_exec, ui_restart, ui_screenshot, ui_status
from web.model_tracker import model_tracker
from web.shared_llm_benchmark import SharedLLMModelBenchmark


# Load .env file if present
def load_dotenv_custom():
    base_dir = Path(__file__).resolve().parent.parent
    dotenv_path = base_dir / ".env"
    if dotenv_path.exists():
        with open(dotenv_path) as f:
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

DEBUG_LOGGING = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes") or os.getenv("DEBUG_LOGGING", "0").lower() in (
    "1",
    "true",
    "yes",
)
if not DEBUG_LOGGING:
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

app = Flask(__name__)
CORS(app)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


@app.after_request
def add_cache_headers(response):
    """Prevent caching of static JS/CSS and API endpoints to avoid stale data/code."""
    if (request.path.startswith("/static/") and request.path.endswith((".js", ".css"))) or request.path.startswith(
        "/api/"
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


benchmark = LLMModelBenchmark()
shared_llm_benchmark = SharedLLMModelBenchmark()
multistep_benchmark = MultiStepBenchmark()

puller_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "alpaca-puller.py")
PROXY_URL = os.environ.get("PROXY_URL", "http://host.docker.internal:11434")
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://llama-server:8080")
AUDIO_SERVER_URL = os.environ.get("AUDIO_SERVER_URL", "http://audio-server:8082")


LOCAL_SUBNETS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("fc00::/7"),
)

LOCAL_HOSTNAMES = frozenset(
    {
        "localhost",
        "testclient",
        "host.docker.internal",
        "gateway.docker.internal",
        "docker",
        "unix",
        "0.0.0.0",
    }
)


def get_flask_client_ip(req) -> str:
    """Extract client IP from Flask request considering forward headers."""
    xff = req.headers.get("x-forwarded-for", "").strip()
    if xff:
        parts = [ip.strip() for ip in xff.split(",")]
        if parts and parts[0]:
            return parts[0]
    x_real = req.headers.get("x-real-ip", "").strip()
    if x_real:
        return x_real
    if req.remote_addr:
        return req.remote_addr.strip()
    return ""


def is_flask_client_local(req) -> bool:
    """Check if the requesting client is from a local LAN / Docker subnet."""
    raw_ip = get_flask_client_ip(req)
    if not raw_ip or raw_ip == "unknown-ip":
        return False
    raw_clean = raw_ip.strip().lower()
    if raw_clean in LOCAL_HOSTNAMES:
        return True
    if raw_clean.startswith("::ffff:"):
        raw_clean = raw_clean[7:]
    if "%" in raw_clean:
        raw_clean = raw_clean.split("%")[0]
    try:
        ip_obj = ipaddress.ip_address(raw_clean)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
            return True
        return any(ip_obj in subnet for subnet in LOCAL_SUBNETS)
    except ValueError:
        return False


def is_flask_request_authenticated(req) -> bool:
    """Check if request is authorized via local bypass, session, or API key."""
    # Local LAN clients (192.168.0.0, 10.0.0.0, 172.16-31, loopback, docker) bypass auth
    if is_flask_client_local(req):
        return True

    expected = os.environ.get("ALPACA_API_KEY", "").strip() or os.environ.get("ADMIN_PASSWORD", "").strip()
    public_mode = os.environ.get("ALPACA_PUBLIC_MODE", "").strip().lower() in ("1", "true", "yes")

    # If no key/password configured, allow external access ONLY if explicitly opted-in via ALPACA_PUBLIC_MODE=1
    if not expected:
        return public_mode

    if session.get("authenticated") is True:
        return True
    auth = req.headers.get("authorization", "").strip()
    if auth:
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else auth
        if token == expected:
            return True
    x_key = (req.headers.get("x-api-key") or req.headers.get("x-api-token") or "").strip()
    return x_key == expected


@app.before_request
def enforce_auth_middleware():
    """Redirect unauthenticated public browser requests to /login and return 401 for API calls."""
    if request.path.startswith("/static/") or request.path in (
        "/login",
        "/logout",
        "/api/auth/status",
        "/health",
        "/favicon.ico",
    ):
        return None

    if is_flask_request_authenticated(request):
        return None

    if request.path.startswith("/api/"):
        return jsonify({"error": "Unauthorized. Please authenticate."}), 401

    return redirect(url_for("login_view"))


def get_proxy_headers(extra_headers: dict[str, str] | None = None) -> dict[str, str]:
    headers = dict(extra_headers or {})
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    if key:
        headers.setdefault("Authorization", f"Bearer {key}")
        headers.setdefault("X-API-Key", key)
    return headers


def _find_proxy_url() -> str | None:
    """Return the first reachable proxy URL from PROXY_SERVER_URLS, or None."""
    import httpx

    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=1.0) as client:
                resp = client.get(f"{url}/api/version")
                if resp.status_code == 200:
                    return url
        except Exception:
            continue
    return None


active_pulls: dict[str, dict[str, Any]] = {}
active_pulls_lock = threading.Lock()


def _terminate_process(process, model_name):
    """Terminate a subprocess with SIGTERM, retry with SIGKILL on timeout."""
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"Warning: could not kill process for {model_name}")


def run_puller_thread(model_name, source, local_name, no_resume=False, companion=False):
    global active_pulls

    cmd = [sys.executable, puller_path, "pull", model_name]
    if source and source != "auto":
        cmd += ["--source", source]
    if local_name:
        cmd += ["--name", local_name]
    if no_resume:
        cmd += ["--no-resume"]
    if companion:
        cmd += ["--companion"]

    process = None
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Explicitly pass ROUTER_MODELS_DIR so puller checks the same stop directory
        router_dir = os.getenv("ROUTER_MODELS_DIR")
        if router_dir:
            env["ROUTER_MODELS_DIR"] = router_dir

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None, "subprocess stdout should not be None with stdout=PIPE"

        print(f"Pull started: {model_name} (PID: {process.pid})")
        socketio.emit("pull_log", {"model": model_name, "line": f"[alpaca] Pull started (PID: {process.pid})"})

        while True:
            # Use select() with 1s timeout instead of blocking readline() forever
            ready, _, _ = select.select([process.stdout], [], [], 1.0)
            if ready:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    line_str = line.rstrip()
                    with active_pulls_lock:
                        if model_name in active_pulls:
                            active_pulls[model_name]["logs"].append(line_str)
                            if len(active_pulls[model_name]["logs"]) > 1000:
                                active_pulls[model_name]["logs"].pop(0)
                    socketio.emit("pull_log", {"model": model_name, "line": line_str})
            else:
                # Timeout from select() - check for stop/cancel
                pass

            # Check for stop/cancel events on every loop iteration
            with active_pulls_lock:
                if model_name not in active_pulls:
                    break
                status = active_pulls[model_name].get("status", "running")
                if status == "cancelled":
                    print(f"Pull cancelled: {model_name}")
                    _terminate_process(process, model_name)
                    socketio.emit("pull_status", {"model": model_name, "status": "cancelled"})
                    return
                elif status == "stopping":
                    active_pulls[model_name]["status"] = "stopping"
                    print(f"Pull stopping: {model_name}")
                    _terminate_process(process, model_name)
                    active_pulls[model_name]["status"] = "stopped"
                    socketio.emit("pull_status", {"model": model_name, "status": "stopped"})
                    stop_dir = Path(router_dir or os.getenv("ROUTER_MODELS_DIR", ".alpaca-router")) / ".alpaca-stop"
                    (stop_dir / re.sub(r"[/:.]", "_", model_name)).unlink(missing_ok=True)
                    (stop_dir / model_name.replace("/", "_").replace(":", "_")).unlink(missing_ok=True)
                    return

        rc = process.poll()
        with active_pulls_lock:
            if model_name in active_pulls:
                status = active_pulls[model_name].get("status", "running")
                if status == "stopping":
                    active_pulls[model_name]["status"] = "stopped"
                    socketio.emit("pull_status", {"model": model_name, "status": "stopped"})
                    stop_dir = Path(router_dir or os.getenv("ROUTER_MODELS_DIR", ".alpaca-router")) / ".alpaca-stop"
                    stop_file = stop_dir / f"{model_name.replace('/', '_').replace(':', '_')}"
                    stop_file.unlink(missing_ok=True)
                    return

        if rc == 0:
            model_tracker.record_model_seen(model_name, source="local")
            socketio.emit("pull_status", {"model": model_name, "status": "success"})
        else:
            # Clean up stop marker on failure so it doesn't block future pulls
            stop_dir = Path(router_dir or os.getenv("ROUTER_MODELS_DIR", ".alpaca-router")) / ".alpaca-stop"
            stop_file = stop_dir / f"{model_name.replace('/', '_').replace(':', '_')}"
            stop_file.unlink(missing_ok=True)
            socketio.emit(
                "pull_status",
                {"model": model_name, "status": "failed", "error": f"Exit code {rc}"},
            )

    except Exception as e:
        if process is not None and process.poll() is None:
            _terminate_process(process, model_name)
        socketio.emit("pull_status", {"model": model_name, "status": "failed", "error": str(e)})
    finally:
        with active_pulls_lock:
            active_pulls.pop(model_name, None)


# Active run status tracking
active_run: dict[str, Any] = {
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
    # Multi-step runs report one event per conversation turn; track cumulative
    # turn counts so the progress bar can move within a long workflow.
    ms_state = {"wf": None, "done": 0, "cur_total": 0}

    def callback(event, data):
        global active_run
        with active_run_lock:
            if event == "benchmark_start":
                ms_state.update({"wf": None, "done": 0, "cur_total": 0})
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
                        "total_models": len(data.get("models", [])),
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

            elif event == "test_step":
                # Multi-step workflows emit one event per agentic turn so the
                # dashboard shows movement during long conversations.
                step_no = data.get("step")
                total = data.get("total")
                label = f"{data.get('workflow_label') or data.get('workflow')} — turn {step_no}/{total}: {data.get('label')}"
                active_run["current_test"] = label
                socketio.emit(
                    "test_start",
                    {
                        "model": data["model"],
                        "category": data.get("category"),
                        "test_id": data.get("workflow"),
                        "test_label": label,
                    },
                )
                budget = data.get("num_predict")
                msg = f"↻ turn {step_no}/{total}: {data.get('label')}"
                if budget:
                    msg += f" (budget {budget} tokens)"
                socketio.emit("benchmark_step", {"model": data["model"], "message": msg})

                # Turn-level progress: each agentic turn counts as one unit so
                # the dashboard bar advances during long multi-turn workflows.
                try:
                    step_i, total_i = int(step_no), int(total)
                except (TypeError, ValueError):
                    step_i = total_i = 0
                wf_key = f"{data.get('model')}::{data.get('workflow')}"
                if ms_state["wf"] != wf_key:
                    ms_state["done"] += ms_state["cur_total"]
                    ms_state["cur_total"] = total_i
                    ms_state["wf"] = wf_key
                active_run["tests_completed"] = ms_state["done"] + max(0, step_i - 1)
                if ms_state["done"] + ms_state["cur_total"]:
                    active_run["total_tests"] = ms_state["done"] + ms_state["cur_total"]
                socketio.emit("sync_status", dict(active_run))

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
                            "percentage": round((active_run["tests_completed"] / active_run["total_tests"]) * 100),
                        },
                    },
                )

            elif event == "model_complete":
                active_run["results"].append(data["results"])
                try:
                    m_name = data.get("model")
                    res_obj = data.get("results", {})
                    tasks = (
                        res_obj.get("tasks", [])
                        if isinstance(res_obj, dict)
                        else (res_obj.get("tests", []) if isinstance(res_obj, dict) else [])
                    )
                    tot = len(tasks)
                    succ = sum(1 for t in tasks if isinstance(t, dict) and t.get("success"))
                    pct = (succ / tot * 100.0) if tot > 0 else 0.0
                    model_tracker.record_benchmark_result(
                        model_id=m_name,
                        score_pct=pct,
                        run_type=run_type,
                        result_file=active_run.get("saved_as") or "",
                    )
                except Exception:
                    pass
                socketio.emit("model_complete", {"model": data["model"], "results": data["results"]})

            elif event == "benchmark_complete":
                active_run["status"] = data.get("status", "completed")
                active_run["current_model"] = None
                active_run["current_test"] = None
                active_run["current_category"] = None
                saved_path = data.get("saved_as") or ""
                active_run["saved_as"] = saved_path
                try:
                    for res_item in data.get("results", []):
                        if isinstance(res_item, dict) and res_item.get("model"):
                            m_name = res_item.get("model")
                            tasks = res_item.get("tasks") or res_item.get("tests") or []
                            tot = len(tasks)
                            succ = sum(1 for t in tasks if isinstance(t, dict) and t.get("success"))
                            pct = (succ / tot * 100.0) if tot > 0 else 0.0
                            model_tracker.record_benchmark_result(
                                model_id=m_name,
                                score_pct=pct,
                                run_type=run_type,
                                result_file=saved_path,
                            )
                except Exception:
                    pass
                socketio.emit("benchmark_complete", data)

            elif event == "benchmark_cancelled":
                active_run["status"] = "cancelled"
                active_run["current_model"] = None
                active_run["current_test"] = None
                active_run["current_category"] = None
                socketio.emit("benchmark_cancelled", {"message": "Benchmark cancelled by user"})

    return callback


def run_general_in_thread(
    models, use_proxy, run_cancel_event, callback, test_ids=None, resume=False, groups=None, tiers=None
):
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
                test_ids=test_ids,
                resume=resume,
                groups=groups,
                tiers=tiers,
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
                    print("[benchmark] Thread exiting with status still 'running' - forcing to 'completed'")
                    active_run["status"] = "completed"
                    active_run["current_model"] = None
                    active_run["current_test"] = None
                    active_run["current_category"] = None
                    socketio.emit(
                        "benchmark_complete",
                        {"status": "completed", "saved_as": active_run.get("saved_as")},
                    )

    loop.run_until_complete(run())
    loop.close()


def run_shared_llm_in_thread(models, use_proxy, run_cancel_event, callback, task_ids=None, custom_keys=None):
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
                task_ids=task_ids,
                custom_keys=custom_keys,
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
                    print("[benchmark] SharedLLM thread exiting with status still 'running' - forcing to 'completed'")
                    active_run["status"] = "completed"
                    active_run["current_model"] = None
                    active_run["current_test"] = None
                    active_run["current_category"] = None
                    socketio.emit(
                        "benchmark_complete",
                        {"status": "completed", "saved_as": active_run.get("saved_as")},
                    )

    loop.run_until_complete(run())
    loop.close()


def run_multistep_in_thread(models, use_proxy, run_cancel_event, callback, workflow_ids=None):
    global active_run
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run():
        try:
            await multistep_benchmark.run_multistep_benchmarks(
                models=models,
                use_proxy=use_proxy,
                progress_callback=callback,
                cancel_event=run_cancel_event,
                workflow_ids=workflow_ids,
            )
        except Exception as e:
            print(f"Error in MultiStep execution: {e}")
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
                    print("[benchmark] MultiStep thread exiting with status still 'running' - forcing to 'completed'")
                    active_run["status"] = "completed"
                    active_run["current_model"] = None
                    active_run["current_test"] = None
                    active_run["current_category"] = None
                    socketio.emit(
                        "benchmark_complete",
                        {"status": "completed", "saved_as": active_run.get("saved_as")},
                    )

    loop.run_until_complete(run())
    loop.close()


# Auth Routes
@app.route("/login", methods=["GET", "POST"])
def login_view():
    """Render login page or authenticate submitted credentials."""
    if request.method == "GET":
        if is_flask_request_authenticated(request):
            return redirect(url_for("index"))
        return render_template("login.html")

    data = request.get_json(silent=True) or request.form
    submitted_key = (data.get("api_key") or data.get("password") or "").strip()
    expected = os.environ.get("ALPACA_API_KEY", "").strip() or os.environ.get("ADMIN_PASSWORD", "").strip()

    if not expected or submitted_key == expected:
        session["authenticated"] = True
        if request.is_json:
            return jsonify({"success": True, "redirect": "/"})
        return redirect(url_for("index"))

    if request.is_json:
        return jsonify({"success": False, "error": "Invalid API key or password."}), 401
    return render_template("login.html", error="Invalid API key or password."), 401


@app.route("/logout")
def logout_view():
    """Clear session and redirect to login page."""
    session.pop("authenticated", None)
    return redirect(url_for("login_view"))


@app.route("/api/auth/status")
def auth_status_api():
    """Return origin type and authentication state."""
    expected = os.environ.get("ALPACA_API_KEY", "").strip() or os.environ.get("ADMIN_PASSWORD", "").strip()
    is_local = is_flask_client_local(request)
    authenticated = is_flask_request_authenticated(request)
    return jsonify(
        {
            "authenticated": authenticated,
            "is_local": is_local,
            "auth_required": bool(expected),
            "client_ip": get_flask_client_ip(request),
        }
    )


# Routes
@app.route("/")
def index():
    """Serve the dashboard HTML"""
    return render_template("index.html")


@app.route("/ui/launcher/<container_id>")
def ui_launcher_page(container_id: str):
    """Full-page launcher for a running UI sandbox container (opened by ⛶ Expand)."""
    st = ui_status(container_id)
    if st.get("error"):
        return f"UI container not found: {container_id}", 404
    host_port = st.get("host_port") or ""
    return render_template("ui_launcher.html", container_id=container_id, host_port=host_port)


@app.route("/api/status")
def get_status():
    """Return the current active benchmark status"""
    with active_run_lock:
        return jsonify(dict(active_run))


def _canonical_model_name(name: str) -> str:
    """Drop the ':latest' tag so the same model isn't listed twice under both
    its bare id (e.g. 'kwaipilot-...-iq4-nl') and its ':latest' form. The proxy's
    router resolution treats both forms identically, so the bare id is canonical.
    """
    return name[:-7] if name.endswith(":latest") else name


@app.route("/api/models")
def get_models():
    """Return available models from proxies and direct ollama instances.
    Includes router GGUF models (standalone files without Ollama manifests)
    by merging with _get_router_text_models(), matching the approach used
    by /api/models/text and /api/models/vision.
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        direct_models = loop.run_until_complete(benchmark.discover_all_models())
        proxy_models = loop.run_until_complete(benchmark.discover_all_proxy_models())
        loop.close()

        # Filter out hardcoded fallback placeholders if we successfully retrieved real proxy models
        fallback = benchmark._get_fallback_models()
        if proxy_models and proxy_models != fallback and direct_models == fallback:
            direct_models = []

        router_models = _get_router_text_models()

        # Normalize + dedupe: router/GGUF and proxy-discovered sources can expose
        # the same model under two id forms (bare vs ':latest'); keep the canonical
        # (bare) form so the dashboard Target Models list shows each model once.
        seen: set[str] = set()
        combined: list[str] = []
        for name in direct_models + proxy_models + router_models:
            canonical = _canonical_model_name(name)
            if canonical not in seen:
                seen.add(canonical)
                combined.append(canonical)
        return jsonify({"models": combined, "direct_models": direct_models, "proxy_models": proxy_models})
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


def _compute_test_hash(test_dict):
    """Compute deterministic SHA-256 hash for a test definition."""
    if not isinstance(test_dict, dict):
        return ""
    atts = [a.get("name", "") for a in test_dict.get("attachments", []) if isinstance(a, dict)]
    # Code-category tests are graded with an appended grader directive; bump the
    # recorded directive version when its text changes so outdated_only re-runs.
    directive_version = ""
    if test_dict.get("type") in ("code", "ui") or test_dict.get("category") in (LLMModelBenchmark.CODE_CATEGORIES):
        directive_version = LLMModelBenchmark.GRADER_DIRECTIVE_VERSION
    canonical = {
        "id": test_dict.get("id", ""),
        "prompt": (test_dict.get("prompt") or "").strip(),
        "expected": str(test_dict.get("expected") or "").strip(),
        "expected_output": str(test_dict.get("expected_output") or "").strip(),
        "type": test_dict.get("type", "functional"),
        "kind": test_dict.get("kind", "text"),
        "attachments": sorted(atts),
        "grader_directive": directive_version,
    }
    dumped = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:12]


def _get_test_benchmark_stats():
    """Scan all model benchmark records and aggregate test run history and currency."""
    stats: dict[str, dict] = {}

    current_tests: dict[str, dict] = {}
    for _cat, t in _iter_all_tests():
        tid = t.get("id")
        if tid:
            current_tests[tid] = t

    def _record_run(data, model_name, run):
        """Aggregate a single run's category blocks into the stats table."""
        if not isinstance(run, dict):
            return
        for cat_key, cat_val in run.items():
            if not isinstance(cat_val, dict) or not cat_key.startswith("category_"):
                continue
            for t in cat_val.get("tests") or []:
                if not isinstance(t, dict):
                    continue
                tid = t.get("test_id")
                if not tid:
                    continue

                if tid not in stats:
                    stats[tid] = {
                        "models_tested": [],
                        "models_passed": [],
                        "models_failed": [],
                        "models_tested_count": 0,
                        "models_passed_count": 0,
                        "models_failed_count": 0,
                        "is_out_of_date": False,
                        "out_of_date_count": 0,
                        "out_of_date_models": [],
                        "last_run": None,
                        "models_scores": {},
                        "models_lint": {},
                        "models_breakdown": {},
                        "models_last_run": {},
                        "models_run_count": {},
                        "models_fail_count": {},
                        "models_latency": {},
                        "models_tokens": {},
                        "models_speed": {},
                    }

                st = stats[tid]
                if model_name not in st["models_tested"]:
                    st["models_tested"].append(model_name)
                    st["models_tested_count"] += 1

                    # Track each model's most recent score for the Test Browser card
                    # (used to show the highest human-rated model alongside its score).
                    score = t.get("score", 0)
                    if isinstance(score, str):
                        try:
                            score = float(score)
                        except (TypeError, ValueError):
                            score = 0
                    st["models_scores"][model_name] = score

                    st["models_lint"][model_name] = bool(t.get("lint_passed", t.get("code_ran") is not False))

                    # Per-model score breakdown so the dashboard can explain WHY a
                    # model scored what it did (execution, functional check, code
                    # quality notes, watermark, and any error) instead of showing a
                    # bare number.
                    st["models_breakdown"][model_name] = {
                        "score": score,
                        "functional_pass": t.get("functional_pass"),
                        "code_ran": t.get("code_ran"),
                        "code_score": t.get("code_score"),
                        "lint_passed": t.get("lint_passed"),
                        "code_quality": (t.get("code_quality") or {}).get("score"),
                        "code_quality_notes": (t.get("code_quality") or {}).get("notes") or [],
                        "code_quality_lang": (t.get("code_quality") or {}).get("language"),
                        "watermark": (t.get("watermark") or {}).get("score"),
                        "watermark_flags": (t.get("watermark") or {}).get("flags") or [],
                        "rubric": t.get("rubric"),
                        "error": t.get("error") or t.get("code_error") or "",
                        "last_run": t.get("last_run"),
                    }

                    passed = bool(t.get("success", False) or (t.get("score", 0) >= 50))
                    if passed:
                        st["models_passed"].append(model_name)
                        st["models_passed_count"] += 1
                    else:
                        st["models_failed"].append(model_name)
                        st["models_failed_count"] += 1

                    run_time = t.get("last_run") or data.get("last_updated") or data.get("generated_at")
                    if run_time and (not st["last_run"] or run_time > st["last_run"]):
                        st["last_run"] = run_time
                    if run_time:
                        st["models_last_run"][model_name] = run_time

                    st["models_run_count"][model_name] = int(t.get("run_count") or 1)
                    st["models_fail_count"][model_name] = int(t.get("fail_count") or (0 if passed else 1))

                    latency = t.get("latency")
                    if isinstance(latency, str):
                        try:
                            latency = float(latency)
                        except (TypeError, ValueError):
                            latency = None
                    if latency is None:
                        eval_ns = t.get("eval_duration") or 0
                        prompt_ns = t.get("prompt_eval_duration") or 0
                        try:
                            latency = (float(eval_ns) + float(prompt_ns)) / 1e9
                        except (TypeError, ValueError):
                            latency = 0.0
                    st["models_latency"][model_name] = float(latency or 0.0)

                    tokens = t.get("tokens_generated")
                    try:
                        tokens = int(tokens or 0)
                    except (TypeError, ValueError):
                        tokens = 0
                    st["models_tokens"][model_name] = tokens
                    speed = (tokens / float(latency)) if latency else 0.0
                    st["models_speed"][model_name] = round(speed, 2)

                    cur_test = current_tests.get(tid)
                    if cur_test:
                        cur_hash = _compute_test_hash(cur_test)
                        rec_hash = t.get("test_hash")
                        rec_prompt = (t.get("prompt") or "").strip()
                        cur_prompt = (cur_test.get("prompt") or "").strip()

                        is_outdated = False
                        if rec_hash:
                            is_outdated = rec_hash != cur_hash
                        elif cur_prompt:
                            is_outdated = not (rec_prompt.startswith(cur_prompt) or rec_prompt == cur_prompt)

                        if is_outdated:
                            st["is_out_of_date"] = True
                            if model_name not in st["out_of_date_models"]:
                                st["out_of_date_models"].append(model_name)
                                st["out_of_date_count"] += 1

    # Source of truth: per-model records (most recent, incremental saves).
    # Merged *_benchmarks_latest.json files are deliberately NOT scanned here —
    # they can contain stale/restored data that predates the current test schema
    # and inflates run counts on cards that were never actually benchmarked.
    models_dir = benchmark.MODELS_DIR
    if models_dir.exists():
        for fp in sorted(models_dir.glob("general_*.json")):
            try:
                with open(fp, encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue

            model_name = data.get("model") or fp.stem.replace("general_", "")
            for run in data.get("results") or []:
                _record_run(data, model_name, run)

    return stats


def _outdated_test_ids(models: list[str]) -> list[str]:
    """Return the test IDs whose definitions have changed for any of the given
    models. Used by ``outdated_only`` runs so a model can redo just the
    benchmarks whose prompts/config have been updated since its last run."""
    stats = _get_test_benchmark_stats()
    ids = []
    for tid, st in stats.items():
        if not st.get("is_out_of_date"):
            continue
        outdated_models = st.get("out_of_date_models", [])
        if any(m in outdated_models for m in models):
            ids.append(tid)
        else:
            outdated_sanitized = {re.sub(r"[/:.]", "_", x) for x in outdated_models}
            if any(re.sub(r"[/:.]", "_", m) in outdated_sanitized for m in models):
                ids.append(tid)
    return ids


@app.route("/api/tests")
def get_tests():
    """Return list of all available tests dynamically loaded from configs,
    annotated with run statistics and currency (out-of-date status)."""
    try:
        all_tests = []
        run_stats = _get_test_benchmark_stats()

        for cat, t in _iter_all_tests():
            tid = t["id"]
            st = run_stats.get(tid, {})
            all_tests.append(
                {
                    "id": tid,
                    "category": cat,
                    "label": t.get("label", tid),
                    "type": t.get("type", "functional"),
                    "kind": t.get("kind", "text"),
                    "prompt": t.get("prompt", ""),
                    "expected": t.get("expected", ""),
                    "attachments": _test_attachment_meta(t),
                    "test_hash": _compute_test_hash(t),
                    "models_tested_count": st.get("models_tested_count", 0),
                    "models_passed_count": st.get("models_passed_count", 0),
                    "models_failed_count": st.get("models_failed_count", 0),
                    "models_tested": st.get("models_tested", []),
                    "models_passed": st.get("models_passed", []),
                    "models_scores": st.get("models_scores", {}),
                    "models_lint": st.get("models_lint", {}),
                    "models_breakdown": st.get("models_breakdown", {}),
                    "models_last_run": st.get("models_last_run", {}),
                    "models_run_count": st.get("models_run_count", {}),
                    "models_fail_count": st.get("models_fail_count", {}),
                    "models_latency": st.get("models_latency", {}),
                    "models_tokens": st.get("models_tokens", {}),
                    "models_speed": st.get("models_speed", {}),
                    "is_out_of_date": st.get("is_out_of_date", False),
                    "out_of_date_count": st.get("out_of_date_count", 0),
                    "out_of_date_models": st.get("out_of_date_models", []),
                    "last_run": st.get("last_run"),
                }
            )
        for perf_id, perf_label in [
            ("perf_medium", "Performance: Medium Load (800 tokens)"),
            ("perf_long", "Performance: Long Load (1000 tokens)"),
        ]:
            st = run_stats.get(perf_id, {})
            all_tests.append(
                {
                    "id": perf_id,
                    "category": "performance",
                    "label": perf_label,
                    "type": "performance",
                    "kind": "text",
                    "prompt": "",
                    "expected": "",
                    "attachments": [],
                    "test_hash": perf_id,
                    "models_tested_count": st.get("models_tested_count", 0),
                    "models_passed_count": st.get("models_passed_count", 0),
                    "models_failed_count": st.get("models_failed_count", 0),
                    "models_tested": st.get("models_tested", []),
                    "models_passed": st.get("models_passed", []),
                    "models_scores": st.get("models_scores", {}),
                    "models_lint": st.get("models_lint", {}),
                    "models_last_run": st.get("models_last_run", {}),
                    "models_run_count": st.get("models_run_count", {}),
                    "models_fail_count": st.get("models_fail_count", {}),
                    "models_latency": st.get("models_latency", {}),
                    "models_tokens": st.get("models_tokens", {}),
                    "models_speed": st.get("models_speed", {}),
                    "is_out_of_date": st.get("is_out_of_date", False),
                    "out_of_date_count": st.get("out_of_date_count", 0),
                    "out_of_date_models": st.get("out_of_date_models", []),
                    "last_run": st.get("last_run"),
                }
            )
        return jsonify({"tests": all_tests})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _iter_all_tests():
    """Yield (category, test) pairs from the dynamic tests config.

    Iterates the loaded config directly so any category present in
    benchmark_tests.json (including future ones) is browsable without a
    hardcoded method lookup.
    """
    for cat, tests in benchmark.tests_config.items():
        for t in tests:
            entry = dict(t)
            entry["category"] = cat
            yield cat, entry


def _find_test(test_id):
    """Return (category, test) for the given test id or None."""
    for cat, t in _iter_all_tests():
        if t.get("id") == test_id:
            return cat, t
    return None, None


def _test_attachment_meta(test):
    """Return lightweight attachment metadata for a test (no file contents)."""
    meta = []
    for att in test.get("attachments", []):
        meta.append(
            {
                "name": att.get("name", "attachment"),
                "mime": att.get("mime", "application/octet-stream"),
                "kind": att.get("kind", "file"),
            }
        )
    return meta


def _resolve_attachment(test, att_name):
    """Resolve an attachment to bytes, or None if it cannot be found.

    Supports three storage modes (fully dynamic, nothing hardcoded):
      - inline base64: {"name": ..., "data_base64": "..."}
      - file path relative to repo root: {"name": ..., "path": "data/..."}
      - URL: {"name": ..., "url": "https://..."}
    """
    for att in test.get("attachments", []):
        if att.get("name") != att_name:
            continue
        if att.get("data_base64"):
            try:
                import base64

                return base64.b64decode(att["data_base64"]), att.get("mime", "application/octet-stream")
            except Exception:
                return None
        if att.get("text") is not None:
            return att["text"].encode("utf-8"), att.get("mime", "application/octet-stream")
        if att.get("data"):
            data = att["data"]
            if isinstance(data, str):
                try:
                    import base64

                    data = base64.b64decode(data)
                except Exception:
                    data = data.encode("utf-8")
            return data, att.get("mime", "application/octet-stream")
        if att.get("path"):
            base_dir = Path(__file__).resolve().parent.parent
            full = (base_dir / att["path"]).resolve()
            if not full.is_relative_to(base_dir):
                return None
            try:
                return full.read_bytes(), att.get("mime", "application/octet-stream")
            except Exception:
                return None
        if att.get("url"):
            try:
                import httpx

                resp = httpx.get(att["url"], timeout=15)
                resp.raise_for_status()
                return resp.content, att.get("mime", resp.headers.get("content-type", "application/octet-stream"))
            except Exception:
                return None
    return None


@app.route("/api/tests/<test_id>/attachment/<att_name>")
def get_test_attachment(test_id, att_name):
    """Serve a test attachment for inline preview or download (?download=1)."""
    try:
        _cat, test = _find_test(test_id)
        if not test:
            return jsonify({"error": "test not found"}), 404
        result = _resolve_attachment(test, att_name)
        if not result:
            return jsonify({"error": "attachment not found"}), 404
        data, mime = result
        download = request.args.get("download") == "1"
        from io import BytesIO

        return send_file(
            BytesIO(data),
            mimetype=mime,
            as_attachment=download,
            download_name=att_name,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tests/<test_id>/download")
def download_test(test_id):
    """Download a single test definition (id, kind, prompt, attachments metadata)."""
    try:
        _cat, test = _find_test(test_id)
        if not test:
            return jsonify({"error": "test not found"}), 404
        payload = {
            "id": test.get("id"),
            "kind": test.get("kind", "text"),
            "label": test.get("label", test.get("id")),
            "category": _cat,
            "prompt": test.get("prompt", ""),
            "expected": test.get("expected"),
            "num_predict": test.get("num_predict"),
            "attachments": _test_attachment_meta(test),
        }
        from io import BytesIO

        return send_file(
            BytesIO(json.dumps(payload, indent=2).encode("utf-8")),
            mimetype="application/json",
            as_attachment=True,
            download_name=f"{test.get('id', 'test')}.json",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tests/<test_id>/responses")
def get_test_responses(test_id):
    """Return model-produced responses for a test (e.g. games) so they can be
    downloaded or played directly from the Test Browser. Scans per-model result
    files and returns every non-empty response keyed by model, keeping the
    longest response per model.
    """
    try:
        _cat, test = _find_test(test_id)
        if not test:
            return jsonify({"error": "test not found"}), 404
        out = []
        models_dir = benchmark.MODELS_DIR
        if models_dir.exists():
            for fp in sorted(models_dir.glob("general_*.json")):
                try:
                    with open(fp) as fh:
                        data = json.load(fh)
                except Exception:
                    continue
                model = data.get("model") or fp.stem
                for run in data.get("results") or []:
                    if not isinstance(run, dict):
                        continue
                    for cat in run.values():
                        if not isinstance(cat, dict):
                            continue
                        for t in cat.get("tests") or []:
                            if not isinstance(t, dict):
                                continue
                            if t.get("test_id") != test_id:
                                continue
                            resp = t.get("response") or ""
                            if not resp.strip():
                                continue
                            is_html = bool(re.search(r"<!doctype|<html|<script|<canvas", resp, re.I))
                            passed = t.get("success") if "success" in t else None
                            score = t.get("score")
                            try:
                                score = float(score) if score is not None else None
                            except (TypeError, ValueError):
                                score = None
                            _tg = t.get("tokens_generated") or 0
                            try:
                                _tg = float(_tg)
                            except (TypeError, ValueError):
                                _tg = 0.0
                            _la = t.get("latency") or 0
                            try:
                                _la = float(_la)
                            except (TypeError, ValueError):
                                _la = 0.0
                            out.append(
                                {
                                    "model": model,
                                    "run_file": fp.name,
                                    "response": resp,
                                    "thinking": t.get("thinking") or None,
                                    "response_len": len(resp),
                                    "is_html": is_html,
                                    "passed": passed,
                                    "score": score,
                                    "lint_passed": t.get("lint_passed"),
                                    "code_error": t.get("code_error") or None,
                                    "code_output": t.get("code_output") or None,
                                    "screenshot": t.get("screenshot"),
                                    "latency": t.get("latency"),
                                    "tokens_generated": t.get("tokens_generated"),
                                    "speed": round(_tg / _la, 2) if _la else 0.0,
                                    "run_count": t.get("run_count") or 1,
                                    "fail_count": t.get("fail_count") or 0,
                                    "last_run": t.get("last_run") or data.get("last_updated"),
                                }
                            )
        best = {}
        for o in out:
            k = o["model"]
            if k not in best or o["response_len"] > best[k]["response_len"]:
                best[k] = o
        return jsonify({"test_id": test_id, "responses": list(best.values())})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Human ratings persistence — stored server-side so ratings survive across
# browsers / machines / sessions. File lives under DATA_DIR (default ./data)
# alongside the telemetry/benchmark JSON dumps. Frontend keeps a localStorage
# mirror for offline fallback but treats this file as the canonical store.
# ---------------------------------------------------------------------------
_RATINGS_FILE = Path(os.getenv("DATA_DIR", "data")) / "human_ratings.json"
_RATINGS_LOCK = threading.Lock()


def _load_ratings_file() -> dict:
    try:
        if _RATINGS_FILE.exists():
            with open(_RATINGS_FILE, encoding="utf-8") as fh:
                d = json.load(fh)
                return d if isinstance(d, dict) else {}
    except Exception:
        pass
    return {}


def _save_ratings_file(data: dict) -> None:
    try:
        _RATINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_RATINGS_FILE, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
    except Exception as e:
        print(f"[ratings] failed to save {_RATINGS_FILE}: {e}")


@app.route("/api/ratings", methods=["GET"])
def get_ratings():
    """Return the canonical human ratings store: { testId: { model: 0.5..5 } }."""
    try:
        with _RATINGS_LOCK:
            data = _load_ratings_file()
        return jsonify({"ratings": data})
    except Exception as e:
        return jsonify({"ratings": {}, "error": str(e)}), 500


@app.route("/api/ratings", methods=["POST"])
def post_rating():
    """Create/update/delete a single human rating. Body: {testId, model, rating}."""
    try:
        payload = request.get_json(silent=True) or {}
        test_id = ""
        for k in ("testId", "test_id", "testid", "id"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                test_id = v.strip()
                break
        model = (payload.get("model") or "").strip() if isinstance(payload.get("model"), str) else ""
        raw_rating = payload.get("rating")
        if not test_id or not model:
            return jsonify({"error": "testId and model are required"}), 400
        try:
            rating = float(raw_rating)
        except (TypeError, ValueError):
            return jsonify({"error": "invalid rating"}), 400
        # Clamp to 0..5 and snap to 0.5 increments; 0 means delete
        rating = max(0.0, min(5.0, round(rating * 2) / 2))
        with _RATINGS_LOCK:
            data = _load_ratings_file()
            if rating == 0:
                if test_id in data and model in data[test_id]:
                    del data[test_id][model]
                    if not data[test_id]:
                        del data[test_id]
            else:
                if test_id not in data:
                    data[test_id] = {}
                data[test_id][model] = rating
            _save_ratings_file(data)
        return jsonify({"ok": True, "ratings": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tests/shared_llm")
def get_shared_llm_tests():
    """Return list of all SharedLLM task definitions"""
    try:
        tasks = shared_llm_benchmark.get_all_tasks()
        all_tests = [
            {
                "id": t["id"],
                "category": t["category"],
                "label": t["label"],
                "type": "shared_llm",
            }
            for t in tasks
        ]
        return jsonify({"tests": all_tests})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_host_system_metrics():
    """Fetch CPU and RAM metrics from the host via psutil.
    GPU/VRAM is intentionally omitted here - the web container has no GPU device
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
            errors.append(f"{url}: {e!s}")
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
            # GPU is NOT fetched here - the proxy already has it via docker exec.
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
                # gpu_info is an error dict - leave gpus as empty list
                system_data["gpus"] = []

            return jsonify(
                {
                    "online": True,
                    "system": system_data,
                    "metrics": metrics_resp.json() if metrics_resp.status_code == 200 else {},
                    "runtime": runtime_resp.json() if runtime_resp.status_code == 200 else {},
                    "slots": slots_resp.json() if slots_resp.status_code == 200 else {},
                    "logs": logs_resp.json().get("logs", []) if logs_resp.status_code == 200 else [],
                }
            )
    except Exception as e:
        return jsonify(
            {
                "online": False,
                "error": f"Failed to fetch metrics from active proxy {proxy_url}: {e!s}",
            }
        )


@app.route("/api/sd/status")
def get_sd_status():
    """Fetch health, active model, and queue depth from Stable Diffusion proxy."""
    import httpx

    try:
        with httpx.Client(timeout=1.5) as client:
            resp = client.get(f"{PROXY_URL}/admin/sd/health", headers=get_proxy_headers())
            if resp.status_code == 200:
                data = resp.json()
                return jsonify(
                    {
                        "online": True,
                        "active_model": data.get("active_model"),
                        "sd_server_healthy": data.get("sd_server_healthy"),
                        "queue_depth": data.get("queue_depth"),
                        "vram_total_mb": data.get("vram_total_mb"),
                        "vram_used_mb": data.get("vram_used_mb"),
                        "vram_free_mb": data.get("vram_free_mb"),
                    }
                )
    except Exception as e:
        return jsonify({"online": False, "error": str(e)})
    return jsonify({"online": False, "error": "Proxy unresponsive"})


@app.route("/api/audio/status")
def get_audio_status():
    """Fetch audio-server health (loaded models, VRAM, voice list)."""
    import httpx

    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{AUDIO_SERVER_URL}/health")
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"status": "offline", "error": str(e)})


@app.route("/api/audio/tts", methods=["POST"])
def audio_tts_api():
    """Forward a TTS request to the audio-server and return wav b64 + metadata."""
    import httpx

    data = request.get_json() or {}
    try:
        # Kokoro synthesis + first model load can take a while on cold start.
        with httpx.Client(timeout=600.0) as client:
            resp = client.post(f"{AUDIO_SERVER_URL}/api/tts", json=data)
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/audio/music", methods=["POST"])
def audio_music_api():
    """Forward a music generation request to the audio-server."""
    import httpx

    data = request.get_json() or {}
    try:
        with httpx.Client(timeout=900.0) as client:
            resp = client.post(f"{AUDIO_SERVER_URL}/api/music", json=data)
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/audio/unload", methods=["POST"])
def audio_unload_api():
    """Ask the audio-server to drop loaded models and free VRAM."""
    import httpx

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{AUDIO_SERVER_URL}/api/unload")
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/sd/unload", methods=["POST"])
def unload_sd_model_api():
    """Request sd-proxy to unload its active model."""
    import httpx

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(f"{PROXY_URL}/admin/sd/unload", headers=get_proxy_headers())
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sd/health", methods=["GET"])
def sd_health_api():
    """Proxy sd-server health/active-model info from the proxy."""
    import httpx

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{PROXY_URL}/admin/sd/health", headers=get_proxy_headers())
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e), "online": False}), 500


@app.route("/api/sd/models", methods=["GET"])
def sd_models_api():
    """List locally available Stable Diffusion / image models."""
    import httpx

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{PROXY_URL}/v1/images/models", headers=get_proxy_headers())
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e), "data": []}), 500


@app.route("/api/sd/presets", methods=["GET"])
def sd_presets_api():
    """Fetch presets for realistic photo editing and flyer text generation."""
    import httpx

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{PROXY_URL}/v1/images/presets", headers=get_proxy_headers())
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sd/load", methods=["POST"])
def sd_load_api():
    """Load a Stable Diffusion model into the sd-server backend (no generation)."""
    import httpx

    data = request.get_json() or {}
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{PROXY_URL}/v1/images/models/load", json=data, headers=get_proxy_headers())
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def embed_qr_code_onto_image(
    b64_image_str: str, qr_text: str, position: str = "bottom_right", label: str = "SCAN ME"
) -> str:
    """Generates a scannable QR Code and merges it onto the base64 flyer image."""
    import base64
    from io import BytesIO

    from PIL import Image, ImageDraw, ImageFont

    try:
        import qrcode
    except ImportError:
        return b64_image_str

    try:
        img_bytes = base64.b64decode(b64_image_str)
        base_img = Image.open(BytesIO(img_bytes)).convert("RGBA")
        bw, bh = base_img.size

        # Generate QR Code image
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=6, border=2)
        qr.add_data(qr_text)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGBA")

        # Size QR badge proportionally (approx 18% of flyer width)
        qr_target_w = int(bw * 0.18)
        qr_target_w = max(120, min(240, qr_target_w))
        qr_img = qr_img.resize((qr_target_w, qr_target_w), Image.Resampling.LANCZOS)

        # Create padded white card container with border and label
        padding = 12
        card_w = qr_target_w + (padding * 2)
        card_h = qr_target_w + (padding * 2) + 20

        card = Image.new("RGBA", (card_w, card_h), (255, 255, 255, 245))
        draw = ImageDraw.Draw(card)

        # Draw dark border around card
        draw.rectangle([(0, 0), (card_w - 1, card_h - 1)], outline=(30, 41, 59, 255), width=2)

        # Paste QR code onto card
        card.paste(qr_img, (padding, padding), qr_img)

        # Draw label text below QR code
        if label:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            draw.text((card_w // 2, card_h - 12), label, fill=(15, 23, 42, 255), anchor="mm", font=font)

        # Determine placement coordinates on base image
        margin = 35
        if position == "bottom_left":
            pos_x = margin
            pos_y = bh - card_h - margin
        elif position == "bottom_center":
            pos_x = (bw - card_w) // 2
            pos_y = bh - card_h - margin
        elif position == "top_right":
            pos_x = bw - card_w - margin
            pos_y = margin
        else:  # default bottom_right
            pos_x = bw - card_w - margin
            pos_y = bh - card_h - margin

        # Paste card onto base image
        base_img.paste(card, (pos_x, pos_y), card)

        # Convert back to RGB JPEG base64
        buf = BytesIO()
        base_img.convert("RGB").save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Warning: QR embedding failed: {e}")
        return b64_image_str


@app.route("/api/sd/generate", methods=["POST"])
def sd_generate_api():
    """Forward an image generation request to the proxy (auto-loads SD model and embeds QR code if requested)."""
    import httpx

    data = request.get_json() or {}
    qr_text = data.pop("qr_text", None) or data.pop("qr_url", None)
    qr_position = data.pop("qr_position", "bottom_right")
    qr_label = data.pop("qr_label", "SCAN ME")

    try:
        with httpx.Client(timeout=600.0) as client:
            resp = client.post(f"{PROXY_URL}/v1/images/generations", json=data, headers=get_proxy_headers())
            if resp.status_code == 200 and qr_text:
                resp_json = resp.json()
                if "data" in resp_json and len(resp_json["data"]) > 0:
                    for item in resp_json["data"]:
                        if "b64_json" in item:
                            item["b64_json"] = embed_qr_code_onto_image(
                                item["b64_json"], qr_text, position=qr_position, label=qr_label
                            )
                return jsonify(resp_json), 200
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/companions", methods=["GET"])
def list_companions():
    """List companion model files (VAE / LLM / CLIP / T5XXL) available for SD models."""
    dirs = [
        os.path.join(os.environ.get("ROUTER_MODELS_DIR", "/router-models"), "companions"),
        os.path.join(os.environ.get("MODELS_DIR", "/models"), "companions"),
        "/models/companions",
        "/router-models/companions",
    ]
    found = []
    for d in dirs:
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith((".gguf", ".safetensors")) and fn not in found:
                    found.append(fn)
    return jsonify({"companions": found})


@app.route("/api/sd/edit", methods=["POST"])
def sd_edit_api():
    """Forward a multipart image-edit request to the proxy (auto-loads SD model and embeds QR code if requested)."""
    import httpx

    try:
        data = {}
        files = {}
        for key in request.form:
            vals = request.form.getlist(key)
            data[key] = vals[0] if len(vals) == 1 else vals
        for key in request.files:
            f = request.files[key]
            files[key] = (f.filename, f.read(), f.mimetype)

        qr_text = data.pop("qr_text", None) or data.pop("qr_url", None)
        qr_position = data.pop("qr_position", "bottom_right")
        qr_label = data.pop("qr_label", "SCAN ME")

        with httpx.Client(timeout=600.0) as client:
            resp = client.post(f"{PROXY_URL}/v1/images/edits", data=data, files=files, headers=get_proxy_headers())
            if resp.status_code == 200 and qr_text:
                resp_json = resp.json()
                if "data" in resp_json and len(resp_json["data"]) > 0:
                    for item in resp_json["data"]:
                        if "b64_json" in item:
                            item["b64_json"] = embed_qr_code_onto_image(
                                item["b64_json"], qr_text, position=qr_position, label=qr_label
                            )
                return jsonify(resp_json), 200
            return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/vision/ocr", methods=["POST"])
def vision_ocr_api():
    """Extract text and document structure (headlines, subtext, badges) using Qwen2.5-VL vision model."""
    import base64
    import json
    from io import BytesIO

    import httpx
    from PIL import Image

    try:
        if "file" not in request.files and "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file_obj = request.files.get("file") or request.files.get("image")
        filename = file_obj.filename.lower()
        file_bytes = file_obj.read()

        b64_image = None

        if filename.endswith(".pdf"):
            try:
                import fitz  # PyMuPDF

                doc = fitz.open(stream=file_bytes, filetype="pdf")
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img.thumbnail((1024, 1024))
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=85)
                b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as pdf_err:
                return jsonify({"error": f"PDF processing error: {pdf_err}"}), 400
        else:
            img = Image.open(BytesIO(file_bytes)).convert("RGB")
            img.thumbnail((1024, 1024))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = (
            "You are an expert Document AI and OCR vision assistant.\n"
            "Analyze the uploaded image or document and perform text extraction and layout parsing.\n\n"
            "Respond ONLY with a valid JSON object with the following structure:\n"
            "{\n"
            '  "full_text": "Complete extracted text from top to bottom...",\n'
            '  "headline": "Main title or headline text found in the image",\n'
            '  "subtext": "Subtitle, body text, or event details",\n'
            '  "badge": "Badge, price tag, or call-to-action text (e.g. 50% OFF, GET TICKETS)"\n'
            "}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                ],
            }
        ]

        model = (request.form.get("model") or request.args.get("model", "")).strip()
        if not model:
            return jsonify({"error": "'model' parameter is required"}), 400

        proxy_model = model.replace("--", ":") if ("--" in model and ":" not in model) else model
        with httpx.Client(timeout=300.0) as client:
            error_details = ""
            raw_text = ""
            try:
                resp = client.post(
                    f"{PROXY_URL}/v1/chat/completions",
                    json={"model": proxy_model, "messages": messages, "max_tokens": 1000, "temperature": 0.1},
                    headers=get_proxy_headers(),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data["choices"][0]["message"]["content"]
                else:
                    error_details = f"Vision proxy returned HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as exc:
                error_details = str(exc)

            if not raw_text or "not supported" in raw_text.lower():
                return (
                    jsonify(
                        {
                            "error": "Vision OCR extraction failed",
                            "details": error_details or "Model produced empty or unsupported response.",
                        }
                    ),
                    502,
                )

            try:
                clean_json = raw_text.strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_json:
                    clean_json = clean_json.split("```")[1].split("```")[0].strip()
                parsed = json.loads(clean_json)
            except Exception:
                parsed = {"full_text": raw_text, "headline": "", "subtext": "", "badge": ""}

            return jsonify({"status": "success", "ocr_result": parsed, "raw_response": raw_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/images/text_compose", methods=["POST"])
def text_compose_api():
    """Deterministic text-line editing: erase a horizontal text band and/or
    draw replacement text. Pure PIL/numpy - no diffusion model, pixel-exact.

    Form params: image (file), band ('y0:y1' inclusive), gap_above, gap_below,
    text (omit for erase-only), font, font_size, color ('r,g,b'),
    post (none|vintage), output_format (png|jpeg).
    """
    import base64
    from io import BytesIO

    from PIL import Image as PILImage

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file_obj = request.files["image"]
        image = PILImage.open(BytesIO(file_obj.read())).convert("RGB")

        band = (request.form.get("band") or request.args.get("band", "")).strip()
        if not band:
            return jsonify({"error": "'band' parameter is required, format y0:y1 (inclusive)"}), 400
        try:
            y0, y1 = (int(v) for v in band.split(":"))
        except ValueError:
            return jsonify({"error": "'band' must be 'y0:y1' (inclusive)"}), 400

        from imageops import draw_text, fill_band_deterministic

        gap_above = request.form.get("gap_above")
        gap_below = request.form.get("gap_below")
        gap_above = int(gap_above) if gap_above else y0 - 1
        gap_below = int(gap_below) if gap_below else y1 + 1

        result = fill_band_deterministic(image, y0, y1 + 1, gap_above, gap_below)
        text = request.form.get("text", "")
        if text:
            color_str = request.form.get("color", "255,255,255")
            color = tuple(int(v) for v in color_str.split(","))
            result = draw_text(result, text, (y0, y1 + 1), request.form.get("font"), None, color)

        post = (request.form.get("post") or "none").strip()
        if post == "vintage":
            import numpy as np

            a = np.asarray(result.convert("RGB")).astype(float)
            sepia = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
            r = a @ sepia.T
            r[:, :, 0] += 42.0
            r[:, :, 1] = r[:, :, 1] * 0.88 + 18.0
            r[:, :, 2] = r[:, :, 2] * 0.66 - 6.0
            r += np.random.default_rng(7).normal(0.0, 12.0, r.shape)
            r = np.clip(r, 0, 255)
            h, w, _ = r.shape
            yy, xx = np.mgrid[0:h, 0:w]
            d = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
            r = r * (1.0 - 0.30 * np.clip(d - 0.35, 0, 1) ** 2)[:, :, None]
            result = PILImage.fromarray(np.clip(r, 0, 255).astype("uint8"))

        out_format = (request.form.get("output_format") or "png").lower()
        buf = BytesIO()
        result.save(buf, format="JPEG" if out_format == "jpeg" else "PNG")
        return jsonify(
            {
                "status": "success",
                "image_b64": base64.b64encode(buf.getvalue()).decode("utf-8"),
                "band": [y0, y1],
                "text": text,
                "post": post,
                "width": result.width,
                "height": result.height,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _router_id_to_public(model_id: str) -> str:
    """Convert router model ID (with -- separator) to public name (with : separator)."""
    if "--" in model_id:
        family, tag = model_id.rsplit("--", 1)
        return f"{family}:{tag}"
    return model_id


def _get_router_text_models() -> list[str]:
    """Return all text/VL model IDs from llama-server router (includes standalone GGUFs without Ollama manifests)."""
    try:
        import httpx

        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{LLAMA_SERVER_URL}/models")
            if resp.status_code == 200:
                # Filter out image-generation models (SD/flux/image-edit)
                sd_keywords = ["image-edit", "qwen-image", "qwen_image", "sd", "flux", "diffusion"]
                models = resp.json().get("data", [])
                return [
                    _router_id_to_public(m["id"])
                    for m in models
                    if not any(kw in m["id"].lower() for kw in sd_keywords)
                ]
    except Exception:
        pass
    return []


def _get_active_text_model() -> str:
    """Helper to return the currently loaded model on proxy, or first available text/VL model."""
    try:
        import httpx

        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{PROXY_URL}/admin/runtime", headers=get_proxy_headers())
            if resp.status_code == 200:
                active = resp.json().get("active_model")
                if active:
                    return active
            tags_resp = client.get(f"{PROXY_URL}/api/tags", headers=get_proxy_headers())
            if tags_resp.status_code == 200:
                models = tags_resp.json().get("models", [])
                text_models = [m["name"] for m in models if m.get("type") != "image"]
                if text_models:
                    return text_models[0]
    except Exception:
        pass
    # Final fallback: query llama-server router directly for any text/VL model
    router_models = _get_router_text_models()
    if router_models:
        return router_models[0]
    return ""


@app.route("/api/models/text")
def get_text_models():
    """Return all available text and vision-language models suitable for chat/vision tasks.
    Merges Ollama-registered models from proxy /api/tags with standalone router GGUFs
    (e.g. Qwen2.5-VL models that have no Ollama manifest).
    """
    import httpx

    try:
        ollama_models: list[str] = []
        with httpx.Client(timeout=5.0) as client:
            tags_resp = client.get(f"{PROXY_URL}/api/tags", headers=get_proxy_headers())
            if tags_resp.status_code == 200:
                for m in tags_resp.json().get("models", []):
                    if m.get("type") != "image":
                        ollama_models.append(m["name"])
    except Exception:
        ollama_models = []

    router_models = _get_router_text_models()

    # Merge, preserving order and deduplicating (':latest' form is canonicalized)
    seen: set[str] = set()
    combined: list[str] = []
    for name in ollama_models + router_models:
        canonical = _canonical_model_name(name)
        if canonical not in seen:
            seen.add(canonical)
            combined.append(canonical)

    return jsonify({"models": combined})


@app.route("/api/models/vision")
def get_vision_models():
    """Return only multimodal Vision-Language (VL) models capable of image analysis."""
    vl_keywords = ("vl", "vision", "llava", "mmproj", "moondream", "cogvlm", "minicpm-v")

    # Get all text/VL models
    all_models_resp = get_text_models()
    all_models = all_models_resp.get_json().get("models", [])

    # Filter strictly for Vision/VL models
    vision_models = [m for m in all_models if any(k in m.lower() for k in vl_keywords)]

    # Sort vision models by parameter count descending (7b > 3b)
    def _sort_key(name: str) -> int:
        name_lower = name.lower()
        if "70b" in name_lower:
            return 70
        if "35b" in name_lower or "32b" in name_lower:
            return 35
        if "14b" in name_lower or "13b" in name_lower:
            return 14
        if "7b" in name_lower or "8b" in name_lower:
            return 7
        if "3b" in name_lower:
            return 3
        if "2b" in name_lower or "1.5b" in name_lower:
            return 2
        return 1

    vision_models.sort(key=_sort_key, reverse=True)
    return jsonify({"models": vision_models})


@app.route("/api/models/online")
def get_online_models_api():
    """Return configured online LLM providers and available catalog of online models."""
    try:
        from online_providers import online_model_provider

        return jsonify(
            {
                "providers": online_model_provider.get_configured_providers(),
                "models": online_model_provider.get_available_models(),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/online/providers", methods=["GET"])
def get_online_providers_credentials_api():
    """Return configured status and masked keys for all online providers."""
    try:
        from online_providers import online_model_provider

        return jsonify(
            {
                "providers": online_model_provider.get_masked_credentials(),
                "configured": online_model_provider.get_configured_providers(),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/online/providers/save", methods=["POST"])
def save_online_providers_credentials_api():
    """Save API credentials to .env and runtime."""
    try:
        from online_providers import online_model_provider

        data = request.get_json() or {}
        keys = {}
        if "alpaca_api_key" in data:
            keys["ALPACA_API_KEY"] = data["alpaca_api_key"].strip()
        if "openrouter_api_key" in data:
            keys["OPENROUTER_API_KEY"] = data["openrouter_api_key"].strip()
        if "huggingface_token" in data:
            keys["HUGGING_FACE_TOKEN"] = data["huggingface_token"].strip()
        if "cloudflare_api_token" in data:
            keys["CLOUDFLARE_API_TOKEN"] = data["cloudflare_api_token"].strip()
        if "cloudflare_account_id" in data:
            keys["CLOUDFLARE_ACCOUNT_ID"] = data["cloudflare_account_id"].strip()
        if "opencode_zen_api_key" in data:
            keys["OPENCODE_ZEN_API_KEY"] = data["opencode_zen_api_key"].strip()
        if "opencode_zen_base_url" in data:
            keys["OPENCODE_ZEN_BASE_URL"] = data["opencode_zen_base_url"].strip()
        if "groq_api_key" in data:
            keys["GROQ_API_KEY"] = data["groq_api_key"].strip()
        if "gemini_api_key" in data:
            keys["GEMINI_API_KEY"] = data["gemini_api_key"].strip()

        result = online_model_provider.save_credentials(keys)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/online/providers/alpaca/generate", methods=["POST"])
def generate_alpaca_token_api():
    """Generate a cryptographically secure token for Alpaca proxy protection."""
    try:
        from online_providers import online_model_provider

        token = online_model_provider.generate_alpaca_token()
        return jsonify({"token": token, "success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/online/providers/test", methods=["POST"])
def test_online_provider_connection_api():
    """Test connection to an online provider with provided or saved credentials."""
    try:
        from online_providers import online_model_provider

        data = request.get_json() or {}
        provider = data.get("provider", "")
        custom_keys = data.get("keys", {})

        result = asyncio.run(online_model_provider.test_connection(provider=provider, custom_keys=custom_keys))
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/online/models/search", methods=["GET"])
def search_online_models_api():
    """Search and discover live online models from remote APIs."""
    try:
        from online_providers import online_model_provider

        provider = request.args.get("provider", "all")
        query = request.args.get("query", "")
        free_only = request.args.get("free_only", "false").lower() in ("true", "1", "yes")

        models = asyncio.run(
            online_model_provider.fetch_live_models(provider=provider, query=query, free_only=free_only)
        )
        return jsonify({"success": True, "models": models, "count": len(models)})
    except Exception as e:
        return jsonify({"success": False, "models": [], "count": 0, "error": str(e)}), 500


@app.route("/api/online/models/selected", methods=["GET", "POST"])
def selected_online_models_api():
    """Get or save user's custom selection of online models."""
    try:
        from online_providers import online_model_provider

        if request.method == "POST":
            data = request.get_json() or {}
            models = data.get("models", [])
            for m in models:
                if isinstance(m, dict) and m.get("id"):
                    model_tracker.record_model_seen(m["id"], source=m.get("provider", "online"))
            result = online_model_provider.save_selected_models(models)
            return jsonify(result)

        models = online_model_provider.get_selected_models()
        return jsonify({"success": True, "models": models, "count": len(models)})
    except Exception as e:
        return jsonify({"success": False, "models": [], "error": str(e)}), 500


@app.route("/api/models/tracking")
def get_models_tracking_api():
    """Returns tracking summary for newly added models vs previously benchmarked items."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            direct_models = loop.run_until_complete(benchmark.discover_all_models())
            proxy_models = loop.run_until_complete(benchmark.discover_all_proxy_models())
        finally:
            loop.close()

        router_models = _get_router_text_models()
        combined_local = list(dict.fromkeys(direct_models + proxy_models + router_models))

        from online_providers import online_model_provider

        online_models = online_model_provider.get_selected_models()

        summary = model_tracker.get_tracking_summary(
            current_local_models=combined_local, current_online_models=online_models
        )
        return jsonify(summary)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/vision/describe", methods=["POST"])
def vision_describe_api():
    """Analyze an uploaded image using Vision AI and output a detailed image description."""
    import base64
    from io import BytesIO

    import httpx
    from PIL import Image

    try:
        if "file" not in request.files and "image" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        file_obj = request.files.get("file") or request.files.get("image")
        img_bytes = file_obj.read()

        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = (
            "Analyze this image in detail for an AI image editing assistant. "
            "Provide a concise, vivid description covering: "
            "1. Subject & Pose\n"
            "2. Hair, Makeup, or Key Features\n"
            "3. Outfit & Accessories\n"
            "4. Background Environment & Scene\n"
            "5. Lighting & Color Palette.\n\n"
            "Return a clean 2-3 sentence summary describing the scene accurately."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                ],
            }
        ]

        model = (request.form.get("model") or request.args.get("model", "")).strip()
        if not model:
            v_resp = get_vision_models()
            v_data = v_resp.get_json() if hasattr(v_resp, "get_json") else {}
            v_models = v_data.get("models", []) if isinstance(v_data, dict) else []
            if v_models:
                model = v_models[0]
            else:
                t_resp = get_text_models()
                t_data = t_resp.get_json() if hasattr(t_resp, "get_json") else {}
                t_models = t_data.get("models", []) if isinstance(t_data, dict) else []
                model = t_models[0] if t_models else "qwen2.5-vl:latest"

        proxy_model = model.replace("--", ":") if ("--" in model and ":" not in model) else model
        model_used = model
        description = ""
        error_detail = None

        with httpx.Client(timeout=300.0) as client:
            # Pre-warm / ensure model is loaded in proxy first
            with contextlib.suppress(Exception):
                client.post(
                    f"{PROXY_URL}/admin/models/switch",
                    json={"model": proxy_model},
                    headers=get_proxy_headers(),
                    timeout=300.0,
                )

            try:
                resp = client.post(
                    f"{PROXY_URL}/v1/chat/completions",
                    json={
                        "model": proxy_model,
                        "messages": messages,
                        "max_tokens": 400,
                        "temperature": 0.2,
                    },
                    headers=get_proxy_headers(),
                )
                if resp.status_code == 200:
                    description = resp.json()["choices"][0]["message"]["content"].strip()
                else:
                    error_detail = f"Proxy returned HTTP {resp.status_code}: {resp.text[:200]}"
                    app.logger.warning(
                        "Vision describe: model %s returned %s - %s", model, resp.status_code, resp.text[:200]
                    )
            except Exception as exc:
                error_detail = f"Connection error: {exc}"
                app.logger.warning("Vision describe: request failed - %s", exc)

            if not description or "error" in description.lower():
                err_msg = error_detail or "Vision AI model returned an empty response"
                app.logger.error("Vision describe failed for model %s: %s", model, err_msg)
                return jsonify({"error": f"Vision AI analysis failed ({model}): {err_msg}"}), 500

            return jsonify(
                {
                    "status": "success",
                    "image_description": description,
                    "model_used": model_used,
                }
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/vision/synthesize_edit_prompt", methods=["POST"])
def vision_synthesize_edit_prompt_api():
    """Synthesize a master Stable Diffusion / Qwen-Image edit prompt based on base description + requested changes."""
    import httpx

    data = request.get_json() or {}
    base_desc = data.get("base_description", "").strip()
    desired_changes = data.get("desired_changes", "").strip()
    style_preset = data.get("style_preset", "photorealistic").strip()

    if not base_desc or not desired_changes:
        return jsonify({"error": "base_description and desired_changes are required"}), 400

    model = (data.get("model") or "").strip()
    if not model:
        t_resp = get_text_models()
        t_data = t_resp.get_json() if hasattr(t_resp, "get_json") else {}
        t_models = t_data.get("models", []) if isinstance(t_data, dict) else []
        model = t_models[0] if t_models else "qwen2.5-coder:latest"

    proxy_model = model.replace("--", ":") if ("--" in model and ":" not in model) else model

    target_image_model = (data.get("target_image_model") or "").lower()
    preserve_face = data.get("preserve_face", True)
    if "redesign" in style_preset.lower() or "change face" in style_preset.lower():
        preserve_face = False

    # Style-to-strength mapping: face/identity edits need low strength; outfit changes medium; full transformations higher
    _style_lower = style_preset.lower()
    if any(k in _style_lower for k in ("retouch", "restore", "polish", "tone", "color grade")):
        strength = 0.25
    elif any(k in _style_lower for k in ("outfit", "style transform", "hair", "makeup", "accessory")):
        strength = 0.40
    elif any(k in _style_lower for k in ("background", "scene", "environment", "lighting")):
        strength = 0.50
    else:
        strength = 0.55

    if not preserve_face:
        strength = max(strength, 0.65)

    negative = (
        "painting, illustration, cartoon, digital art, anime, drawing, sketch, watercolor, "
        "oil painting, rendered, CGI, 3D render, plastic skin, airbrushed, doll, "
        "blurry, low resolution, distorted geometry, noise, grain, overexposed"
    )
    if preserve_face:
        negative += ", deformed face, changed identity, different person, altered face"

    face_inst = (
        "Keep the subject's face, facial features, identity, and skin texture exactly the same."
        if preserve_face
        else "Allow changing the subject's face and identity to match the new character style."
    )

    if "qwen" in target_image_model:
        system_msg = (
            "You are an expert at writing image editing instructions for Qwen Image Edit (an instruction-following VLM). "
            "The AI editor understands plain English instructions - it does NOT use Stable Diffusion tag syntax. "
            "Your task is to write a single, clear, natural language editing instruction. "
            "Critical rules: "
            "1. Write in natural language, like instructions to a human photo editor. "
            "2. Start with what to CHANGE, then state what to KEEP the same. "
            f"3. {face_inst} "
            "4. Specify that the output should be a photorealistic photograph, not a painting or digital art. "
            "5. Output ONLY the final instruction - no explanations, no preamble, no quotes."
        )
    elif "flux" in target_image_model:
        system_msg = (
            "You are an expert prompt engineer for Flux image generation models. "
            "Your task is to write a rich, detailed natural-language description of the modified image. "
            "Critical rules: "
            "1. Describe the full scene in vivid visual detail. "
            f"2. {face_inst} "
            "3. Output ONLY the final prompt paragraph - no explanations, no quotes, no preamble."
        )
    else:  # Stable Diffusion / SDXL
        face_tag = "preserve exact face and identity, " if preserve_face else ""
        system_msg = (
            "You are an expert AI image prompt engineer for Stable Diffusion and SDXL in-painting. "
            "Your task is to write a single, cohesive Stable Diffusion img2img prompt. "
            "Critical rules: "
            "1. Combine original scene elements with requested modifications cleanly using descriptive tags and keywords. "
            f"2. {face_tag}Include quality tags: photorealistic photograph, 8k resolution, RAW photo, sharp focus, professional photography. "
            "3. Output ONLY the final synthesized prompt string - no explanations, no quotes, no preamble."
        )

    user_msg = (
        f"Original scene: {base_desc}\n"
        f"Requested changes: {desired_changes}\n"
        f"Style goal: {style_preset}\n"
        f"Target Model: {target_image_model}\n\n"
        f"Write the prompt now:"
    )

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{PROXY_URL}/v1/chat/completions",
                json={
                    "model": proxy_model,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": 600,
                    "temperature": 0.4,
                    "think": False,
                },
                headers=get_proxy_headers(),
            )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                # Strip any thinking blocks thinking models may emit
                import re

                master_prompt = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
            else:
                app.logger.warning(
                    "Synthesis: model %s returned %s - %s", proxy_model, resp.status_code, resp.text[:300]
                )
                master_prompt = ""

            if not master_prompt:
                master_prompt = (
                    f"photorealistic photograph, {base_desc}, {desired_changes}, preserve exact face and identity, "
                    f"{style_preset} style, 8k resolution, RAW photo, DSLR camera, natural lighting, "
                    f"sharp focus, real skin texture, professional photography"
                )

            return jsonify(
                {
                    "status": "success",
                    "master_prompt": master_prompt,
                    "suggested_strength": strength,
                    "suggested_negative": negative,
                }
            )
    except Exception as exc:
        app.logger.warning("Synthesis: request failed - %s", exc)
        master_prompt = (
            f"photorealistic photograph, {base_desc}, {desired_changes}, preserve exact face and identity, "
            f"{style_preset} style, 8k resolution, RAW photo, DSLR camera, natural lighting, sharp focus"
        )
        return jsonify(
            {
                "status": "success",
                "master_prompt": master_prompt,
                "suggested_strength": strength,
                "suggested_negative": negative,
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
    test_ids = data.get("test_ids", None)
    resume = bool(data.get("resume", False))
    groups = data.get("groups", None)
    tiers = data.get("tiers", None)
    outdated_only = bool(data.get("outdated_only", False))

    if not models:
        return jsonify({"error": "No models specified"}), 400

    if outdated_only:
        test_ids = _outdated_test_ids(models)
        if not test_ids:
            return jsonify(
                {
                    "status": "No outdated benchmarks",
                    "message": "All benchmark definitions are up to date for the selected models — nothing to redo.",
                }
            ), 200

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
        active_run["groups"] = groups
        active_run["tiers"] = tiers

        benchmark_thread = threading.Thread(
            target=run_general_in_thread,
            args=(models, use_proxy, cancel_event, callback, test_ids, resume, groups, tiers),
            daemon=True,
        )
        benchmark_thread.start()

    return jsonify({"status": "Benchmark started", "active_run": dict(active_run)})


@app.route("/api/sandbox/serve", methods=["POST"])
def sandbox_serve():
    """Host benchmark-produced web/Node/Python code on a local port for viewing."""
    data = request.get_json() or {}
    code = data.get("code", "")
    lang = data.get("lang", "html")
    if not code:
        return jsonify({"error": "No code provided"}), 400
    res = serve_app(code, lang)
    if res.get("error"):
        return jsonify({"error": res["error"]}), 500
    return jsonify(res)


def _serve_container_host_port(container_id: str) -> str | None:
    """Resolve a serving container's published host port via the docker socket."""
    try:
        client = docker.DockerClient(base_url="unix:///var/run/docker.sock")
        try:
            c = client.containers.get(container_id)
            c.reload()
            for ports in (c.ports or {}).values():
                if ports:
                    hp = ports[0].get("HostPort")
                    if hp:
                        return str(hp)
        finally:
            client.close()
    except Exception:  # pragma: no cover - runtime dependent
        return None
    return None


@app.route("/serve/<container_id>/", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
@app.route("/serve/<container_id>/<path:subpath>", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
def sandbox_serve_proxy(container_id: str, subpath: str = ""):
    """Reverse-proxy a sandboxed app through the dashboard origin.

    The dashboard is reached over port 5000 (LAN, VPN, or a forwarded/external
    host), while sandbox apps bind random ephemeral ports that external networks
    cannot reach. This route tunnels the app through the dashboard's own origin so
    it works from anywhere. The web container reaches the sandbox's published port
    via ``host.docker.internal`` (compose ``extra_hosts`` host-gateway).
    """
    host_port = _serve_container_host_port(container_id)
    if not host_port:
        return jsonify({"error": "Serving container not found or no published port"}), 404

    upstream = f"http://host.docker.internal:{host_port}"
    url = f"{upstream}/{subpath}"
    query = request.query_string.decode("utf-8") if request.query_string else ""
    if query:
        url = f"{url}?{query}"

    fwd_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "connection", "transfer-encoding", "accept-encoding")
    }
    fwd_headers["Host"] = f"host.docker.internal:{host_port}"

    body = request.get_data() if request.method in ("POST", "PUT", "PATCH") else None
    try:
        with httpx.Client(timeout=60.0, follow_redirects=False) as client:
            resp = client.request(
                request.method,
                url,
                headers=fwd_headers,
                content=body,
            )
    except httpx.HTTPError as e:
        return jsonify({"error": f"Upstream unreachable: {e}"}), 502

    resp_headers = {}
    for k, v in resp.headers.items():
        if k.lower() in ("content-length", "connection", "transfer-encoding", "content-encoding"):
            continue
        resp_headers[k] = v
    return Response(
        resp.content,
        status=resp.status_code,
        headers=resp_headers,
        content_type=resp.headers.get("content-type", "text/html"),
    )


def _ws_proxy_pump(src, dst):
    """Relay WebSocket frames from ``src`` to ``dst`` until either side closes."""
    try:
        while True:
            data = src.receive()
            if data is None:
                break
            dst.send(data)
    except Exception:  # pragma: no cover - transport dependent
        pass
    finally:
        with contextlib.suppress(Exception):
            dst.close()


@app.route("/serve/ws/<container_id>/", methods=["GET"], websocket=True)
@app.route("/serve/ws/<container_id>/<path:ws_path>", methods=["GET"], websocket=True)
def sandbox_serve_ws_proxy(container_id: str, ws_path: str = ""):
    """Tunnel a sandbox app's WebSocket endpoint through the dashboard origin.

    noVNC connects to ``ws://<host>:<host_port>/websockify`` which is not
    reachable from a remote machine — only port 5000 is forwarded. This route
    terminates the browser's WebSocket on the dashboard and relays frames to the
    sandbox container's websockify via ``host.docker.internal``.
    """
    host_port = _serve_container_host_port(container_id)
    if not host_port:
        return jsonify({"error": "Serving container not found or no published port"}), 404

    from simple_websocket import Client, Server

    try:
        ws = Server(request.environ)
    except Exception as e:  # pragma: no cover - transport dependent
        return jsonify({"error": f"WebSocket handshake failed: {e}"}), 400

    upstream = None
    try:
        upstream = Client.connect(f"ws://host.docker.internal:{host_port}/websockify")
    except Exception as e:
        with contextlib.suppress(Exception):
            ws.close()
        return jsonify({"error": f"Upstream websocket unreachable: {e}"}), 502

    t1 = threading.Thread(target=_ws_proxy_pump, args=(ws, upstream), daemon=True)
    t2 = threading.Thread(target=_ws_proxy_pump, args=(upstream, ws), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    return ""


@app.route("/api/sandbox/serve_ui", methods=["POST"])
def sandbox_serve_ui():
    """Run benchmark-produced X11/UI code (e.g. pygame) and stream it to a browser iframe."""
    data = request.get_json() or {}
    code = data.get("code", "")
    lang = data.get("lang", "python")
    if not code:
        return jsonify({"error": "No code provided"}), 400
    res = serve_ui(code, lang)
    if res.get("error"):
        return jsonify({"error": res["error"]}), 500
    return jsonify(res)


@app.route("/api/sandbox/stop_serve", methods=["POST"])
def sandbox_stop_serve():
    """Stop a serving container started by /api/sandbox/serve."""
    data = request.get_json() or {}
    cid = data.get("container_id")
    if not cid:
        return jsonify({"error": "No container_id provided"}), 400
    return jsonify(stop_serve(cid))


@app.route("/api/sandbox/ui/exec", methods=["POST"])
def sandbox_ui_exec():
    """Run an arbitrary shell command inside a UI container (launcher terminal)."""
    data = request.get_json() or {}
    cid = data.get("container_id")
    command = (data.get("command") or "").strip()
    if not cid:
        return jsonify({"error": "No container_id provided"}), 400
    if not command:
        return jsonify({"error": "No command provided"}), 400
    return jsonify(ui_exec(cid, command))


@app.route("/api/sandbox/ui/status", methods=["POST"])
def sandbox_ui_status():
    """Report a UI container's runtime state (pid, exit code, stdout tail)."""
    data = request.get_json() or {}
    cid = data.get("container_id")
    if not cid:
        return jsonify({"error": "No container_id provided"}), 400
    return jsonify(ui_status(cid))


@app.route("/api/sandbox/ui/screenshot", methods=["POST"])
def sandbox_ui_screenshot():
    """Capture the current Xvfb framebuffer of a UI container as a PNG (base64)."""
    data = request.get_json() or {}
    cid = data.get("container_id")
    if not cid:
        return jsonify({"error": "No container_id provided"}), 400
    return jsonify(ui_screenshot(cid))


@app.route("/api/sandbox/ui/restart", methods=["POST"])
def sandbox_ui_restart():
    """Relaunch the app inside a UI container on the same X display."""
    data = request.get_json() or {}
    cid = data.get("container_id")
    if not cid:
        return jsonify({"error": "No container_id provided"}), 400
    return jsonify(ui_restart(cid))


@app.route("/api/benchmark/groups", methods=["GET"])
def benchmark_groups():
    """List the available benchmark groups (top-level categories)."""
    try:
        groups = sorted(benchmark.tests_config.keys())
        return jsonify({"groups": groups})
    except Exception as e:
        return jsonify({"groups": [], "error": str(e)})


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
    test_ids = data.get("test_ids") or None  # None means run all
    custom_keys = data.get("custom_keys") or None

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
            args=(models, use_proxy, cancel_event, callback, test_ids, custom_keys),
            daemon=True,
        )
        benchmark_thread.start()

    return jsonify({"status": "SharedLLM Benchmark started", "active_run": dict(active_run)})


@app.route("/api/tests/multistep", methods=["GET"])
def list_multistep_workflows():
    """List registered multi-step agentic workflows."""
    try:
        workflows = [
            {
                "id": w["id"],
                "category": w["category"],
                "label": w["label"],
                "description": w.get("description", ""),
                "steps": len(w["steps"]),
                "type": "multistep",
            }
            for w in multistep_benchmark.get_all_workflows()
        ]
        return jsonify({"tests": workflows})
    except Exception as e:
        return jsonify({"tests": [], "error": str(e)})


@app.route("/api/multistep/artifact/<path:filename>", methods=["GET"])
def serve_multistep_artifact(filename):
    """Serve a multi-step workflow artifact (final game HTML or per-turn raw text).

    Only files that live inside ARTIFACTS_DIR are served; path traversal and
    non-artifact extensions are rejected so this cannot read arbitrary files.
    """
    try:
        base = Path(multistep_benchmark.ARTIFACTS_DIR).resolve()
        target = (base / filename).resolve()
        if base != target and base not in target.parents:
            return jsonify({"error": "Invalid artifact path"}), 400
        if not target.is_file():
            return jsonify({"error": "Artifact not found"}), 404
        if target.suffix.lower() not in (".html", ".txt"):
            return jsonify({"error": "Unsupported artifact type"}), 400
        mimetype = "text/html" if target.suffix.lower() == ".html" else "text/plain"
        return send_file(str(target), mimetype=mimetype)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _annotate_multistep_artifact_urls(data: dict) -> dict:
    """Attach web-viewable URLs to multistep result payloads.

    Adds task.artifact_url (final assembled document) and
    step.response_url (per-turn raw response) when those files exist,
    so the dashboard can link to the FULL generated code instead of the
    truncated response excerpt stored in the snapshot.
    """
    try:
        base = Path(multistep_benchmark.ARTIFACTS_DIR).resolve()
        for rec in data.get("results", []):
            for task in rec.get("tasks", []):
                art = task.get("artifact")
                if art:
                    p = Path(art)
                    if (base / p.name).is_file():
                        task["artifact_url"] = f"/api/multistep/artifact/{p.name}"
                for step in task.get("steps") or []:
                    rp = step.get("response_path")
                    if rp and (base / Path(rp).name).is_file():
                        step["response_url"] = f"/api/multistep/artifact/{Path(rp).name}"
    except Exception:
        pass
    return data


@app.route("/api/run/multistep", methods=["POST"])
def start_multistep_benchmark():
    """Start a multi-step agentic benchmark run (long multi-turn workflows)"""
    global cancel_event, benchmark_thread, active_run
    with active_run_lock:
        if active_run["status"] == "running":
            return jsonify({"error": "Benchmark is already running"}), 409

    data = request.get_json() or {}
    models = data.get("models", [])
    use_proxy = data.get("use_proxy", True)
    workflow_ids = data.get("workflow_ids") or None  # None means run all

    if not models:
        return jsonify({"error": "No models specified"}), 400

    with active_run_lock:
        cancel_event = threading.Event()
        callback = get_progress_callback("multistep")

        active_run["status"] = "running"
        active_run["type"] = "multistep"
        active_run["models"] = models
        active_run["use_proxy"] = use_proxy
        active_run["tests_completed"] = 0
        active_run["total_tests"] = 0
        active_run["results"] = []
        active_run["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        active_run["saved_as"] = None

        benchmark_thread = threading.Thread(
            target=run_multistep_in_thread,
            args=(models, use_proxy, cancel_event, callback, workflow_ids),
            daemon=True,
        )
        benchmark_thread.start()

    return jsonify({"status": "MultiStep Benchmark started", "active_run": dict(active_run)})


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
        shared_files = list(shared_dir.glob("shared_llm_benchmarks_*.json")) if shared_dir.exists() else []

        # Load MultiStep results
        multistep_dir = multistep_benchmark.RESULTS_DIR
        multistep_files = list(multistep_dir.glob("multistep_benchmarks_*.json")) if multistep_dir.exists() else []

        # Load per-model result files (results follow the model)
        per_model_general = list(benchmark.MODELS_DIR.glob("general_*.json")) if benchmark.MODELS_DIR.exists() else []
        per_model_shared = (
            list(shared_llm_benchmark.MODELS_DIR.glob("shared_*.json"))
            if shared_llm_benchmark.MODELS_DIR.exists()
            else []
        )
        per_model_multistep = (
            list(multistep_benchmark.MODELS_DIR.glob("multistep_*.json"))
            if multistep_benchmark.MODELS_DIR.exists()
            else []
        )

        results_list = []

        def _append_result(file_path, type_name):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                results_list.append(
                    {
                        "filename": file_path.name,
                        "type": type_name,
                        "generated_at": data.get("generated_at"),
                        "benchmark_type": data.get("benchmark_type"),
                        "models_tested": data.get("models_tested"),
                        "status": data.get("status", "completed"),
                        "models": [r.get("model") for r in data.get("results", []) if r.get("model")],
                        "per_model": bool(data.get("per_model")),
                        "saved_as": str(file_path),
                    }
                )
            except Exception as fe:
                print(f"Error reading file {file_path.name}: {fe}")

        # Process general files
        for file_path in general_files:
            _append_result(file_path, "general")

        # Process SharedLLM files
        for file_path in shared_files:
            _append_result(file_path, "shared_llm")

        # Process MultiStep files
        for file_path in multistep_files:
            _append_result(file_path, "multistep")

        # Process per-model files last so per-model (latest per model) is authoritative on dedupe
        for file_path in per_model_general:
            _append_result(file_path, "general")
        for file_path in per_model_shared:
            _append_result(file_path, "shared_llm")
        for file_path in per_model_multistep:
            _append_result(file_path, "multistep")

        # Sort files by timestamp (newest first)
        results_list.sort(key=lambda x: x.get("generated_at") or "", reverse=True)
        return jsonify({"results": results_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _rebuild_per_model_files():
    """Prune per-model aggregate files that no longer have any backing run snapshot.

    Run snapshots are the source of truth for benchmark history; per-model files
    are derived aggregates. After a run snapshot is deleted, drop per-model files
    for models that no longer appear in ANY remaining run snapshot so a deleted
    run no longer lingers in the UI. Existing per-model files for models still
    covered by a snapshot are left untouched (they are authoritative and may hold
    newer data than any single snapshot).
    """
    try:
        general_snapshots = (
            sorted(benchmark.RESULTS_DIR.glob("benchmarks_*.json")) if benchmark.RESULTS_DIR.exists() else []
        )
        shared_snapshots = (
            sorted(shared_llm_benchmark.RESULTS_DIR.glob("shared_llm_benchmarks_*.json"))
            if shared_llm_benchmark.RESULTS_DIR.exists()
            else []
        )
        multistep_snapshots = (
            sorted(multistep_benchmark.RESULTS_DIR.glob("multistep_benchmarks_*.json"))
            if multistep_benchmark.RESULTS_DIR.exists()
            else []
        )

        # Models still covered by at least one remaining run snapshot.
        general_models: set[str] = set()
        shared_models: set[str] = set()
        multistep_models: set[str] = set()
        for snapshot_path in general_snapshots:
            try:
                with open(snapshot_path) as f:
                    snapshot = json.load(f)
            except Exception:
                continue
            general_models.update(r.get("model") for r in snapshot.get("results", []) if r.get("model"))
        for snapshot_path in shared_snapshots:
            try:
                with open(snapshot_path) as f:
                    snapshot = json.load(f)
            except Exception:
                continue
            shared_models.update(r.get("model") for r in snapshot.get("results", []) if r.get("model"))
        for snapshot_path in multistep_snapshots:
            try:
                with open(snapshot_path) as f:
                    snapshot = json.load(f)
            except Exception:
                continue
            multistep_models.update(r.get("model") for r in snapshot.get("results", []) if r.get("model"))

        # Remove per-model files for models no longer covered by any run snapshot.
        if benchmark.MODELS_DIR.exists():
            for pm_path in benchmark.MODELS_DIR.glob("general_*.json"):
                try:
                    with open(pm_path) as f:
                        pm = json.load(f)
                except Exception:
                    continue
                model = pm.get("model")
                if model and model not in general_models:
                    os.remove(pm_path)
        if shared_llm_benchmark.MODELS_DIR.exists():
            for pm_path in shared_llm_benchmark.MODELS_DIR.glob("shared_*.json"):
                try:
                    with open(pm_path) as f:
                        pm = json.load(f)
                except Exception:
                    continue
                model = pm.get("model")
                if model and model not in shared_models:
                    os.remove(pm_path)
        if multistep_benchmark.MODELS_DIR.exists():
            for pm_path in multistep_benchmark.MODELS_DIR.glob("multistep_*.json"):
                try:
                    with open(pm_path) as f:
                        pm = json.load(f)
                except Exception:
                    continue
                model = pm.get("model")
                if model and model not in multistep_models:
                    os.remove(pm_path)
    except Exception as e:
        print(f"[results] _rebuild_per_model_files error: {e}")


@app.route("/api/results/<filename>", methods=["GET", "DELETE"])
def get_result_detail(filename):
    """Get or delete a specific benchmark result file"""
    try:
        filename = os.path.basename(filename)

        # Determine directory based on name prefix.
        # NOTE: "shared_llm_" MUST be checked before "shared_" because run files
        # are named "shared_llm_benchmarks_*.json" and would otherwise be
        # misrouted to the per-model MODELS_DIR (which only holds "shared_<model>.json").
        # Same pattern for MultiStep: run snapshots are "multistep_benchmarks_*",
        # per-model files are "multistep_<model>.json".
        if filename.startswith("multistep_benchmarks_"):
            file_path = multistep_benchmark.RESULTS_DIR / filename
        elif filename.startswith("multistep_"):
            file_path = multistep_benchmark.MODELS_DIR / filename
        elif filename.startswith("shared_llm_"):
            file_path = shared_llm_benchmark.RESULTS_DIR / filename
        elif filename.startswith("shared_"):
            file_path = shared_llm_benchmark.MODELS_DIR / filename
        elif filename.startswith("general_"):
            file_path = benchmark.MODELS_DIR / filename
        else:
            file_path = benchmark.RESULTS_DIR / filename

        if not file_path.exists():
            return jsonify({"error": "Result file not found"}), 404

        if request.method == "DELETE":
            was_run_snapshot = bool(
                filename.startswith("benchmarks_")
                or filename.startswith("shared_llm_benchmarks_")
                or filename.startswith("multistep_benchmarks_")
            )
            was_per_model = bool(
                filename.startswith("general_") or filename.startswith("shared_") or filename.startswith("multistep_")
            )
            affected_models = set()
            try:
                with open(file_path) as f:
                    doc = json.load(f)
                if doc.get("model"):
                    affected_models.add(doc.get("model"))
                for r in doc.get("results", []):
                    if r.get("model"):
                        affected_models.add(r.get("model"))
            except Exception:
                pass

            os.remove(file_path)

            # Run snapshots are the source of truth for per-model aggregates.
            # If we just removed a run snapshot, rebuild the per-model files from the
            # remaining snapshots so deleted runs no longer linger in the views.
            if was_run_snapshot:
                _rebuild_per_model_files()

            # If a per-model file was removed directly, prune from latest snapshots
            if was_per_model:
                for am in affected_models:
                    latest_gen = benchmark.RESULTS_DIR / "all_benchmarks_latest.json"
                    if latest_gen.exists():
                        try:
                            with open(latest_gen) as f:
                                ldoc = json.load(f)
                            ldoc["results"] = [m for m in ldoc.get("results", []) if m.get("model") != am]
                            with open(latest_gen, "w") as f:
                                json.dump(ldoc, f, indent=2, default=str)
                        except Exception:
                            pass
                    latest_sh = shared_llm_benchmark.RESULTS_DIR / "all_shared_benchmarks_latest.json"
                    if latest_sh.exists():
                        try:
                            with open(latest_sh) as f:
                                ldoc = json.load(f)
                            ldoc["results"] = [m for m in ldoc.get("results", []) if m.get("model") != am]
                            with open(latest_sh, "w") as f:
                                json.dump(ldoc, f, indent=2, default=str)
                        except Exception:
                            pass
                    latest_ms = multistep_benchmark.RESULTS_DIR / "all_multistep_benchmarks_latest.json"
                    if latest_ms.exists():
                        try:
                            with open(latest_ms) as f:
                                ldoc = json.load(f)
                            ldoc["results"] = [m for m in ldoc.get("results", []) if m.get("model") != am]
                            with open(latest_ms, "w") as f:
                                json.dump(ldoc, f, indent=2, default=str)
                        except Exception:
                            pass

            with contextlib.suppress(Exception):
                model_tracker.scan_historical_benchmarks()

            return jsonify({"status": "deleted", "filename": filename})

        with open(file_path) as f:
            data = json.load(f)
        if filename.startswith("multistep"):
            data = _annotate_multistep_artifact_urls(data)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/benchmarks/export", methods=["GET"])
def export_benchmarks():
    """Export all general benchmark data for every model.

    Query params:
      format=json (default) -> full JSON document
      format=csv           -> flat per-test CSV
      model=<name>         -> restrict to a single model
    """
    try:
        fmt = (request.args.get("format") or "json").lower()
        model_filter = request.args.get("model")

        # Aggregate every per-model general file + the latest merged snapshot.
        rows = []
        sources = []
        latest = benchmark.RESULTS_DIR / "all_benchmarks_latest.json"
        if latest.exists():
            sources.append(latest)
        sources.extend(sorted(benchmark.MODELS_DIR.glob("general_*.json")))

        for src in sources:
            try:
                with open(src) as f:
                    doc = json.load(f)
            except Exception:
                continue
            models = doc.get("results", [])
            if doc.get("per_model") and doc.get("model"):
                models = doc["results"]
            for m in models:
                mname = m.get("model")
                if model_filter and mname != model_filter:
                    continue
                for key, cat in m.items():
                    if not key.startswith("category_") or not isinstance(cat, dict):
                        continue
                    for t in cat.get("tests", []):
                        rows.append(
                            {
                                "model": mname,
                                "category": key.replace("category_", ""),
                                "test_id": t.get("test_id"),
                                "label": t.get("test_label"),
                                "success": bool(t.get("success")),
                                "code_quality": (t.get("code_quality") or {}).get("score"),
                                "syntax_valid": (t.get("code_quality") or {}).get("syntax_valid"),
                                "watermark": (t.get("watermark") or {}).get("score"),
                                "tokens_per_sec": cat.get("avg_tokens_per_sec"),
                                "ttft_ms": cat.get("avg_ttft_ms"),
                                "tokens_generated": cat.get("avg_tokens_generated"),
                                "last_run": t.get("last_run"),
                            }
                        )

        if fmt == "csv":
            import csv
            import io

            buf = io.StringIO()
            cols = [
                "model",
                "category",
                "test_id",
                "label",
                "success",
                "code_quality",
                "syntax_valid",
                "watermark",
                "tokens_per_sec",
                "ttft_ms",
                "tokens_generated",
                "last_run",
            ]
            w = csv.DictWriter(buf, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
            return Response(
                buf.getvalue(),
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment; filename=benchmarks_export.csv"},
            )

        return jsonify(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "format": "json",
                "test_count": len(rows),
                "rows": rows,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/benchmarks/completed", methods=["GET"])
def benchmarks_completed():
    """Return the test_ids already completed (pass or fail) for a given model.

    Used by the dashboard to pre-deselect finished tests when launching a resume run.
    """
    model = (request.args.get("model") or "").strip()
    if not model:
        return jsonify({"error": "model query param required"}), 400
    pm_file = benchmark.MODELS_DIR / f"general_{benchmark._sanitize_model_filename(model)}.json"
    completed = []
    try:
        if pm_file.exists():
            with open(pm_file) as f:
                doc = json.load(f)
            for mres in doc.get("results", []):
                for cat in mres:
                    if cat.startswith("category_"):
                        for t in mres[cat].get("tests", []):
                            tid = t.get("test_id")
                            if tid and tid not in completed:
                                completed.append(tid)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"model": model, "completed": completed})


def _purge_model_benchmarks(model: str) -> dict[str, Any]:
    """Completely purge all benchmark history, per-model files, artifacts, snapshots, and tracker entries for a model."""
    clean_model = (model or "").strip()
    if not clean_model:
        return {"removed": False, "general": False, "shared": False, "snapshots_pruned": 0}

    variants = {
        clean_model,
        clean_model.replace("--", ":"),
        clean_model.replace(":", "--"),
        clean_model.replace("--", "_"),
        clean_model.replace(":", "_"),
        clean_model.removesuffix(":latest"),
        clean_model + ":latest" if not clean_model.endswith(":latest") else clean_model,
        clean_model.removesuffix(".gguf"),
        clean_model.split("/")[-1],
    }

    # 1. Delete per-model files and artifacts from both suites
    gen_removed = benchmark.delete_model_results(clean_model)
    shared_removed = shared_llm_benchmark.delete_model_results(clean_model)

    # 2. Prune from latest aggregate snapshots
    for latest_path in [
        benchmark.RESULTS_DIR / "all_benchmarks_latest.json",
        shared_llm_benchmark.RESULTS_DIR / "all_shared_benchmarks_latest.json",
    ]:
        if latest_path.exists():
            try:
                with open(latest_path) as f:
                    doc = json.load(f)
                orig_len = len(doc.get("results", []))
                doc["results"] = [m for m in doc.get("results", []) if m.get("model") not in variants]
                if len(doc["results"]) != orig_len:
                    with open(latest_path, "w") as f:
                        json.dump(doc, f, indent=2, default=str)
            except Exception:
                pass

    # 3. Prune from all run snapshot files
    snapshots_pruned = 0
    for res_dir in [benchmark.RESULTS_DIR, shared_llm_benchmark.RESULTS_DIR]:
        if res_dir.exists():
            for snap_path in list(res_dir.glob("*.json")):
                if snap_path.name.startswith("all_"):
                    continue
                try:
                    with open(snap_path) as f:
                        doc = json.load(f)
                    orig_len = len(doc.get("results", []))
                    if orig_len == 0:
                        continue
                    doc["results"] = [r for r in doc.get("results", []) if r.get("model") not in variants]
                    if len(doc["results"]) == 0:
                        snap_path.unlink()
                        snapshots_pruned += 1
                    elif len(doc["results"]) != orig_len:
                        doc["models_tested"] = [m for m in doc.get("models_tested", []) if m not in variants]
                        with open(snap_path, "w") as f:
                            json.dump(doc, f, indent=2, default=str)
                        snapshots_pruned += 1
                except Exception:
                    pass

    # 4. Remove from model tracker
    with contextlib.suppress(Exception):
        model_tracker.delete_model(clean_model)

    # 5. Rebuild any dependent per-model files & resync tracker
    _rebuild_per_model_files()
    with contextlib.suppress(Exception):
        model_tracker.scan_historical_benchmarks()

    return {
        "removed": bool(gen_removed or shared_removed or snapshots_pruned > 0),
        "general": gen_removed,
        "shared": shared_removed,
        "snapshots_pruned": snapshots_pruned,
    }


@app.route("/api/benchmarks/model/<path:model>", methods=["DELETE"])
def delete_model_benchmarks(model):
    """Delete all benchmark data for a single model across general and SharedLLM suites."""
    try:
        purge_res = _purge_model_benchmarks(model)
        return jsonify(
            {
                "status": "deleted" if purge_res["removed"] else "nothing",
                "model": model,
                **purge_res,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/benchmarks/clear", methods=["POST", "DELETE"])
def clear_all_benchmarks():
    """Clear all benchmark data across general and SharedLLM suites, artifacts, and tracker."""
    try:
        deleted_files = 0

        # 1. Clean general benchmark results & models
        if benchmark.RESULTS_DIR.exists():
            for f in benchmark.RESULTS_DIR.glob("*.json"):
                try:
                    f.unlink()
                    deleted_files += 1
                except Exception:
                    pass
        if benchmark.MODELS_DIR.exists():
            for f in benchmark.MODELS_DIR.glob("*.json"):
                try:
                    f.unlink()
                    deleted_files += 1
                except Exception:
                    pass
        if benchmark.ARTIFACTS_DIR.exists():
            for f in benchmark.ARTIFACTS_DIR.glob("*"):
                try:
                    f.unlink()
                    deleted_files += 1
                except Exception:
                    pass

        # 2. Clean SharedLLM benchmark results & models
        if shared_llm_benchmark.RESULTS_DIR.exists():
            for f in shared_llm_benchmark.RESULTS_DIR.glob("*.json"):
                try:
                    f.unlink()
                    deleted_files += 1
                except Exception:
                    pass
        if shared_llm_benchmark.MODELS_DIR.exists():
            for f in shared_llm_benchmark.MODELS_DIR.glob("*.json"):
                try:
                    f.unlink()
                    deleted_files += 1
                except Exception:
                    pass
        if shared_llm_benchmark.ARTIFACTS_DIR.exists():
            for f in shared_llm_benchmark.ARTIFACTS_DIR.glob("*"):
                try:
                    f.unlink()
                    deleted_files += 1
                except Exception:
                    pass

        # 3. Reset Model Tracker
        with contextlib.suppress(Exception):
            model_tracker.clear_all()

        return jsonify(
            {
                "status": "cleared",
                "message": f"Successfully cleared all benchmark data ({deleted_files} files removed).",
                "files_removed": deleted_files,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Failed to clear benchmark data: {e}"}), 500


@app.route("/api/artifacts", methods=["GET", "POST"])
def manage_artifacts():
    """List saved benchmark artifacts, or save a new artifact (overwrites on re-run)."""
    artifacts_dir = benchmark.ARTIFACTS_DIR
    try:
        if request.method == "POST":
            body = request.get_json(silent=True) or {}
            model = (body.get("model") or "").strip()
            test_id = (body.get("test_id") or "").strip()
            filename = (body.get("filename") or "").strip()
            content = body.get("content") or ""
            artifact_type = (body.get("type") or "python").strip().lower()

            if not model or not test_id or not content:
                return jsonify({"error": "model, test_id and content are required"}), 400

            sanitized_model = re.sub(r"[/:.]", "_", model)
            if not filename:
                ext = "html" if artifact_type == "html" else "py"
                filename = f"{sanitized_model}__{test_id}.{ext}"
            else:
                filename = os.path.basename(filename)

            base_name = os.path.splitext(filename)[0]
            file_path = artifacts_dir / filename

            if artifact_type == "html":
                file_path.write_text(content)
                viewer_path = file_path
            else:
                file_path.write_text(content)
                viewer_path = artifacts_dir / f"{base_name}.html"
                escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                viewer_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Benchmark Artifact: {filename}</title>
<style>
  body {{ background:#0f172a; color:#e2e8f0; font-family:'Segoe UI',system-ui,sans-serif; margin:0; padding:24px; }}
  h1 {{ font-size:18px; color:#a5b4fc; margin:0 0 4px; }}
  .meta {{ color:#64748b; font-size:13px; margin-bottom:16px; }}
  .toolbar {{ margin-bottom:16px; }}
  .toolbar a {{ display:inline-block; background:#6366f1; color:#fff; text-decoration:none; padding:8px 16px; border-radius:6px; font-size:13px; margin-right:8px; }}
  pre {{ background:#0b1120; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:16px; overflow:auto; font-family:'JetBrains Mono',Consolas,monospace; font-size:13px; line-height:1.5; color:#cbd5e1; }}
</style>
</head>
<body>
  <h1>Benchmark Artifact</h1>
  <div class="meta">Model: {model} &middot; Test: {test_id}</div>
  <div class="toolbar">
    <a href="/api/artifacts/{base_name}.py?download=1" download>Download .py</a>
    <a href="#" onclick="window.close();return false;">Close</a>
  </div>
  <pre>{escaped}</pre>
</body>
</html>
"""
                viewer_path.write_text(viewer_html)

            return jsonify(
                {
                    "status": "saved",
                    "filename": filename,
                    "model": model,
                    "test_id": test_id,
                    "type": artifact_type,
                    "download_url": f"/api/artifacts/{filename}?download=1",
                    "host_url": f"/api/artifacts/{viewer_path.name}",
                }
            )

        artifacts = []
        for file_path in sorted(artifacts_dir.glob("*.py"), key=lambda p: p.stat().st_mtime, reverse=True):
            artifacts.append(
                {
                    "filename": file_path.name,
                    "type": "python",
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime,
                    "download_url": f"/api/artifacts/{file_path.name}",
                }
            )
        for file_path in sorted(artifacts_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True):
            artifacts.append(
                {
                    "filename": file_path.name,
                    "type": "html",
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime,
                    "download_url": f"/api/artifacts/{file_path.name}",
                    "host_url": f"/api/artifacts/{file_path.name}",
                }
            )
        artifacts.sort(key=lambda a: a.get("modified", 0), reverse=True)
        return jsonify({"artifacts": artifacts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/artifacts/<filename>", methods=["GET", "DELETE"])
def get_artifact(filename):
    """Download or delete a saved benchmark artifact."""
    try:
        filename = os.path.basename(filename)
        file_path = benchmark.ARTIFACTS_DIR / filename
        if not file_path.exists():
            return jsonify({"error": "Artifact not found"}), 404

        if request.method == "DELETE":
            os.remove(file_path)
            return jsonify({"status": "deleted", "filename": filename})

        as_attachment = request.args.get("download", "0") == "1"
        return send_file(str(file_path.resolve()), as_attachment=as_attachment, download_name=filename)
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
    """Retrieve all model profile sections from models.ini, and include standalone .profile.json overlays"""
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
            profiles["*"]["backend"] = "llama.cpp"

        def merge_companion_profiles(section_name, target_dict):
            # Load and merge in order: gguf, safetensors, profile
            for ext in [".gguf.profile.json", ".safetensors.profile.json", ".profile.json"]:
                p_path = ini_path.parent / f"{section_name}{ext}"
                if p_path.exists():
                    try:
                        with open(p_path) as pf:
                            profile_data = json.load(pf)
                            if isinstance(profile_data, dict):
                                target_dict.update(profile_data)
                    except Exception as pe:
                        print(f"Failed to merge profile {p_path}: {pe}")

        for section in config.sections():
            if section != "*":
                profiles[section] = dict(config[section])
                merge_companion_profiles(section, profiles[section])

                # Smart classification: if the section has any SD-specific parameters,
                # mark it as stable-diffusion backend instead of llama.cpp
                sd_keys = {"vae", "clip_l", "t5xxl", "llm", "model_family", "gpu_layers", "threads"}
                if any(k in profiles[section] for k in sd_keys):
                    profiles[section]["backend"] = "stable-diffusion"
                else:
                    profiles[section]["backend"] = "llama.cpp"

        # Discover all *.profile.json files in the router directory
        # to ensure SD / image models (which are excluded from models.ini LLM sections)
        # can also be loaded and edited in the profiles editor UI.
        try:
            router_dir = ini_path.parent
            if router_dir.exists():
                for entry in router_dir.glob("*.profile.json"):
                    base = entry.name[:-13]  # strip ".profile.json"
                    # A standalone profile is only valid if its backing model file
                    # still exists; otherwise it is an orphan left behind by a
                    # deleted model. Skip those so stale profiles don't reappear.
                    backing = base
                    if not (backing.endswith(".gguf") or backing.endswith(".safetensors")):
                        backing = base + ".gguf"
                    if not (
                        (router_dir / backing).exists()
                        or (router_dir / (base + ".safetensors")).exists()
                        or (router_dir / base).exists()
                    ):
                        continue
                    # E.g. "qwen-vl.profile.json" -> section "qwen-vl"
                    section_name = base
                    if section_name.endswith(".gguf") or section_name.endswith(".safetensors"):
                        section_name = section_name.rsplit(".", 1)[0]
                    if section_name not in profiles:
                        profiles[section_name] = {}
                        merge_companion_profiles(section_name, profiles[section_name])

                        # Smart classification for discovered profiles
                        sd_keys = {"vae", "clip_l", "t5xxl", "llm", "model_family", "gpu_layers", "threads"}
                        if any(k in profiles[section_name] for k in sd_keys):
                            profiles[section_name]["backend"] = "stable-diffusion"
                        else:
                            profiles[section_name]["backend"] = "llama.cpp"
        except Exception as pe:
            print(f"Failed to scan standalone profile JSONs: {pe}")

        return jsonify({"profiles": profiles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/profiles/save", methods=["POST"])
def save_profile():
    """Save/update a model profile.

    For llama.cpp models this writes the ``[section]`` to ``models.ini`` and a
    companion ``<section>.profile.json`` overlay. For image/SD models (backend
    ``stable-diffusion``) the profile lives exclusively in a
    ``<section>.<ext>.profile.json`` overlay next to the router symlink, so we
    write there and deliberately do NOT touch ``models.ini`` (SD models must
    never be registered as llama.cpp LLM sections).
    """
    data = request.get_json() or {}
    section = data.get("section")
    settings = data.get("settings")
    backend = data.get("backend", "llama.cpp")
    if not section or not isinstance(settings, dict):
        return jsonify({"error": "Invalid payload: 'section' and 'settings' required"}), 400

    ini_path = get_models_ini_path()
    try:
        if backend == "stable-diffusion":
            # Persist to the canonical SD profile overlay only.
            profile_path = _resolve_sd_profile_path(ini_path.parent, section)
            existing_profile = {}
            if profile_path.exists():
                try:
                    with open(profile_path) as pf:
                        existing_profile = json.load(pf)
                except Exception:
                    pass
            for k, v in settings.items():
                if v is None or v == "":
                    existing_profile.pop(k, None)
                else:
                    existing_profile[k] = v
            if not existing_profile:
                if profile_path.exists():
                    with contextlib.suppress(Exception):
                        os.remove(profile_path)
            else:
                with open(profile_path, "w") as pf:
                    json.dump(existing_profile, pf, indent=4)
                with contextlib.suppress(Exception):
                    os.chmod(profile_path, 0o666)
            return jsonify(
                {
                    "status": "success",
                    "message": f"Successfully updated SD profile {profile_path.name}",
                }
            )

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
        with contextlib.suppress(Exception):
            os.chmod(ini_path, 0o666)

        # Write to profile.json as well so reindexing doesn't discard overrides
        if section != "*":
            try:
                profile_path = ini_path.parent / f"{section}.profile.json"
                existing_profile = {}
                if profile_path.exists():
                    try:
                        with open(profile_path) as pf:
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
                with contextlib.suppress(Exception):
                    os.chmod(profile_path, 0o666)
            except Exception as pe:
                print(f"Failed to save profile json overlay: {pe}")

        return jsonify({"status": "success", "message": f"Successfully updated section [{section}]"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _resolve_sd_profile_path(router_dir, section):
    """Return the canonical ``.profile.json`` path for an image/SD model section.

    SD models are stored in the router directory as ``<section>.<ext>`` symlinks
    (where ``<ext>`` is ``.gguf`` or ``.safetensors``); the proxy reads the
    companion profile as ``<symlink> + '.profile.json'``. Match that location so
    UI edits are picked up by the sd-server.
    """
    for ext in (".gguf", ".safetensors"):
        symlink = router_dir / f"{section}{ext}"
        if symlink.exists() or symlink.is_symlink():
            return router_dir / f"{section}{ext}.profile.json"
    return router_dir / f"{section}.profile.json"


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

        removed = False
        if config.has_section(section):
            config.remove_section(section)
            with open(ini_path, "w") as f:
                config.write(f)
            with contextlib.suppress(Exception):
                os.chmod(ini_path, 0o666)
            removed = True

        # Remove any companion profile overlays (LLM <section>.profile.json and
        # image/SD <section>.<ext>.profile.json). SD models are not in models.ini
        # but still own a profile overlay, so deletion must not 404 on them.
        removed_any_profile = False
        for cand in (
            ini_path.parent / f"{section}.profile.json",
            _resolve_sd_profile_path(ini_path.parent, section),
        ):
            try:
                if cand.exists():
                    os.remove(cand)
                    removed_any_profile = True
            except Exception as pe:
                print(f"Failed to remove profile json file: {pe}")

        if removed or removed_any_profile:
            return jsonify({"status": "success", "message": f"Successfully deleted profile [{section}]"})
        else:
            return jsonify({"error": f"Section [{section}] not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/logs/download")
def download_logs():
    """Download historical proxy logs as a text file"""
    import httpx

    proxy_url = _find_proxy_url()

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
        return f"Failed to retrieve logs: {e!s}", 500


@app.route("/api/proxy/restart", methods=["POST"])
def restart_proxy_services():
    """Trigger background restart of llama-server and alpaca-proxy docker containers"""
    import httpx

    # 1. Try to find the active proxy and call /admin/restart
    proxy_url = _find_proxy_url()

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
    return jsonify({"status": "success", "message": "Backend restart sequence initiated via fallback."})


@app.route("/api/requests")
def get_active_requests():
    """Fetch active and recently completed requests from the proxy"""
    import httpx

    proxy_url = _find_proxy_url()

    if not proxy_url:
        return jsonify({"error": "Could not connect to any proxy endpoints."}), 503

    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{proxy_url}/admin/requests")
            if resp.status_code == 200:
                data = resp.json()
                from online_providers import get_online_requests

                online = get_online_requests()
                data.setdefault("active_requests", []).extend(online.get("active_requests", []))
                data.setdefault("completed_requests", []).extend(online.get("completed_requests", []))
                # Merge proxy + online entries into a single time-ordered list so the
                # dashboard's newest-first render does not put stale entries on top
                # (online entries are appended after the proxy's, not interleaved).
                data["active_requests"].sort(key=lambda r: r.get("started_at") or 0)
                data["completed_requests"].sort(key=lambda r: r.get("completed_at") or 0)
                return jsonify(data)
            else:
                return jsonify({"error": f"Proxy returned status {resp.status_code}"}), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to fetch requests telemetry from proxy: {e!s}"}), 500


@app.route("/api/requests/clear", methods=["POST"])
def clear_completed_requests():
    """Clear completed requests buffer in the proxy"""
    import httpx

    proxy_url = _find_proxy_url()

    if not proxy_url:
        return jsonify({"error": "Could not connect to any proxy endpoints."}), 503

    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.post(f"{proxy_url}/admin/requests/clear")
            if resp.status_code == 200:
                from online_providers import clear_completed_online_requests

                clear_completed_online_requests()
                return jsonify(resp.json())
            else:
                return jsonify({"error": f"Proxy returned status {resp.status_code}"}), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to clear requests history in proxy: {e!s}"}), 500


@app.route("/api/requests/cancel", methods=["POST"])
def cancel_stuck_request():
    """Cancel a stuck/active request in the proxy (searches all proxies)"""
    import httpx

    data = request.get_json() or {}
    request_id = data.get("request_id")

    if not request_id:
        return jsonify({"error": "request_id is required"}), 400

    from online_providers import cancel_online_request

    if request_id.startswith("online-") and cancel_online_request(request_id):
        return jsonify({"status": "cancelled", "request_id": request_id, "model": "online"})

    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=3.0) as client:
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

    from online_providers import get_online_requests, online_model_provider

    # Online requests live only in the in-process tracker (never reach any proxy)
    online_data = get_online_requests()
    online_req = None
    for r in online_data.get("active_requests", []) + online_data.get("completed_requests", []):
        if r.get("request_id") == request_id:
            online_req = r
            break
    if online_req is not None:
        prompt = online_req.get("prompt", "")
        model = online_req.get("model", "")
        if not prompt:
            return jsonify({"error": "Online request has no prompt to resubmit"}), 400
        try:
            result = asyncio.run(
                online_model_provider.query_online_model(
                    model_identifier=model,
                    prompt=prompt,
                    max_tokens=4000,
                    request_source="web",
                    client_ip="web",
                )
            )
            return jsonify({"status": "resubmitted", "result": result})
        except Exception as e:
            return jsonify({"error": f"Failed to resubmit online request: {e!s}"}), 500

    for url in benchmark.PROXY_SERVER_URLS:
        try:
            with httpx.Client(timeout=3.0) as client:
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
                    all_requests = all_data.get("active_requests", []) + all_data.get("completed_requests", [])
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
                    return jsonify({"error": f"Proxy returned status {resp.status_code}"}), resp.status_code

                result = resp.json()

                with contextlib.suppress(Exception):
                    client.delete(f"{url}/admin/requests/{request_id}")

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

            proxy_url = _find_proxy_url()
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
        for search_dir in [
            telemetry_dir,
            Path("/app/data/telemetry"),
            Path("web").parent / "data" / "telemetry",
        ]:
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
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    points.append(json.loads(line))
        return jsonify({"model": model, "history": points[-limit:]})
    except Exception as e:
        return jsonify({"error": f"Failed to read telemetry: {e!s}"}), 500


@app.route("/api/telemetry/recommendations")
def get_telemetry_recommendations():
    """Run configuration analyzer and fetch recommendations"""
    model = request.args.get("model")
    strategy = request.args.get("strategy", "performance")

    if not model:
        # Default to currently loaded model
        try:
            import httpx

            proxy_url = _find_proxy_url()
            if proxy_url:
                with httpx.Client(timeout=1.0) as client:
                    resp = client.get(f"{proxy_url}/admin/runtime")
                    if resp.status_code == 200:
                        model = resp.json().get("active_model")
        except Exception:
            pass

    if not model:
        return jsonify(
            {
                "status": "insufficient_data",
                "model_alias": "None",
                "detected_issues": ["No active model running, and no model specified."],
                "recommendations": {},
                "explanation": "Please load a model or specify one via '?model=name'.",
            }
        )

    import re

    sanitized_model = re.sub(r"[^\w\-.\.]", "_", model)
    try:
        from analyzer import analyze_telemetry

        perf_first = strategy == "performance"
        analysis = analyze_telemetry(sanitized_model, performance_first=perf_first)
        return jsonify(analysis)
    except Exception as e:
        # Fallback analysis if importer/analyzer fails
        return jsonify(
            {
                "status": "error",
                "model_alias": model,
                "detected_issues": [f"Tuning analyzer engine failed: {e!s}"],
                "recommendations": {},
                "explanation": "Ensure analyzer.py is mounted correctly in the web container path.",
            }
        )


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
                if (
                    clean_target in clean_sec
                    or clean_sec in clean_target
                    or clean_target.replace("latest", "") in clean_sec
                ):
                    profile_stem = entry.stem
                    break

    profile_path = target_dir / f"{profile_stem}.profile.json"

    try:
        profile_data = {}
        if profile_path.exists():
            with open(profile_path, encoding="utf-8") as f:
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

        return jsonify(
            {
                "status": "success",
                "message": f"Applied tuning properties for {model} {ini_msg}.",
                "applied": recommendations,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Failed to apply tuning properties: {e!s}"}), 500


@app.route("/api/analyze/all")
def analyze_all_models():
    """Run resource usage analysis across all models with telemetry data.

    Compares actual VRAM/RAM/GPU utilization against current profile settings
    and returns prioritized optimization recommendations for each model.
    """
    telemetry_dir = Path(os.getenv("TELEMETRY_DIR", "data/telemetry"))
    strategy = request.args.get("strategy", "performance")

    if not telemetry_dir.exists():
        return jsonify({"error": "Telemetry directory not found", "models": []}), 404

    try:
        from analyzer import analyze_telemetry
    except ImportError as e:
        return jsonify({"error": f"Analyzer module unavailable: {e}", "models": []}), 500

    perf_first = strategy != "safe"
    results = []
    skipped = []

    for jsonl_file in sorted(telemetry_dir.glob("*.jsonl")):
        model_alias = jsonl_file.stem
        # Skip non-model files
        if model_alias in ("none", "system_idle", "unknown_model"):
            skipped.append(model_alias)
            continue

        try:
            analysis = analyze_telemetry(model_alias, performance_first=perf_first)

            # Skip models with no data
            if analysis.get("status") == "insufficient_data":
                skipped.append(model_alias)
                continue

            # Compute an optimization priority score for sorting:
            # Higher score = more urgent / impactful to act on
            metrics = analysis.get("metrics_summary", {})
            vram = metrics.get("vram", {})
            ram = metrics.get("system_ram", {})
            recs = analysis.get("recommendations", {})

            vram_pct = vram.get("max_pct", 0)
            vram_headroom = vram.get("headroom_mb", 0)
            ram_pct = ram.get("max_pct", 0)
            n_recommendations = len(recs)

            # Priority: critical OOM issues score highest, then optimization opportunities
            status = analysis.get("status", "ok")
            priority_score = 0
            if status == "critical":
                priority_score = 100
            elif status == "warning":
                priority_score = 60
            elif n_recommendations > 0:
                # Optimization opportunity: score based on potential gain
                # More VRAM headroom with partial GPU offload = bigger opportunity
                priority_score = min(50, int(vram_headroom / 100))

            results.append(
                {
                    "model_alias": model_alias,
                    "status": status,
                    "priority_score": priority_score,
                    "vram_summary": {
                        "total_mb": vram.get("total_mb", 0),
                        "used_mb": vram.get("max_used_mb", 0),
                        "headroom_mb": vram_headroom,
                        "max_pct": round(vram_pct, 1),
                    },
                    "ram_summary": {
                        "max_pct": round(ram_pct, 1),
                        "mean_pct": round(ram.get("mean_pct", 0), 1),
                    },
                    "gpu_util_pct": metrics.get("gpu_util_pct", {}),
                    "current_config": analysis.get("recommendations", {}),
                    "recommendations": recs,
                    "detected_issues": analysis.get("detected_issues", []),
                    "explanation": analysis.get("explanation", ""),
                    "baseline_comparison": analysis.get("baseline_comparison", {}),
                }
            )
        except Exception as e:
            app.logger.warning(f"Analysis failed for {model_alias}: {e}")
            skipped.append(model_alias)

    # Sort by priority (highest first), then by VRAM headroom descending for same priority
    results.sort(key=lambda r: (-r["priority_score"], -r["vram_summary"]["headroom_mb"]))

    return jsonify(
        {
            "strategy": strategy,
            "models_analyzed": len(results),
            "models_skipped": skipped,
            "results": results,
        }
    )


def _get_currently_loaded_model():
    """Return the name of the model currently loaded in the proxy, or None."""
    import httpx

    proxy_url = _find_proxy_url()
    if not proxy_url:
        return None
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{proxy_url}/admin/runtime")
            if resp.status_code == 200:
                loaded = resp.json().get("loaded_models", [])
                if loaded:
                    return loaded[0].get("name") or loaded[0].get("backend_model")
    except Exception:
        pass
    return None


@app.route("/api/routing/matrix", methods=["GET", "POST"])
def get_or_post_routing_matrix():
    """GET current model capability routing matrix, or POST modifications to it"""
    matrix_file = Path("data/routing_matrix.json")
    if not matrix_file.parent.exists():
        matrix_file = Path("web").parent / "data" / "routing_matrix.json"

    # Default routing matrix template - no hardcoded models: each task starts
    # unconfigured and the routing endpoint falls back to the currently loaded
    # model until the user assigns one.
    default_matrix = {
        "fast_chat": {
            "model": None,
            "description": "Sub-second latency chat for voice assistant or general conversation.",
            "min_tps": 40.0,
            "max_ttft_ms": 250,
            "reasoning_required": False,
        },
        "complex_coding": {
            "model": None,
            "description": "Accurate syntax completions, code editing, and structural debugging.",
            "min_tps": 20.0,
            "max_ttft_ms": 500,
            "reasoning_required": False,
        },
        "reasoning": {
            "model": None,
            "description": "Deep thinking, logical reasoning, multi-step problem solving, math/science.",
            "min_tps": 15.0,
            "max_ttft_ms": 800,
            "reasoning_required": True,
        },
        "summarization": {
            "model": None,
            "description": "Document parsing, entity extraction, context summaries, and long context tasks.",
            "min_tps": 30.0,
            "max_ttft_ms": 300,
            "reasoning_required": False,
        },
    }

    if request.method == "POST":
        data = request.get_json() or {}
        try:
            matrix_file.parent.mkdir(parents=True, exist_ok=True)
            with open(matrix_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return jsonify(
                {
                    "status": "success",
                    "message": "Routing matrix updated successfully.",
                    "matrix": data,
                }
            )
        except Exception as e:
            return jsonify({"error": f"Failed to save routing matrix: {e!s}"}), 500

    # GET method
    if matrix_file.exists():
        try:
            with open(matrix_file, encoding="utf-8") as f:
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
            with open(matrix_file, encoding="utf-8") as f:
                matrix = json.load(f)
        except Exception:
            pass

    # Match criteria - no hardcoded defaults: use the task's configured model if
    # one is set, otherwise whatever model is currently loaded in the proxy.
    task_config = matrix.get(task, {})
    loaded_model = _get_currently_loaded_model()

    optimal_model = task_config.get("model")
    if optimal_model:
        explanation = f"Matched to configured model for task type '{task}'."
    else:
        optimal_model = loaded_model
        explanation = f"No model configured for task type '{task}'; falling back to the currently loaded model."

    # If the caller requests specific speed overrides, validate models based on benchmarks
    if optimal_model:
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
                                with open(path) as f:
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
            explanation += f" (Benchmark validation skipped: {e!s})"

    return jsonify(
        {
            "optimal_model": optimal_model,
            "task": task,
            "explanation": explanation,
            "fallback_model": loaded_model or optimal_model,
        }
    )


@app.route("/api/usage")
def get_model_usage():
    """Get model usage statistics from the proxy"""
    import httpx

    proxy_url = _find_proxy_url()

    if not proxy_url:
        return jsonify({"error": "Could not connect to any proxy endpoints."}), 503

    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{proxy_url}/admin/usage")
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                return jsonify({"error": f"Proxy returned status {resp.status_code}"}), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to fetch model usage stats: {e!s}"}), 500


@app.route("/api/models/switch", methods=["POST"])
def switch_model():
    """Switch to a different model via the proxy"""
    import httpx

    data = request.get_json() or {}
    model = data.get("model")
    if not model:
        return jsonify({"error": "model is required"}), 400

    proxy_url = _find_proxy_url()

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
        return jsonify({"error": f"Failed to switch model: {e!s}"}), 500


@app.route("/api/models/unload", methods=["POST"])
def unload_model():
    """Unload a model via the proxy"""
    import httpx

    data = request.get_json() or {}
    model = data.get("model")
    if not model:
        return jsonify({"error": "model is required"}), 400

    proxy_url = _find_proxy_url()

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
        return jsonify({"error": f"Failed to unload model: {e!s}"}), 500


@app.route("/api/vram/clear", methods=["POST"])
def clear_vram():
    """Clear VRAM via the proxy"""
    import httpx

    proxy_url = _find_proxy_url()

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
        return jsonify({"error": f"Failed to clear VRAM: {e!s}"}), 500


@app.route("/api/errors")
def get_model_errors():
    """Proxy to /admin/errors on the proxy - returns recent structured model error log."""
    import httpx

    model = request.args.get("model")
    error_type = request.args.get("error_type")
    limit = request.args.get("limit", "100")

    proxy_url = _find_proxy_url()

    if not proxy_url:
        # Fall back to reading the JSONL file directly if proxy is unreachable
        errors_file = Path(os.getenv("DATA_DIR", "data")) / "model_errors.jsonl"
        if not errors_file.exists():
            return jsonify({"total": 0, "error_type_counts": {}, "errors": []})
        try:
            records = []
            with open(errors_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            if model:
                records = [r for r in records if model.lower() in (r.get("model") or "").lower()]
            if error_type:
                records = [r for r in records if r.get("error_type") == error_type]
            records = records[-int(limit) :][::-1]
            counts: dict = {}
            for r in records:
                t = r.get("error_type", "unknown")
                counts[t] = counts.get(t, 0) + 1
            return jsonify({"total": len(records), "error_type_counts": counts, "errors": records})
        except Exception as e:
            return jsonify({"error": f"Failed to read error log: {e!s}"}), 500

    try:
        params = {"limit": limit}
        if model:
            params["model"] = model
        if error_type:
            params["error_type"] = error_type
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{proxy_url}/admin/errors", params=params)
            if resp.status_code == 200:
                return jsonify(resp.json())
            else:
                return jsonify({"error": resp.text}), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to fetch errors: {e!s}"}), 500


@app.route("/api/errors/clear", methods=["POST"])
def clear_model_errors():
    """Clear in-memory error buffer via the proxy."""
    import httpx

    proxy_url = _find_proxy_url()

    if not proxy_url:
        return jsonify({"error": "Proxy unreachable"}), 503
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(f"{proxy_url}/admin/errors/clear")
            return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/delete", methods=["POST"])
def delete_model():
    """Delete a model and clean up blobs via the proxy"""
    import httpx

    data = request.get_json() or {}
    model = data.get("model")
    if not model:
        return jsonify({"error": "model is required"}), 400

    proxy_url = _find_proxy_url()

    if not proxy_url:
        return jsonify({"error": "Could not connect to any proxy endpoints."}), 503

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{proxy_url}/admin/models/delete", json={"model": model})
            if resp.status_code == 200:
                response_data = dict(resp.json())
                # The model is gone from disk, so its tracker entry is stale.
                # Prune it unconditionally; if benchmark history files were kept,
                # scan_historical_benchmarks() will still surface the model as
                # "previously benchmarked" from the on-disk run snapshots.
                with contextlib.suppress(Exception):
                    model_tracker.delete_model(model)
                if data.get("remove_benchmarks"):
                    purge_info = _purge_model_benchmarks(model)
                    response_data["benchmark_results_removed"] = purge_info
                return jsonify(response_data)
            else:
                try:
                    err_msg = resp.json().get("detail", resp.text)
                except Exception:
                    err_msg = resp.text
                return jsonify({"error": err_msg}), resp.status_code
    except Exception as e:
        return jsonify({"error": f"Failed to delete model: {e!s}"}), 500


@app.route("/api/models/search", methods=["POST"])
def search_models():
    """Search for models on Ollama Registry and Hugging Face"""
    import re
    from html import unescape

    import httpx

    # Keywords that identify Stable Diffusion / image-generation models
    _SD_NAME_KEYWORDS = [
        "stable-diffusion",
        "sdxl",
        "sd1.",
        "sd2.",
        "sd3",
        "flux",
        "pony",
        "photoreal",
        "sd-",
        "illustrious",
        "diffusion",
        "imagen",
        "dalle",
        "kandinsky",
        "playground",
        "waifu-diffusion",
    ]
    _SD_HF_TAGS = {
        "diffusers",
        "stable-diffusion",
        "text-to-image",
        "image-generation",
        "stable-diffusion-xl",
        "stable-diffusion-3",
        "flux",
        "image-to-image",
    }

    def _detect_model_type(name: str, tags: list) -> str:
        """Returns 'stable-diffusion' or 'llm' based on name and HF tags."""
        name_lower = name.lower()
        if any(kw in name_lower for kw in _SD_NAME_KEYWORDS):
            return "stable-diffusion"
        if any(t in _SD_HF_TAGS for t in tags):
            return "stable-diffusion"
        return "llm"

    data = request.get_json() or {}
    query = data.get("query")
    source = data.get("source", "all")  # "ollama", "huggingface", or "all"
    type_filter = data.get("type", "all")  # "all", "llm", "stable-diffusion"

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
                for library_name, _span_name, desc in matches:
                    clean_name = library_name.strip()
                    clean_desc = unescape(desc.strip().replace("\n", " "))
                    model_type = _detect_model_type(clean_name, [])
                    results.append(
                        {
                            "name": clean_name,
                            "description": clean_desc,
                            "source": "ollama",
                            "type": model_type,
                            "downloads": None,
                            "likes": None,
                            "tags": [],
                            "author": None,
                        }
                    )
        except Exception as e:
            print(f"Ollama search error: {e}")

    # 2. Hugging Face Search - include both GGUF (LLM) and all types (for SD)
    if source in ("huggingface", "all"):
        try:
            # Search GGUF models (primarily LLMs)
            url_gguf = f"https://huggingface.co/api/models?search={query}&filter=gguf&limit=20"
            resp_gguf = httpx.get(url_gguf, headers={"User-Agent": "Mozilla/5.0"}, timeout=10.0)

            # Also search for diffusers/SD models
            url_sd = f"https://huggingface.co/api/models?search={query}&filter=diffusers&limit=10"
            resp_sd = httpx.get(url_sd, headers={"User-Agent": "Mozilla/5.0"}, timeout=10.0)

            seen_ids: set = set()

            def _process_hf_item(model_item: dict, forced_type: str | None = None) -> dict | None:
                model_id = model_item.get("id")
                if not model_id or model_id in seen_ids:
                    return None
                seen_ids.add(model_id)
                tags = model_item.get("tags", [])
                model_type = forced_type or _detect_model_type(model_id, tags)
                downloads = model_item.get("downloads", 0)
                likes = model_item.get("likes", 0)
                author = model_item.get("author", "")
                display_tags = [
                    t for t in tags if t not in ("gguf", "diffusers", "transformers", "pytorch", "safetensors")
                ][:5]
                desc = f"Repository by {author}. Downloads: {downloads:,} | Likes: {likes:,}"
                return {
                    "name": model_id,
                    "description": desc,
                    "source": "huggingface",
                    "type": model_type,
                    "downloads": downloads,
                    "likes": likes,
                    "tags": display_tags,
                    "author": author,
                }

            if resp_gguf.status_code == 200:
                for item in resp_gguf.json():
                    entry = _process_hf_item(item)
                    if entry:
                        results.append(entry)

            if resp_sd.status_code == 200:
                for item in resp_sd.json():
                    entry = _process_hf_item(item, forced_type="stable-diffusion")
                    if entry:
                        results.append(entry)

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
                if model_id and not any(r["name"] == model_id for r in results):
                    tags = model_item.get("tags", [])
                    model_type = _detect_model_type(model_id, tags)
                    downloads = model_item.get("downloads", 0)
                    likes = model_item.get("likes", 0)
                    author = model_item.get("author", "")
                    display_tags = [
                        t for t in tags if t not in ("gguf", "diffusers", "transformers", "pytorch", "safetensors")
                    ][:5]
                    results.insert(
                        0,
                        {
                            "name": model_id,
                            "description": f"[Direct Match] Repository by {author}. Downloads: {downloads:,} | Likes: {likes:,}",
                            "source": "huggingface",
                            "type": model_type,
                            "downloads": downloads,
                            "likes": likes,
                            "tags": display_tags,
                            "author": author,
                        },
                    )
        except Exception as e:
            print(f"Precise HF lookup error: {e}")

    # Apply optional type filter
    if type_filter in ("llm", "stable-diffusion"):
        results = [r for r in results if r.get("type") == type_filter]

    return jsonify({"results": results})


@app.route("/api/models/huggingface/files", methods=["GET"])
def get_hf_files():
    """List downloadable model files (.gguf and .safetensors) in a Hugging Face repository"""
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
                jsonify({"error": f"Failed to fetch model info from Hugging Face: {resp.text}"}),
                resp.status_code,
            )

        model_info = resp.json()
        tags = model_info.get("tags", [])
        # Detect if this repo is a Stable Diffusion / image generation repo
        _SD_HF_TAGS = {
            "diffusers",
            "stable-diffusion",
            "text-to-image",
            "image-generation",
            "stable-diffusion-xl",
            "flux",
        }
        _SD_NAME_KW = [
            "stable-diffusion",
            "sdxl",
            "sd1.",
            "sd2.",
            "sd3",
            "flux",
            "pony",
            "photoreal",
            "sd-",
            "illustrious",
            "diffusion",
        ]
        is_sd_repo = any(t in _SD_HF_TAGS for t in tags) or any(kw in repo.lower() for kw in _SD_NAME_KW)

        siblings = model_info.get("siblings", [])
        model_files = []
        for s in siblings:
            fname = s.get("rfilename", "")
            is_gguf = fname.endswith(".gguf")
            is_safetensors = fname.endswith(".safetensors")

            # Include .gguf for all repos; include .safetensors only for SD repos
            if not is_gguf and not (is_sd_repo and is_safetensors):
                continue

            size = s.get("size")
            size_str = ""
            if size:
                size_str = f"{size / 1024**3:.2f} GB" if size > 1024**3 else f"{size / 1024**2:.1f} MB"

            file_type = "stable-diffusion" if (is_safetensors or is_sd_repo) else "llm"
            model_files.append(
                {
                    "filename": fname,
                    "size": size_str,
                    "type": file_type,
                    "format": "safetensors" if is_safetensors else "gguf",
                }
            )

        return jsonify({"files": model_files, "repo_type": "stable-diffusion" if is_sd_repo else "llm"})
    except Exception as e:
        return jsonify({"error": f"Error fetching Hugging Face files: {e!s}"}), 500


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
                jsonify({"error": f"Failed to fetch tags from Ollama Library: {resp.text}"}),
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
        return jsonify({"error": f"Error fetching Ollama model tags: {e!s}"}), 500


@app.route("/api/models/pull", methods=["POST"])
def trigger_model_pull():
    """Trigger a background pull of a model via alpaca-puller.py"""
    data = request.get_json() or {}
    model = data.get("model")
    source = data.get("source", "auto")
    local_name = data.get("local_name")
    no_resume = data.get("no_resume", False)
    companion = data.get("companion", False)

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
            "logs": [],
        }

    t = threading.Thread(target=run_puller_thread, args=(model, source, local_name, no_resume, companion), daemon=True)
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
        return jsonify(
            {
                "active_pulls": {
                    k: {
                        "model": v["model"],
                        "source": v["source"],
                        "local_name": v.get("local_name", ""),
                        "status": v.get("status", "running"),
                        "logs": v["logs"],
                    }
                    for k, v in active_pulls.items()
                }
            }
        )


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

    return jsonify({"status": "stopping", "message": f"Stopping pull for {pull['model']}..."})


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

    return jsonify({"status": "cancelled", "message": f"Pull for {pull['model']} cancelled."})


@socketio.on("connect")
def handle_connect():
    print("Client connected")
    with active_run_lock:
        socketio.emit("sync_status", dict(active_run))


# ---------------------------------------------------------------------------
# Code sandbox terminal (Test Browser "Run" for text-based languages)
#
# On Run we spin up a short-lived Docker container (alpaca-sandbox image,
# locked down: no network, non-root, memory/pid limits) and stream a real
# Python/Node process over a SocketIO channel so the browser gets a small
# interactive terminal where multiple inputs/outputs can be exercised.
# ---------------------------------------------------------------------------
SANDBOX_IMAGE = "alpaca-sandbox:latest"
SANDBOX_TIMEOUT = 600  # seconds before a run is force-killed

_sandbox_runs: dict[str, "SandboxRun"] = {}
_sandbox_lock = threading.Lock()


def detect_language(code, lang_hint=None):
    if lang_hint in ("python", "py"):
        return "python"
    if lang_hint in ("node", "js", "javascript"):
        return "node"
    low = code.lower()
    py_score = low.count("def ") + low.count("import ") + low.count("print(") + low.count("self.")
    js_score = (
        low.count("console.log")
        + low.count("function ")
        + low.count("=>")
        + low.count("require(")
        + low.count("document.")
    )
    if py_score == 0 and js_score == 0:
        return "python"
    return "python" if py_score >= js_score else "node"


class SandboxRun:
    def __init__(self, sid, code, language, initial_input):
        self.sid = sid
        self.language = language
        self.run_id = uuid.uuid4().hex[:12]
        self.client = docker.DockerClient(base_url="unix:///var/run/docker.sock")
        self.container = None
        self.sock = None
        self.reader = None
        self.watchdog = None
        self.alive = True
        self.send_lock = threading.Lock()
        ext = "py" if language == "python" else "js"
        bin_ = "python3" if language == "python" else "node"
        try:
            self.container = self.client.containers.run(
                SANDBOX_IMAGE,
                command=["sleep", "3600"],
                detach=True,
                tty=False,
                stdin_open=True,
                network_mode="none",
                mem_limit="256m",
                pids_limit=128,
                user="sandbox",
                working_dir="/sandbox",
                environment={"PYTHONUNBUFFERED": "1", "NODE_DISABLE_COLORS": "1"},
                name=f"alpaca-sandbox-{self.run_id}",
                remove=False,
            )
            # write the code file into the container
            tar_bytes = io.BytesIO()
            with tarfile.open(fileobj=tar_bytes, mode="w") as tf:
                data = code.encode("utf-8")
                info = tarfile.TarInfo(name=f"code.{ext}")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            self.container.put_archive("/sandbox", tar_bytes.getvalue())
            # Run the code as an interactive exec. With socket=True the SDK
            # returns a SocketIO over the hijacked connection: it is read-only
            # for output, but the underlying socket is duplex, so writing to
            # _sock delivers stdin to the process (tty=True, raw stream).
            res = self.container.exec_run(
                [bin_, f"/sandbox/code.{ext}"],
                stdout=True,
                stderr=True,
                stdin=True,
                tty=True,
                socket=True,
                stream=False,
            )
            self.sock = res.output
            self.raw = self.sock._sock
            self.reader = threading.Thread(target=self._pump, daemon=True)
            self.reader.start()
            self.watchdog = threading.Thread(target=self._watchdog, daemon=True)
            self.watchdog.start()
            if initial_input:
                for line in initial_input.split("\n"):
                    self.send(line + "\n")
        except Exception as e:
            self._emit("sandbox_error", f"Failed to start sandbox: {e}")
            self.cleanup()

    def _pump(self):
        ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
        try:
            while self.alive:
                data = self.sock.read(4096)
                if not data:
                    break
                text = data.decode("utf-8", "replace")
                text = text.replace("\r", "")
                text = ANSI_RE.sub("", text)
                if text:
                    self._emit("sandbox_out", text)
        except Exception as e:
            if self.alive:
                app.logger.warning(f"Sandbox pump read error: {e}")
                self._emit("sandbox_error", f"Sandbox stream error: {e}")
        finally:
            self._emit("sandbox_done", "")
            self.cleanup()

    def _watchdog(self):
        deadline = time.time() + SANDBOX_TIMEOUT
        while self.alive and time.time() < deadline:
            time.sleep(2)
        if self.alive:
            self._emit("sandbox_error", "Sandbox timed out and was stopped.")
            self.cleanup()

    def send(self, text):
        if not self.raw:
            return
        with contextlib.suppress(Exception), self.send_lock:
            self.raw.send(text.encode("utf-8", "replace"))

    def _emit(self, event, payload):
        with contextlib.suppress(Exception):
            socketio.emit(event, payload, to=self.sid)

    def cleanup(self):
        if not self.alive:
            return
        self.alive = False
        with contextlib.suppress(Exception):
            if self.raw:
                self.raw.close()
        with contextlib.suppress(Exception):
            if self.container:
                self.container.kill()
        with contextlib.suppress(Exception):
            if self.container:
                self.container.remove(force=True)
        with contextlib.suppress(Exception):
            if self.client:
                self.client.close()


@socketio.on("sandbox_run")
def on_sandbox_run(data):
    data = data or {}
    sid = request.sid
    code = data.get("code") or ""
    lang = detect_language(code, data.get("lang"))
    initial = data.get("input") or ""
    with _sandbox_lock:
        prev = _sandbox_runs.get(sid)
        if prev:
            prev.cleanup()
        run = SandboxRun(sid, code, lang, initial)
        _sandbox_runs[sid] = run
    socketio.emit("sandbox_started", {"lang": lang}, to=sid)


@socketio.on("sandbox_input")
def on_sandbox_input(data):
    data = data or {}
    sid = request.sid
    run = _sandbox_runs.get(sid)
    if run and run.alive:
        run.send(str(data.get("text") or ""))


@socketio.on("sandbox_kill")
def on_sandbox_kill():
    sid = request.sid
    with _sandbox_lock:
        run = _sandbox_runs.pop(sid, None)
    if run:
        run.cleanup()


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    with _sandbox_lock:
        run = _sandbox_runs.pop(sid, None)
    if run:
        run.cleanup()


if __name__ == "__main__":
    # debug/use_reloader MUST stay off: the reloader restarts the process whenever a
    # watched .py file changes, which would kill the in-process benchmark thread mid-run.
    # Long-running benchmarks are protected by per-model incremental saves + resume instead.
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)

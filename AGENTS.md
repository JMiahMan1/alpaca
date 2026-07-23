# AGENTS.md — Alpaca LLM Benchmark Suite

## Project Overview
Alpaca is an LLM model management, benchmarking, and telemetry dashboard for local GPU deployments. It provides:
- **Model pull/download** from Ollama Registry and Hugging Face GGUF repos
- **Router-based model management** with automatic hot-swap between models
- **Functional & performance benchmarking** across SharedLLM task categories
- **Real-time telemetry** (VRAM, RAM, context usage) with self-healing triggers
- **Web dashboard** with SocketIO for live status updates

## Architecture & Services (Docker Compose)
The project runs 4 Docker services defined in `docker-compose.yml`:

| Service | Image | Role | Network | Port |
|---------|-------|------|---------|------|
| `llama-server` | `Dockerfile.llama-server` (llama.cpp CUDA) | Direct GPU inference server | Compose network | 8080 |
| `sd-server` | `Dockerfile.sd-server` (stable-diffusion.cpp CUDA) | Direct GPU image generation server | Compose network | 8081 |
| `alpaca-proxy` | `Dockerfile.proxy` (FastAPI/Uvicorn) | Model router, proxy, slot management | **host** | 11434 |
| `alpaca-web` | `Dockerfile.web` (Flask/SocketIO) | Web dashboard, benchmark runner | Compose network | 5000 |
| `alpaca-telemetry` | `Dockerfile.proxy` (async daemon) | Polls metrics, writes telemetry logs | Compose network | — |
| `alpaca-indexer` | `Dockerfile.proxy` (one-shot) | Model reindexing at startup | Compose network | — |

### Key Environment Variables
- `MODELS_DIR` — Path to Ollama models directory (`/models` in containers, `/usr/share/ollama/.ollama/models` on host)
- `ROUTER_MODELS_DIR` — Router symlink directory (`.alpaca-router` / `.alpaca-router`)
- `LLAMA_SERVER_URL` — URL to llama-server (usually `http://llama-server:8080` or `http://localhost:8080`)
- `LLAMA_DOCKER_CONTAINER` — Docker container name for llama-server (used by telemetry to query child process)
- `TELEMETRY_DIR` — Output directory for `.jsonl` telemetry logs
- `SLOTS_CACHE_DIR` — KV cache checkpoint directory
- `PROXY_URL` — URL to alpaca-proxy (must be `http://host.docker.internal:11434` for `alpaca-telemetry` since proxy uses `network_mode: host`)
- `OLLAMA_REGISTRY` — Ollama registry URL (default: `https://registry.ollama.ai/v2`)
- `HUGGING_FACE_TOKEN` — Required for gated/private GGUF repos on Hugging Face

### Cross-Service Communication
- **llama-server** exposes REST API at `/props`, `/slots`, `/completion`, `/embeddings`, etc.
- **alpaca-proxy** sits in front of llama-server, provides `/api/tags`, `/api/chat`, `/admin/runtime`, `/admin/slots`, slot caching, model expiry
- **alpaca-web** communicates with both proxy (model list, benchmark orchestration) and llama-server (telemetry history)
- **alpaca-telemetry** polls proxy runtime + llama-server props/slots, writes per-model `.jsonl` files
- **Network topology**: `alpaca-proxy` is on `host` network; all other services use compose network. Telemetry needs `extra_hosts` + `PROXY_URL=http://host.docker.internal:11434` to reach proxy.

## Key Files & Responsibilities

### Core Python Files
| File | Role |
|------|------|
| `alpaca-puller.py` | CLI tool for pulling/downloading models from Ollama registry or Hugging Face |
| `alpaca-proxy.py` | FastAPI proxy/router managing llama-server lifecycle, model switching, slot allocation, KV cache checkpoints |
| `alpaca_puller.py` | (Secondary) Puller module — check if different from `alpaca-puller.py` |
| `llm_benchmark_suite.py` | `LLMModelBenchmark` class — functional + performance benchmarks for llama.cpp models |
| `web/shared_llm_benchmark.py` | `SharedLLMModelBenchmark` class — benchmark runner for SharedLLM task categories |
| `telemetry_monitor.py` | Async daemon polling llama-server metrics, writes `data/telemetry/{model}.jsonl` |
| `analyzer.py` | Benchmark result analyzer |
| `benchmark-configs.py` | Benchmark test configurations |
| `web/app.py` | Flask web server + SocketIO — dashboard API, benchmark runner, model pull orchestration |

### Web Layer (`web/`)
| File | Role |
|------|------|
| `web/app.py` | Flask backend: all REST API routes, SocketIO events, benchmark execution, model pull orchestration |
| `web/shared_llm_benchmark.py` | SharedLLM benchmark logic (reused by web app) |
| `web/templates/index.html` | Single-page HTML dashboard |
| `web/static/js/dashboard.js` | Frontend logic: model grid, benchmark runner, search modal, SocketIO listeners |
| `web/static/css/style.css` | Styles |

### Configuration & Build
| File | Role |
|------|------|
| `docker-compose.yml` | Service definitions, volumes, network config |
| `Dockerfile.proxy` | Proxy/web builder image (python:3.11-slim + docker-cli + FastAPI deps) |
| `Dockerfile.web` | Web builder image (Flask + SocketIO + dashboard deps) |
| `Dockerfile.llama-server` | llama.cpp server image (CUDA) |
| `llama-server-entrypoint.sh` | Entrypoint script for llama-server container |
| `llama-server-flags.py` | llama-server CLI flags configuration |
| `mypy.ini` | mypy config — `disallow_untyped_defs = False` for `web.*` package |
| `pyproject.toml` | Ruff config: `line-length = 100`, lints `F` (pyflakes) + `I` (isort) |
| `pytest.ini` | Pytest config: `asyncio_default_fixture_loop_scope = function` |
| `requirements-dev.txt` | Dev deps: `ruff`, `pytest`, `pytest-asyncio` |
| `.env` | Secret/token storage (gitignored) — e.g. `HUGGING_FACE_TOKEN` |

## Critical Patterns & Gotchas

### Model Pulling
- Pulls run as **subprocesses** (`alpaca-puller.py pull <model>`) in background threads
- Stop/cancel use `.alpaca-stop/` marker files: `.alpaca-stop/{sanitized_model_name}`
- `_should_stop()` in `alpaca-puller.py` checks marker files and `_STOPPED` flag
- **Important**: Always clean up stop markers on failure/pull completion, or they persist and block future pulls
- Download resume is built-in via HTTP `Range` headers; `--no-resume` forces fresh download
- Hugging Face GGUF imports: single-file download, direct blob copy to Ollama blob store
- Ollama pulls: multi-layer manifest + blob download

### Router Model IDs
- Model IDs in the router use `--` as separator: `qwen3.6-35b-a3b--q4_k_m`
- The proxy stores this in `/admin/runtime` as `backend_model`
- llama-server's `/slots` endpoint **requires** `?model=` query parameter — calling without it returns 400
- Pass the full `backend_model` ID (with `--`) to `/slots` — do NOT strip `--`
- When calling from `alpaca-telemetry`, use `PROXY_URL=http://host.docker.internal:11434` since proxy is on host network

### Telemetry
- `telemetry_monitor.py` runs as async daemon, polls every ~5 seconds
- Fetches from proxy `/admin/runtime` (for model name/backend_model) + llama-server `/props` + `/slots`
- Writes `data/telemetry/{sanitized_model_name}.jsonl`
- Telemetry file lookup: checks exact match, then searches all files for model name substring
- **Gotcha**: When `backend_model` is `None` (model loading/not loaded), skip `/slots` call entirely instead of calling without params

### Benchmark Suite
- `llm_benchmark_suite.py` (`LLMModelBenchmark`) — benchmarks llama.cpp models directly via `/completion`
- `web/shared_llm_benchmark.py` (`SharedLLMModelBenchmark`) — benchmarks via Ollama proxy
- **Thinking model handling**: Both proxy and direct paths must set `"think": False` in request payload and strip `<think>...</think>` blocks from responses
- `strip_thinking()` regex: `r'<think>.*?</think>\s*'` (non-greedy)
- Results saved to `data/llm_benchmarks/shared_llm_benchmarks_{timestamp}_{proxy|direct}.json`

### Web Dashboard
- Uses SocketIO for real-time pull logs and benchmark progress
- Model list fetched from proxy `/api/tags` or llama-server `/api/tags`
- Model search: queries Ollama `/search` HTML + Hugging Face `/api/models?search=` JSON
- Search modal: "Find & Pull Models" — supports Ollama library, HF GGUF, and precise HF repo lookup (`username/repo`)
- **Browser caching**: Flask must serve `Cache-Control: no-store` for `.js`/`.css` files — use `@app.after_request` decorator
- Model pull triggers: `/api/models/pull` POST → starts background thread → emits SocketIO events

### Error Types & Fixes
| Symptom | Cause | Fix |
|---------|-------|-----|
| `/slots` 400 Bad Request | Missing `?model=` param | Always pass `model` query param |
| `/slots` 400 with `--` in model ID | (Old bug — was fixed by passing full ID) | Use `backend_model` as-is with `--` |
| Proxy unreachable from telemetry | Proxy on `host` network, telemetry on compose | Add `extra_hosts` + `PROXY_URL=http://host.docker.internal:11434` |
| Stop marker blocks new pulls | Leftover `.alpaca-stop/{model}` file | Clean up on pull failure/completion |
| Browser shows stale search results | Cached `dashboard.js` | Add `Cache-Control: no-store` via `@app.after_request` |
| Benchmark returns empty for thinking models | Model outputs `<think>` block with no content | Add `"think": False` + `strip_thinking()` |
| Download hangs/stuck | `readline()` blocks forever | Use `select.select()` with 1s timeout |

## Testing
- **Unit/Integration tests**: `test_web_integration.py` (49 tests), `test_puller_unit.py`, `test_proxy_unit.py`
- Run tests: `pytest` (uses `pytest-asyncio`)
- Tests run against live Docker services — ensure `docker compose up` first

## Linting & Type Checking
```bash
# Ruff linting
ruff check .
ruff format .

# mypy (selective strict mode)
mypy web/app.py llm_benchmark_suite.py   # strict (disallow_untyped_defs = True)
mypy web/shared_llm_benchmark.py         # lenient (web.* exempt)
```

## Common Workflows

### Rebuild & Restart Services (Use `sudo` for full model directory & docker socket access)
```bash
sudo docker compose up -d --build alpaca-web       # rebuild web
sudo docker compose up -d --build alpaca-telemetry # rebuild telemetry
sudo docker compose up -d --build alpaca-proxy     # rebuild proxy
sudo docker compose up -d                          # all services
```

### View Logs
```bash
sudo docker compose logs -f alpaca-web
sudo docker compose logs -f alpaca-telemetry
sudo docker compose logs -f alpaca-proxy
sudo docker compose logs -f llama-server
sudo docker compose logs -f sd-server
```

### Clean Stop Markers (if stuck)
```bash
rm -f .alpaca-router/.alpaca-stop/*
```

### Run Benchmarks Manually
```bash
# Direct llama.cpp benchmark
python llm_benchmark_suite.py

# SharedLLM benchmark (via proxy)
# Accessed via web dashboard UI or programmatic call to /api/run/shared_llm
```

### Model Pull from CLI
```bash
python alpaca-puller.py pull qwen3.6-35b-a3b:q4_k_m
python alpaca-puller.py pull username/repo.gguf --source huggingface
python alpaca-puller.py pull model --no-resume   # fresh download
python alpaca-puller.py reindex   # rebuild router symlinks
python alpaca-puller.py remove model_name
```

## Naming Conventions
- Model names: `family:quantization` (e.g., `qwen3.6-35b-a3b:q4_k_m`)
- Router symlink names: `{family}--{quantization}.gguf` (e.g., `qwen3.6-35b-a3b--q4_k_m.gguf`)
- Sanitized names (for filenames/markers): replace `/`, `:`, `.` with `_` → `qwen3.6-35b-a3b_q4_k_m`
- Router model ID: `family--quantization` (double `--` as separator)

## Environment Notes
- Python version: 3.11 (container), 3.14+ (host)
- GPU: NVIDIA (NVIDIA Container Toolkit, CUDA 12)
- Ollama models stored at `/usr/share/ollama/.ollama/models` (host) → `/models` (container)
- Router symlinks at `.alpaca-router/` (local) → `/router-models` (container)
- Data directory: `./data/` (telemetry logs, benchmark results)
- Slots cache: `.slots-cache/`

## Debug Tips
1. **Search returns 0 results**: API works (curl test), browser has stale `dashboard.js` — hard-refresh or check cache headers
2. **Pull fails with "Interrupted"**: Check for leftover `.alpaca-stop/{model}` marker files
3. **400 on `/slots`**: Always pass `?model=` parameter, use full backend_model ID with `--`
4. **Telemetry container can't reach proxy**: Proxy is on `host` network; use `host.docker.internal:11434`
5. **Benchmark empty results for thinking models**: Ensure `"think": False` and `strip_thinking()` applied
6. **Download stuck**: Check if `select.select()` timeout is working, verify `_terminate_process()` is called on cancel
7. **Model not appearing in list**: Run `python alpaca-puller.py reindex` to rebuild router symlinks

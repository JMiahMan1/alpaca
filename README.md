# Alpaca

Alpaca is an Ollama-compatible proxy in front of `llama.cpp` router mode.

The design goal is:

- keep the client-facing API close to Ollama
- keep `llama.cpp` strengths available, including `grammar`, `json_schema`, and lower-level runtime controls
- avoid model switching by mutating Compose or restarting the backend container

## Architecture

- `llama-server` runs as a long-lived router-mode process
- models are discovered from the local Ollama-style manifest/blob store
- the proxy loads and unloads backend models through `llama.cpp` router APIs
- Ollama-style `keep_alive` is enforced in the proxy

This means Alpaca can behave like Ollama from the client perspective while still only keeping one model resident if VRAM is tight.

## Components

- `alpaca-proxy.py`: FastAPI proxy on port `11434`
- `alpaca-puller.py`: standalone pull/remove utility for Ollama-format model storage
- `docker-compose.yml`: router-mode `llama-server` plus proxy wiring
- `alpaca-indexer`: sidecar scanner that mirrors local Ollama-discovered models into Alpaca's router index
- `test_proxy_unit.py`: unit coverage for router resolution and keep-alive lifecycle

## Model Lifecycle

Alpaca now uses `llama.cpp` router-mode APIs instead of rewriting `docker-compose.yml`.

Current lifecycle:

1. Client requests `/api/chat` or `/api/generate` with a `model`.
2. Proxy resolves that model against local manifests and router-visible backend IDs.
3. If `MAX_LOADED_MODELS=1`, the proxy unloads any other loaded backend model first.
4. Proxy calls router `POST /models/load` when the requested model is not already loaded.
5. After the response, the proxy applies Ollama-style `keep_alive` behavior.

Supported `keep_alive` behavior:

- finite durations such as `5m` or `3600`
- `0` to unload immediately after the response
- negative values such as `-1` to keep the model loaded indefinitely

## Supported API Surface

The proxy currently implements:

- `POST /api/chat`
- `POST /api/generate`
- `GET /api/tags`
- `GET /api/ps`
- `POST /api/show`
- `GET /api/version`

## Request Compatibility

### `/api/chat`

Accepted Ollama-style request fields include:

- `model`
- `messages`
- `tools`
- `format`
- `options`
- `stream`
- `think`
- `keep_alive`
- `logprobs`
- `top_logprobs`

Accepted `llama.cpp` passthrough fields include:

- `grammar`
- `json_schema`
- `grammar_lazy`
- `response_format`
- `top_k`
- `top_p`
- `min_p`
- `mirostat`
- `n_ctx`
- `n_predict`

### `/api/generate`

Accepted Ollama-style request fields include:

- `model`
- `prompt`
- `suffix`
- `images`
- `format`
- `system`
- `stream`
- `think`
- `raw`
- `keep_alive`
- `options`
- `logprobs`
- `top_logprobs`
- `context`

Behavior notes:

- requests with `system` or `think` are routed through `llama.cpp` chat completions
- requests with `suffix` or `context` keep completion-oriented behavior
- `format: "json"` and schema objects are translated into structured-output controls for `llama.cpp`

## Response Compatibility

The proxy returns Ollama-shaped responses for chat and generate, including:

- `model`
- `created_at`
- `done`
- `done_reason`
- `total_duration`
- `load_duration`
- `prompt_eval_count`
- `prompt_eval_duration`
- `eval_count`
- `eval_duration`
- `logprobs` when available

For streaming responses, the terminal chunk carries completion metrics.

`/api/tags`, `/api/ps`, and `/api/show` derive metadata from local manifests/config blobs where possible.

## llama.cpp-Specific Strengths

The proxy intentionally preserves backend-native controls when clients send them.

Examples:

- `grammar` for GBNF-constrained output
- `json_schema` for structured generation
- lower-level sampling controls such as `mirostat`, `repeat_penalty`, and `tfs_z`

## Configuration

Environment variables supported by the proxy:

- `LLAMA_SERVER_URL`
- `OLLAMA_BASE`
- `MODEL_NAMESPACE`
- `ENGINE_STARTUP_TIMEOUT_SECONDS`
- `API_VERSION`
- `OLLAMA_KEEP_ALIVE`
- `MAX_LOADED_MODELS`

Important defaults:

- `OLLAMA_KEEP_ALIVE` defaults to `5m`
- `MAX_LOADED_MODELS` defaults to `1`

## Deployment

Bring the stack up:

```bash
sudo docker compose up -d --build
```

The proxy listens at:

```text
http://localhost:11434
```

The bundled Compose file starts `llama-server` in router mode with:

- `--models-dir /router-models`
- `--sleep-idle-seconds 300`

Alpaca keeps a separate router index in `./.alpaca-router`. It does not require creating extra directories inside the Ollama model store.
The `alpaca-indexer` sidecar scans the existing local Ollama manifests and blobs and keeps that router index refreshed automatically.

## Pulling Models

```bash
sudo python3 alpaca-puller.py pull llama3:8b
sudo python3 alpaca-puller.py pull --source huggingface \
  'Qwen/Qwen3.6-35B-A3B-GGUF:Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf' \
  --name qwen3.6-35b-a3b:q4_k_m
sudo python3 alpaca-puller.py remove llama3:8b
sudo python3 alpaca-puller.py remove qwen3.6-35b-a3b:q4_k_m
```

The puller currently supports:

- `pull <model>` from the Ollama registry with resumable layer downloads
- `pull --source huggingface <repo:file.gguf>` to import a Hugging Face GGUF into the local Ollama manifest/blob store with resumable downloads
- Hugging Face references in `hf://repo/file.gguf`, `repo:file.gguf`, or `https://huggingface.co/.../resolve/main/file.gguf` form
- `--name <local-model>` to control the local Ollama model name created for Hugging Face imports
- `reindex` to rebuild the named llama-server router index on demand, though the `alpaca-indexer` sidecar normally handles this automatically
- atomic manifest writes after all required blobs are present
- `remove <model>` with shared-blob protection so blobs used by other local models are not deleted
- `--insecure` for HTTP access to a non-TLS Ollama registry endpoint

The puller writes manifests only after all required blobs are present, which keeps proxy discovery stable.
Each successful pull/import also creates a stable router-visible `.gguf` symlink under `./.alpaca-router`, so `llama-server` router mode can discover local Ollama and Hugging Face models by name without modifying the Ollama model directory layout.

### Puller Usage Notes

Ollama registry pull:

```bash
python3 alpaca-puller.py pull llama3:8b
```

Hugging Face GGUF import with an explicit local name:

```bash
python3 alpaca-puller.py pull --source huggingface \
  'Qwen/Qwen3.6-35B-A3B-GGUF:Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf' \
  --name qwen3.6-35b-a3b:q4_k_m
```

Hugging Face GGUF import with auto-detected source from a URL:

```bash
python3 alpaca-puller.py pull \
  'https://huggingface.co/Qwen/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf' \
  --name qwen3.6-35b-a3b:q4_k_m
```

If you omit `--name` for a Hugging Face import, Alpaca derives one from the repository and GGUF filename. Hugging Face downloads resume from the existing partial file if the transfer is interrupted. For private or gated repositories, set `HF_TOKEN` or `HUGGING_FACE_TOKEN` before running the puller.

After switching `docker-compose.yml` to `--models-dir /router-models`, restart the stack. Existing Ollama models should be indexed automatically within a few seconds by `alpaca-indexer`:

```bash
docker compose up -d --build
```

## Verification

To verify that a model is visible to both native Ollama and Alpaca, run:

```bash
python3 test-alpaca.py qwen3:8b
```

Use the local model name such as `qwen3:8b`. Do not use internal router filenames such as `qwen3--8b.gguf`; those are only implementation details for `llama-server` discovery.

The verifier checks:

- the Ollama manifest/blob files exist on disk
- `ollama list` includes the model
- `ollama show <model>` succeeds
- Alpaca `/api/tags` includes the model

## Testing

Local syntax check:

```bash
python3 -m py_compile alpaca-proxy.py
```

Local unit tests:

```bash
pip install -r requirements-dev.txt
pip install fastapi uvicorn httpx
pytest -q test_proxy_unit.py test_puller_unit.py
```

## GitHub Actions

GitHub Actions runs:

- Python `3.11`
- `pytest -q test_proxy_unit.py test_puller_unit.py`

The workflow file is:

- `.github/workflows/test.yml`

The CI coverage is focused on:

- keep-alive parsing
- router model candidate resolution
- router entry matching
- incomplete manifest handling
- unload-on-switch behavior for `MAX_LOADED_MODELS=1`
- loaded-model filtering for `/api/ps`
- non-stream `/api/chat` request mapping and Ollama-shaped responses
- non-stream `/api/generate` request mapping, chat-backend routing, and `keep_alive` handling
- puller model-name normalization
- safe removal of shared versus unshared blobs

## Current Limits

- compatibility is focused on the implemented Ollama endpoints above, not the full Ollama management API such as create/copy/push/pull/delete
- prompt templating for `/api/generate` is still an approximation, especially for complex Ollama templates
- `MAX_LOADED_MODELS=1` is the safest default for limited VRAM, but it means model switches still incur load latency

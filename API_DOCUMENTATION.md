# Alpaca API Reference

This document provides a comprehensive specification of the API endpoints exposed by **Alpaca**. 

Alpaca runs two primary servers:
1. **Alpaca Proxy (Port `11434`)**: Serves as the primary Ollama & OpenAI compatibility layer, handles model loading/unloading, local prompt filtering (API key/password redaction), and streams inference chunks while logging telemetry.
2. **Web Dashboard Server (Port `5000`)**: Runs the graphical monitor dashboard, triggers standard & SharedLLM benchmarks, manages model profiles (`models.ini`), and exposes consolidated telemetry endpoints.

---

## Table of Contents
1. [Ollama Compatibility APIs (Port 11434)](#1-ollama-compatibility-apis-port-11434)
   - [POST /api/chat](#post-apichat)
   - [POST /api/generate](#post-apigenerate)
   - [GET /api/tags](#get-apitags)
   - [GET /api/ps](#get-apips)
   - [POST /api/show](#post-apishow)
   - [GET /api/version](#get-apiversion)
2. [OpenAI Compatibility APIs (Port 11434)](#2-openai-compatibility-apis-port-11434)
   - [POST /v1/chat/completions](#post-v1chatcompletions)
   - [GET /v1/models](#get-v1models)
   - [POST /v1/completions](#post-v1completions)
   - [POST /v1/embeddings](#post-v1embeddings)
3. [Proxy Administration & Telemetry APIs (Port 11434)](#3-proxy-administration--telemetry-apis-port-11434)
   - [GET /admin/system](#get-adminsystem)
   - [GET /admin/runtime](#get-adminruntime)
   - [GET /admin/slots](#get-adminslots)
   - [GET /admin/metrics](#get-adminmetrics)
   - [GET /admin/requests](#get-adminrequests)
   - [POST /admin/requests/clear](#post-adminrequestsclear)
   - [GET /api/logs](#get-apilogs)
4. [Dashboard Backend APIs (Port 5000)](#4-dashboard-backend-apis-port-5000)
   - [GET /api/status](#get-apistatus)
   - [GET /api/models](#get-apimodels)
   - [POST /api/run](#post-apirun)
   - [POST /api/run/shared_llm](#post-apirunshared_llm)
   - [POST /api/cancel](#post-apicancel)
   - [GET /api/results](#get-apiresults)
   - [GET /api/results/<filename>](#get-apiresultsfilename)
   - [GET /api/profiles](#get-apiprofiles)
   - [POST /api/profiles/save](#post-apiprofilessave)
   - [POST /api/profiles/delete](#post-apiprofilesdelete)
   - [POST /api/proxy/restart](#post-apiproxyrestart)
   - [GET /api/logs/download](#get-apilogsdownload)

---

## 1. Ollama Compatibility APIs (Port 11434)

These endpoints fully implement the standard Ollama API specification. You can point integrations like **OpenWebUI**, **LlamaIndex**, and **LangChain** directly to `http://localhost:11434`.

### POST /api/chat
Generate a chat completion from a model. Support streaming and non-streaming modes. Intercepts reasoning steps (`<thinking>`) and filters sensitive credentials before inference.

* **Headers**: `Content-Type: application/json`
* **JSON Payload Parameters**:
  - `model` (string, required): Model name as registered in the router/INI profiles.
  - `messages` (array of objects, required): Conversation message history. Each message has `role` (`user`, `assistant`, `system`) and `content`.
  - `stream` (boolean, optional): Stream chunks using line-delimited JSON. Defaults to `true`.
  - `keep_alive` (string/integer, optional): Override cache keep-alive TTL (e.g. `"5m"`, `"1h"`, `-1` for infinite).
  - `options` (object, optional): Sub-parameters like `temperature`, `top_p`, `num_predict`, `num_ctx`.

* **Curl Example**:
  ```bash
  curl http://localhost:11434/api/chat -d '{
    "model": "qwen2.5-coder:7b",
    "messages": [
      {"role": "user", "content": "Write a python quicksort."}
    ],
    "stream": false
  }'
  ```

* **JSON Response Example (stream=false)**:
  ```json
  {
    "model": "qwen2.5-coder:7b",
    "created_at": "2026-06-23T19:15:00Z",
    "message": {
      "role": "assistant",
      "content": "Here is quicksort in Python:\n..."
    },
    "done": true,
    "done_reason": "stop",
    "total_duration": 1850000000,
    "load_duration": 12000000,
    "prompt_eval_count": 28,
    "prompt_eval_duration": 420000000,
    "eval_count": 92,
    "eval_duration": 1410000000
  }
  ```

---

### POST /api/generate
Generate a text completion for a given raw prompt. If the prompt contains a `system` prompt or reasoning tags, it redirects formatting logic dynamically.

* **Headers**: `Content-Type: application/json`
* **JSON Payload Parameters**:
  - `model` (string, required): Model name.
  - `prompt` (string, required): Prompt string to complete.
  - `suffix` (string, optional): Text after the generated completion.
  - `system` (string, optional): Override system prompt.
  - `stream` (boolean, optional): Stream response chunks. Defaults to `true`.
  - `keep_alive` (string/integer, optional): Keep alive TTL.

* **Curl Example**:
  ```bash
  curl http://localhost:11434/api/generate -d '{
    "model": "qwen2.5-coder:7b",
    "prompt": "def calculate_pi(n):",
    "stream": false
  }'
  ```

---

### GET /api/tags
Retrieve a list of models available locally in the manifest registry.

* **Curl Example**:
  ```bash
  curl http://localhost:11434/api/tags
  ```

* **JSON Response Example**:
  ```json
  {
    "models": [
      {
        "name": "qwen2.5-coder:7b",
        "modified_at": "2026-06-20T12:00:00Z",
        "size": 4700000000,
        "digest": "sha256:7c8b0933d...",
        "details": {
          "format": "gguf",
          "family": "qwen2",
          "parameter_size": "7B",
          "quantization_level": "Q4_K_M"
        }
      }
    ]
  }
  ```

---

### GET /api/ps
Check currently active (loaded) models running in memory inside `llama-server`.

* **Curl Example**:
  ```bash
  curl http://localhost:11434/api/ps
  ```

* **JSON Response Example**:
  ```json
  {
    "models": [
      {
        "name": "qwen2.5-coder:7b",
        "model": "qwen2.5-coder:7b",
        "size": 4700000000,
        "digest": "sha256:7c8b0933d...",
        "expires_at": "2026-06-23T19:30:00Z",
        "keep_alive": 900
      }
    ]
  }
  ```

---

### POST /api/show
Show configuration details and metadata parameters for a specific local model.

* **Headers**: `Content-Type: application/json`
* **Payload**: `{"name": "qwen2.5-coder:7b"}`

* **Curl Example**:
  ```bash
  curl http://localhost:11434/api/show -d '{"name": "qwen2.5-coder:7b"}'
  ```

---

### GET /api/version
Returns the active API proxy compatibility version.

* **Curl Example**:
  ```bash
  curl http://localhost:11434/api/version
  ```
* **JSON Response Example**:
  ```json
  {"version": "0.1.48"}
  ```

---

## 2. OpenAI Compatibility APIs (Port 11434)

These endpoints provide transparent support for the OpenAI standard. You can configure any OpenAI library to target `http://localhost:11434/v1`.

### POST /v1/chat/completions
Generate a chat completion formatted under the OpenAI specification. Fully processes and filters thinking fields (like `<thinking>` tags or `reasoning_content` objects) and redacts credentials locally.

* **Headers**: `Content-Type: application/json`
* **JSON Payload Parameters**:
  - `model` (string, required): Model name to resolve.
  - `messages` (array of objects, required): Role and content objects.
  - `stream` (boolean, optional): Stream chunks as Server-Sent Events (SSE). Defaults to `false`.
  - `temperature` (number, optional): Temperature setting.

* **Curl Example**:
  ```bash
  curl http://localhost:11434/v1/chat/completions -d '{
    "model": "deepseek-coder:6.7b",
    "messages": [{"role": "user", "content": "Explain binary search."}],
    "stream": false
  }'
  ```

* **JSON Response Example (stream=false)**:
  ```json
  {
    "id": "chatcmpl-a8f9c102",
    "object": "chat.completion",
    "created": 1782221800,
    "model": "deepseek-coder:6.7b",
    "choices": [
      {
        "index": 0,
        "message": {
          "role": "assistant",
          "content": "Binary search is an efficient algorithm..."
        },
        "finish_reason": "stop"
      }
    ],
    "usage": {
      "prompt_tokens": 15,
      "completion_tokens": 42,
      "total_tokens": 57
    }
  }
  ```

---

### GET /v1/models
Retrieve the list of models matching the OpenAI structure.

* **Curl Example**:
  ```bash
  curl http://localhost:11434/v1/models
  ```

---

### POST /v1/completions
Submit raw prompt requests under standard OpenAI text completion structure.

---

### POST /v1/embeddings
Generate embeddings for input strings.

* **Curl Example**:
  ```bash
  curl http://localhost:11434/v1/embeddings -d '{
    "model": "nomic-embed-text",
    "input": "The quick brown fox jumps over the lazy dog."
  }'
  ```

---

## 3. Proxy Administration & Telemetry APIs (Port 11434)

These endpoints provide low-level insights, logs, metrics, active GPU/system details, and completed completions telemetry directly from the running proxy backend.

### GET /admin/system
Queries active hardware indicators from host platforms (CPU load averages, RAM allocations, GPU offloads, VRAM free space).

* **Curl Example**:
  ```bash
  curl http://localhost:11434/admin/system
  ```

---

### GET /admin/runtime
Fetches loaded models in memory, their relative active sizes, their expiration keeps, and active slot mappings.

* **Curl Example**:
  ```bash
  curl http://localhost:11434/admin/runtime
  ```

---

### GET /admin/slots
Retrieves deep speculative and evaluation slots statuses directly from the local `llama-server`.

* **Curl Example**:
  ```bash
  curl http://localhost:11434/admin/slots
  ```

---

### GET /admin/metrics
Returns statistics on proxy prompt evaluations, tokens generated, timings, and throughput averages.

* **Curl Example**:
  ```bash
  curl http://localhost:11434/admin/metrics
  ```

---

### GET /admin/requests
Exposes the active requests list (including reasoning, stream contents) and the 10 most recently completed requests.

* **Curl Example**:
  ```bash
  curl http://localhost:11434/admin/requests
  ```

---

### POST /admin/requests/clear
Clears the finished completed requests log. Active requests are unaffected.

* **Curl Example**:
  ```bash
  curl -X POST http://localhost:11434/admin/requests/clear
  ```

---

### GET /api/logs
Returns raw logs matching standard system processes for the Alpaca daemon stack. Supports a `limit` parameter.

* **Curl Example**:
  ```bash
  curl http://localhost:11434/api/logs?limit=50
  ```

---

## 4. Dashboard Backend APIs (Port 5000)

Exposed by the GUI metrics dashboard web server to run evaluations, save profiles, retrieve compiled results, and fetch proxy metrics.

### GET /api/status
Get status metadata on the currently running benchmark loop.

* **Curl Example**:
  ```bash
  curl http://localhost:5000/api/status
  ```

* **JSON Response Example**:
  ```json
  {
    "status": "idle",
    "type": "general",
    "models": [],
    "tests_completed": 0,
    "total_tests": 0,
    "results": [],
    "current_model": null,
    "current_test": null,
    "current_category": null,
    "start_time": null
  }
  ```

---

### GET /api/models
Retrieve the list of models matching proxy records merged with standard local direct instances.

* **Curl Example**:
  ```bash
  curl http://localhost:5000/api/models
  ```

---

### POST /api/run
Start a model evaluation suite run for specific model files.

* **Headers**: `Content-Type: application/json`
* **Payload Parameters**:
  - `models` (array of strings, required): List of models to evaluate.
  - `use_proxy` (boolean, optional): Route requests through Alpaca proxy. Defaults to `true`.

* **Curl Example**:
  ```bash
  curl http://localhost:5000/api/run -d '{
    "models": ["qwen2.5-coder:7b"],
    "use_proxy": true
  }'
  ```

---

### POST /api/run/shared_llm
Trigger validation suites checking specs for the standard SharedLLM profiles.

* **Curl Example**:
  ```bash
  curl http://localhost:5000/api/run/shared_llm -d '{"models": ["qwen2.5-coder:7b"]}'
  ```

---

### POST /api/cancel
Cancel an active running benchmarking job immediately.

* **Curl Example**:
  ```bash
  curl -X POST http://localhost:5000/api/cancel
  ```

---

### GET /api/results
List all saved evaluation result files from previous benchmarks.

* **Curl Example**:
  ```bash
  curl http://localhost:5000/api/results
  ```

---

### GET /api/results/<filename>
Fetch complete raw details for a specific benchmark JSON result file.

* **Curl Example**:
  ```bash
  curl http://localhost:5000/api/results/benchmarks_20260623_120000.json
  ```

---

### GET /api/profiles
Read custom configuration overrides defined under `models.ini`.

* **Curl Example**:
  ```bash
  curl http://localhost:5000/api/profiles
  ```

---

### POST /api/profiles/save
Save or modify parameters (like `n_ctx`, `n_gpu_layers`, `flash_attn`, or `keep_alive`) for a model profile inside `models.ini`.

* **Headers**: `Content-Type: application/json`
* **Payload Parameters**:
  - `section` (string, required): Profile name (e.g. `qwen2.5-coder:7b` or `*` for defaults).
  - `settings` (object, required): Parameter map to write.

* **Curl Example**:
  ```bash
  curl http://localhost:5000/api/profiles/save -d '{
    "section": "qwen2.5-coder:7b",
    "settings": {
      "n_gpu_layers": "32",
      "flash_attn": "true"
    }
  }'
  ```

---

### POST /api/profiles/delete
Delete a configured profile section from `models.ini`.

* **Headers**: `Content-Type: application/json`
* **Payload**: `{"section": "qwen2.5-coder:7b"}`

* **Curl Example**:
  ```bash
  curl http://localhost:5000/api/profiles/delete -d '{"section": "qwen2.5-coder:7b"}'
  ```

---

### POST /api/proxy/restart
Trigger a safe restart sequence of proxy and backend `llama-server` docker containers.

* **Curl Example**:
  ```bash
  curl -X POST http://localhost:5000/api/proxy/restart
  ```

---

### GET /api/logs/download
Download complete historical proxy logs as an attachment.

* **Curl Example**:
  ```bash
  curl -O -J http://localhost:5000/api/logs/download
  ```

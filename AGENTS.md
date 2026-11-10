# AGENTS.md - Alpaca LLM Benchmark Suite

## Project Overview
Alpaca is an LLM model management, benchmarking, and telemetry dashboard for local GPU deployments. It provides:
- **Model pull/download** from Ollama Registry and Hugging Face GGUF repos
- **Router-based model management** with automatic hot-swap between models
- **Functional & performance benchmarking** across 307 general tests, 49 SharedLLM tasks and one
  multi-turn agentic workflow
- **Real-time telemetry** (VRAM, RAM, context usage) with self-healing triggers
- **Image generation** (stable-diffusion.cpp) with deterministic text composition
- **Audio**: Kokoro TTS, OpenVoice V2 voice cloning, speaker identification, and a procedural
  podcast mixer
- **Web dashboard** with SocketIO for live status updates, an arcade, and a sandbox

## Architecture & Services (Docker Compose)
The project runs 7 Docker services defined in `docker-compose.yml`:

| Service | Image | Role | Network | Port |
|---------|-------|------|---------|------|
| `llama-server` | `Dockerfile.llama-server` (llama.cpp CUDA) | Direct GPU inference server | Compose network | 8080 |
| `sd-server` | `Dockerfile.sd-server` (stable-diffusion.cpp CUDA) | Direct GPU image generation server | Compose network | 8081 |
| `audio-server` | `Dockerfile.audio` (pytorch CUDA) | TTS, music, voice cloning, speaker ID | Compose network | 8082 |
| `alpaca-proxy` | `Dockerfile.proxy` (FastAPI/Uvicorn) | Model router, proxy, slot management | **host** | 11434 |
| `alpaca-web` | `Dockerfile.web` (Flask/SocketIO) | Web dashboard, benchmark runner, podcast mixing | Compose network | 5000 |
| `alpaca-telemetry` | `Dockerfile.proxy` (async daemon) | Polls metrics, writes telemetry logs | Compose network | - |
| `alpaca-indexer` | `Dockerfile.proxy` (one-shot) | Model reindexing at startup | Compose network | - |

**Which files need a rebuild, not a restart:** `web/`, `llm_benchmark_suite.py`, `online_providers.py`,
`analyzer.py`, `sandbox_exec.py`, `benchmark_tests.json`, `manuscript_rubric.py`, `alpaca-puller.py`
and `data/` are bind-mounted, so a change needs only `./scripts/restart-when-idle.sh`.
`alpaca-proxy.py`, `audio_server.py`, `tts_text.py` and `voice_clone.py` are COPYed into their
images and need `sudo docker compose up -d --build <service>`.

### Time zone
- Containers default to UTC and the code writes naive local timestamps (`time.strftime`), so a UTC
  container stamped results 7 hours ahead of the host and the dashboard showed them as the next day.
  Every service passes `TZ: ${TZ:-UTC}`; set `TZ=` in `.env` to the host's `timedatectl` zone
  (this box: `America/Phoenix`). A changed TZ needs the containers recreated (`docker compose up -d`),
  not restarted - and recreating `alpaca-web` kills a running benchmark, so wait until idle.

### Key Environment Variables
- `MODELS_DIR` - Path to Ollama models directory (`/models` in containers, `/usr/share/ollama/.ollama/models` on host)
- `ROUTER_MODELS_DIR` - Router symlink directory (`.alpaca-router` / `.alpaca-router`)
- `LLAMA_SERVER_URL` - URL to llama-server (usually `http://llama-server:8080` or `http://localhost:8080`)
- `AUDIO_SERVER_URL` - URL to audio-server (`http://audio-server:8082` or `http://localhost:8082`)
- `LLAMA_DOCKER_CONTAINER` - Docker container name for llama-server (used by telemetry to query child process)
- `TELEMETRY_DIR` - Output directory for `.jsonl` telemetry logs
- `SLOTS_CACHE_DIR` - KV cache checkpoint directory
- `PROXY_URL` - URL to alpaca-proxy (must be `http://host.docker.internal:11434` for `alpaca-telemetry` since proxy uses `network_mode: host`)
- `OLLAMA_REGISTRY` - Ollama registry URL (default: `https://registry.ollama.ai/v2`)
- `HUGGING_FACE_TOKEN` - Required for gated/private GGUF repos on Hugging Face
- `DOCKER_SOCK` - Docker socket path (`${DOCKER_SOCK:-/var/run/docker.sock}`), set this on colima

### Cross-Service Communication
- **llama-server** exposes REST API at `/props`, `/slots`, `/completion`, `/embeddings`, etc.
- **alpaca-proxy** sits in front of llama-server, provides `/api/tags`, `/api/chat`, `/admin/runtime`, `/admin/slots`, slot caching, model expiry. `POST /admin/restart` restarts llama-server (it cannot restart its own container, and answers that explicitly if asked).
- **audio-server** provides `/api/tts`, `/api/music`, `/api/voices*`, `POST /api/voices/identify` and `/calibrate`. It is the only service with torch, which is why speaker identification lives here and not in the dashboard.
- **alpaca-web** communicates with both proxy (model list, benchmark orchestration) and llama-server (telemetry history), and bridges `/api/audio/*` → audio-server plus the podcast mixer on top of it
- **alpaca-telemetry** polls proxy runtime + llama-server props/slots, writes per-model `.jsonl` files
- **Network topology**: `alpaca-proxy` is on `host` network; all other services use compose network. Telemetry needs `extra_hosts` + `PROXY_URL=http://host.docker.internal:11434` to reach proxy.


## Key Files & Responsibilities

### Core Python Files
| File | Role |
|------|------|
| `alpaca-puller.py` | CLI tool for pulling/downloading models from Ollama registry or Hugging Face |
| `alpaca-proxy.py` | FastAPI proxy/router managing llama-server lifecycle, model switching, slot allocation, KV cache checkpoints |
| `llm_benchmark_suite.py` | `LLMModelBenchmark` class - functional + performance benchmarks for llama.cpp models |
| `online_providers.py` | `OnlineModelProvider` adapter - queries OpenRouter, Hugging Face, Cloudflare Workers AI, OpenCode Zen, and the Claude/Codex/DeepSeek/Pi/Cline CLIs |
| `audio_server.py` | TTS (Kokoro) + music (MusicGen) + OpenVoice cloning + speaker identification, on :8082 |
| `voice_clone.py` | OpenVoice V2 tone-colour cloning, enrolment quality analysis, and cosine-based speaker ID |
| `tts_text.py` | Pure text normalisation for the TTS front end (scripture, dates, roman numerals, lexicon) |
| `multistep_benchmark.py` | The only genuinely multi-turn harness: 4 fixed turns + one healing pass |
| `context_awareness.py` | Resolves the live `n_ctx` and clamps `num_predict`; raises rather than guessing |
| `sandbox_exec.py` | The Docker sandbox: lint, execute, screenshot, serve, noVNC, CLI fixtures |
| `settings_scan.py` | Resource-guided llama.cpp settings search with thermal guards |
| `manuscript_rubric.py` | The John 1 illuminated-manuscript rubric (the scripture travels with the rules) |
| `imageops.py` | Deterministic band-fill + text draw - the only non-diffusion way to put real type on an image |
| `analyzer.py` | Telemetry-driven llama.cpp tuning recommendations |
| `telemetry_monitor.py` | Async daemon polling llama-server metrics, writes `data/telemetry/{model}.jsonl` |
| `benchmark-configs.py` | ctx × cache × flash-attn sweep, then a prefill-only batch sweep, then a quality check |
| `benchmark_tests.json` | **307 tests in 48 categories** - the general suite's data |
| `web/app.py` | Flask web server + SocketIO - dashboard API, benchmark runner, model pull orchestration, podcast routes |
| `web/podcast_mixer.py` | The podcast mixer: script parsing, bed synthesis, ducking, WAV encode - numpy only |

### Web Layer (`web/`)
| File | Role |
|------|------|
| `web/app.py` | Flask backend: all REST API routes, SocketIO events, benchmark execution, model pull orchestration |
| `web/podcast_mixer.py` | Podcast mixing (see **Podcast Studio** below) |
| `web/shared_llm_benchmark.py` | SharedLLM benchmark logic (reused by web app) |
| `web/arcade_publish.py` | Publishes benchmark games into the standalone arcade service (a `mobilegames` game joins the Family collection) |
| `web/model_tracker.py` | Discovery/seen state and benchmark history per model |
| `web/thermal.py` | Thermal watchdog: throttle hysteresis, abort, pre-test cool-down |
| `web/templates/index.html` | Single-page HTML dashboard (10 hash-routed tabs) |
| `web/static/js/dashboard.js` | Frontend logic: model grid, benchmark runner, search modal, Podcast Studio, SocketIO listeners |
| `web/static/css/style.css` | Styles |

### Configuration & Build
| File | Role |
|------|------|
| `docker-compose.yml` | Service definitions, volumes, network config |
| `Dockerfile.proxy` | Proxy/web builder image (python:3.11-slim + docker-cli + FastAPI deps) |
| `Dockerfile.web` | Web builder image (Flask + SocketIO + dashboard deps) |
| `Dockerfile.audio` | Audio server image (pytorch CUDA + espeak-ng + ffmpeg + OpenVoice) |
| `Dockerfile.llama-server` | llama.cpp server image (CUDA) |
| `llama-server-entrypoint.sh` | Entrypoint script for llama-server container |
| `llama-server-flags.py` | llama-server CLI flags configuration |
| `mypy.ini` | mypy config - `disallow_untyped_defs = False` for `web.*` package, `python_version = 3.12` |
| `pyproject.toml` | Ruff config: `line-length = 120`, lints `E,F,W,I,UP,B,SIM,RUF`; pytest config: `testpaths = ["tests"]`, `pythonpath = ["."]`, `asyncio_default_fixture_loop_scope = function`, `addopts = "-q -m 'not live'"`, plus the `markers` and `per-file-ignores` tables |
| `requirements-dev.txt` | Dev deps: `ruff`, `pytest`, `pytest-asyncio`, `types-PyYAML` **plus every runtime import the suite needs** (CI installs only this) |
| `.env` | Secret/token storage (gitignored) - e.g. `HUGGING_FACE_TOKEN` |
| `.env.example` | The required vars a fresh clone cannot start without (`LLAMA_REASONING_BUDGET`, `LLAMA_REASONING_FORMAT`) |

## Critical Patterns & Gotchas

### Model Pulling
- Pulls run as **subprocesses** (`alpaca-puller.py pull <model>`) in background threads
- Stop/cancel use `.alpaca-stop/` marker files: `.alpaca-stop/{sanitized_model_name}`
- `_should_stop()` in `alpaca-puller.py` checks marker files and `_STOPPED` flag
- **Important**: Always clean up stop markers on failure/pull completion, or they persist and block future pulls
- Download resume is built-in via HTTP `Range` headers; `--no-resume` forces fresh download
- Hugging Face GGUF imports: single-file download, direct blob copy to Ollama blob store
- Ollama pulls: multi-layer manifest + blob download
- **Companions are shared by filename** (`.alpaca-router/companions/`), and upstream names are
  generic: Gemma 4's projector is `mmproj-F16.gguf`, the same name qwen2.5-vl's uses. A companion
  pull now refuses to replace a same-named file of a different size; pass `--name <distinct-file>`.
  (Before, it deleted the existing file, silently breaking the other model's vision.)
- **A pull rewrites models.ini** (`import_huggingface_gguf` calls `update_models_ini` itself, not just
  `pull_model`). Never let that happen while `settings_scan.py` runs: the scan writes the whole ini
  from its in-memory copy, so the two erase each other's sections. During a scan use
  `alpaca-puller.py pull <hf ref> --name <n> --download-only` (blob + manifest only), then register
  with `sudo docker compose run --rm alpaca-indexer` once the scan is done.
- **Delete models through the API, never by hand:** dashboard `POST /api/models/delete` → proxy
  `/admin/models/delete` (unloads it, removes the manifest, keeps blobs another manifest shares, drops
  its router link, profiles and models.ini section, reloads the router). If it cannot do the job, fix
  it rather than going around it.
- **A running settings scan owns models.ini** and holds `.alpaca-router/.settings-scan.lock`
  (refreshed on every write, released on exit or SIGTERM; older than an hour = a crashed scan). The
  delete API answers 409 while it is held, and `update_models_ini` skips with a warning instead of
  racing it (skip, not fail: the startup indexer must still exit 0). Before the lock, a section the
  API removed mid-scan would have been written back by the scan's next probe.
- **Reindex inside the container, never on the host.** Router links are relative to the container
  layout (`../models/blobs` = `/models`), so on the host every one of them dangles. A host-side
  `update_models_ini` used to drop the section of every model it could not follow: on 2026-11-09 a
  host-side import wiped models.ini down to the one new model (and gave that model a host-relative
  link the container cannot follow). It now keeps an existing section whose link does not resolve
  and warns; the container's `alpaca-indexer` is still the only correct way to (re)build sections.

### Router Model IDs
- Model IDs in the router use `--` as separator: `qwen3.6-35b-a3b--q4_k_m`
- The proxy stores this in `/admin/runtime` as `backend_model`
- llama-server's `/slots` endpoint **requires** `?model=` query parameter - calling without it returns 400
- Pass the full `backend_model` ID (with `--`) to `/slots` - do NOT strip `--`
- When calling from `alpaca-telemetry`, use `PROXY_URL=http://host.docker.internal:11434` since proxy is on host network

### Telemetry
- `telemetry_monitor.py` runs as async daemon, polls every ~5 seconds
- Fetches from proxy `/admin/runtime` (for model name/backend_model) + llama-server `/props` + `/slots`
- Writes `data/telemetry/{sanitized_model_name}.jsonl`
- Telemetry file lookup: checks exact match, then searches all files for model name substring
- **Gotcha**: When `backend_model` is `None` (model loading/not loaded), skip `/slots` call entirely instead of calling without params
- **Appends are unbuffered and self-repairing.** Each record is one `os.write` of one complete line to an `O_APPEND` fd opened `buffering=0`, and if the file's last byte is not `\n` the new record is prefixed with one. Text-mode `open(..., "a")` buffers in user space, so a kill mid-flush truncates a record mid-key and the next `O_APPEND` seeks to EOF *inside* it - two records on one line. This was not hypothetical: one torn line at `system_idle.jsonl:172` (of 365,326) was failing `/api/telemetry/history` for that model with a 500. The reader skips bad lines rather than failing the file (`skipped_lines`), and the writer prevents new damage.
- **Two independent bounds, so neither can delete the file being appended to.** Age is `prune_old_telemetry()` (mtime older than `TELEMETRY_RETENTION_DAYS`, swept every `TELEMETRY_PRUNE_INTERVAL_S`, once shortly after startup); size is `_rotate_if_oversized()` (`m.jsonl` → `.1` → `.2`, keeping `TELEMETRY_MAX_GENERATIONS`, past `TELEMETRY_MAX_BYTES`). Defaults cap one hot model at ≈ 16 MiB × 4 = 64 MiB; the directory had reached **1.2 GB across 33 files and 42 days** unbounded.
- Rotation failure is contained and logged - housekeeping must never cost a data point.

### Benchmark Suite
- `llm_benchmark_suite.py` (`LLMModelBenchmark`) - benchmarks llama.cpp models directly via `/completion`
- `web/shared_llm_benchmark.py` (`SharedLLMModelBenchmark`) - benchmarks via Ollama proxy and online model providers (`openrouter:`, `huggingface:`, `cloudflare:`, `opencode_zen:`)
- **Game conventions**: games are UI-first (pygame / HTML5 canvas / three.js / raw WebGL). Only
  three CLI games exist - `guess_game`, `text_adventure` (story-driven), `game_checkers_cli`
  (terminal TUI). Every arcade/board game prompt requires scoring + name/initials entry +
  persistent top-5 high-score board (JSON file or localStorage) + score reset to 0 on new game;
  `_has_persistent_scoreboard` enforces all four during grading.
- **Outdated-only runs**: `POST /api/run` with `"outdated_only": true` computes stale test ids
  (hash/prompt mismatch vs `benchmark_tests.json` via `_compute_test_hash`) for the selected
  models and runs only those. Returns `{"status": "No outdated benchmarks"}` when nothing is
  stale. Dashboard button: **⚠ Run Outdated**.
- **Quick Benchmark** (`quick_benchmark` group, listed first in the dashboard): 27 existing tests
  across 13 categories (`QUICK_BENCHMARK_IDS` in `llm_benchmark_suite.py`) for checking a settings
  change or a new model in well under an hour instead of the full suite. It is a *selection*, not a
  category: `expand_benchmark_groups()` in `web/app.py` turns it into real groups + `test_ids`, so
  results file under each test's own category and keep their history and outdated tracking. Other
  groups selected with it run in full; `outdated_only` narrows it. Picked from the stored results
  for **disagreement between models** (keyed tests every model passes - ifeval, knowledge, hle -
  measure nothing and are left out), objective grading, and speed; the frontier_diagnostics set
  is its backbone. Swap a test out when its pass rate across models drifts to 0 or 1.
- **Gap-filling keyed tests** (Nov 2026; answers computed by running code, the schedule
  brute-forced to one solution): `frontier_grounding_absent` (the memo never gives the number, so the
  answer is 0; the traps are 28 = 140-112 and 30) and `frontier_grounding_counterfactual` (the document
  says 410 where habit says 404) are the suite's first grounding/hallucination checks;
  `code_trace_late_binding` and `code_trace_mutable_default` (debugging) test predicting a program's
  output; `logic_schedule` is a unique-solution constraint puzzle. Three `math_hard` items every
  model passed (2^10, a hexagon's angle sum, a diagonal determinant) became `math_hard_last_two_digits`,
  `_inclusion_exclusion` and `_expected_rolls`.
- **New keyed tests are numeric, not multiple choice, unless MC is essential.**
  `rebalance_choice_keys.py` assigns keys greedily over a fixed order, so adding or removing ONE
  letter-keyed test reshuffles most of the others (27 of 39 here), and every rewritten prompt
  re-runs against every model. `test_shipped_file_is_already_rebalanced` catches it.
- **Redundant tests removed (Nov 2026):** `debug_offbyone`, `debug_null_deref`, `debug_data_race`
  and `debug_sql_injection` named the bug in the prompt, while `cr_off_by_one` / `cr_null_deref` /
  `cr_race_condition` / `cr_sql_injection` make the model find the same bug class in real code;
  `game_breakout` duplicated `game_breakout_rt` (both pygame Breakout). Deliberately KEPT: the same
  game in a different language or engine (Space Invaders x6, Pong, Snake, Asteroids), the two mobile
  block puzzles (Family collection, and the deluxe one is a richer spec), and the two Okazaki GPQA
  items (different facts; removing a letter-keyed test also reshuffles the key balance).
- **Sandbox stdin**: CLI games calling `input()` get a scripted `/tmp/stdin.txt` redirect in
  `run_code_once` so they don't EOFError; `pids_limit` is 1024 for UI runs (Chromium), 128 for CLI.
- **SharedLLM Evaluation Tiers**:
  1. FastPath Intent Routing (HA lights, Life360/Geo location, Music Assistant playback, Climate, Raven dispatch)
  2. Librarian Structured Tool Use (Nextcloud directory listing, RAG knowledge search, Geofence checking)
  3. Raven Autonomous Coding (Redis MultiTenantLock, FastAPI async router with Pydantic AST validation, self-healing bug fixes)
  4. Raven Planning & Troubleshooting (Multi-step DAG plan JSON, Music Assistant stream error diagnosis)
  5. Context Retention (Needle in haystack secret token retrieval)
- **Thinking model handling**: Both proxy and direct paths must set `"think": False` in request payload and strip `<think>...</think>` blocks from responses
- `strip_thinking()` regex: `r'<think>.*?</think>\s*'` (non-greedy)
- **Per-benchmark reasoning budgets**: each test carries `reasoning_budget` (2048 heavy / 1024 light / 512 minimal) and an explicit `reasoning_estimate` (tokens a competent model needs for thinking on THAT test; generated by `scripts/gen_reasoning_estimates.py` from task-shape tiers: UI games 4096, runnable-code 3072, deliberate exams 2048, else 1024). **Thinking is decided per MODEL, not per test** (`_effective_thinking`: a profile's `thinking`/`think` toggle, then the benchmark-wide default, then an effective model reasoning budget; with none of those set - the current state of every profile - all tests run thinking-OFF, which is also how SharedLLM calls these models; the per-test `reasoning_budget` no longer switches it on). `_test_num_predict()` returns `base + 2 x reasoning_estimate` when thinking is ON (doubled headroom so think + full answer fit without truncation) and plain base otherwise. Per-result `think`/`reasoning_budget`/`num_predict` fields record which mode ran - fairness note: disabling thinking on light tests can lower scores for reasoning-tuned models.
- **Grader directive v2**: code/UI prompts append a grading notice ("output ONLY final code in one fenced block or fail"); versioned via `LLMModelBenchmark.GRADER_DIRECTIVE_VERSION`, included in `_compute_test_hash` so `outdated_only` re-runs code/ui tests once after directive changes
- **Code extraction** (`sandbox_exec.py extract_clean_code`): prefers fenced blocks that start like real code over plan-fences, handles truncated final fences (n_predict cap), and strict syntax-only prose indicators - prevents reasoning prose leaking into linted code
- **Language selection chain**: `test.lang` > explicit fence tag (`_fence_lang`) > content sniffing (`_infer_lang`); Python-with-embedded-SQL answers run as python, not piped to sqlite3
- **Online provider un-stick**: empty content + `finish_reason=length` + captured thinking is deterministic exhaustion → one phase-2 direct-answer continuation fires, then the retry loop breaks instead of backing off forever
Results saved to `data/shared_llm_benchmarks/shared_llm_benchmarks_{timestamp}_{proxy|direct}.json`

### Web Dashboard
- Uses SocketIO for real-time pull logs and benchmark progress
- Model list fetched from proxy `/api/tags` or llama-server `/api/tags`
- Model search: queries Ollama `/search` HTML + Hugging Face `/api/models?search=` JSON
- Search modal: "Find & Pull Models" - supports Ollama library, HF GGUF, and precise HF repo lookup (`username/repo`)
- **Browser caching**: Flask must serve `Cache-Control: no-store` for `.js`/`.css` files - use `@app.after_request` decorator
- Model pull triggers: `/api/models/pull` POST → starts background thread → emits SocketIO events

### Objective (Keyed) Benchmark Grading
- Tests carrying an `expected` letter or number are graded against the span the model
  **presented as its answer**, not the whole response. `_answer_windows()` yields, most
  explicit first: a `\boxed{}` value, the text after the last `answer:`/`option:` marker,
  then the closing line. `_extract_choice_letters()` / `_extract_answer_numbers()` read
  the first span that names a candidate; when none does, grading falls back to the old
  whole-response search so an unusual presentation is not failed.
- **Why**: "A" and "I" are ordinary English words and a chain of thought enumerates most
  small numbers, so searching the whole response passed wrong answers whose reasoning
  mentioned the key. That measured the grader, not the model.
- A *declared* span (boxed / after a marker / a response of `_TERSE_ANSWER_WORDS` or
  fewer) counts every value it names. A longer response's closing line can still carry
  working ("7 rounds, so the total is 8"), so only its **last** number is the result.
- **Answer-key balance**: `scripts/rebalance_choice_keys.py` permutes multiple-choice
  options so the keys spread across the letters. Before it, 28 of 36 keyed tests answered
  B or C and a model that always replied "B" scored 39%; now no single letter beats 25%.
  The script is idempotent (options are sorted before the id-seeded shuffle) - run it with
  `--apply` after adding multiple-choice tests, and re-run benchmarks, since rewriting a
  prompt changes its `_compute_test_hash`.

### Grader Versioning
- A functional grader can change while its prompt does not, leaving stored results scored
  by rules that no longer exist. `web/app.py` records this so `outdated_only` re-runs the
  affected tests:
  - `FUNCTIONAL_GRADER_VERSIONS` - per-test-id version; bump a test's entry whenever its
    grading changes materially.
  - `OBJECTIVE_GRADER_VERSION` - applies to every test with an `expected` key.
  - `LLMModelBenchmark.GRADER_DIRECTIVE_VERSION` - the appended code/UI grading notice.
- Logic-puzzle graders check the **conclusion** now, not the vocabulary: `logic_modus`
  needs p false (it is modus *tollens*), `logic_knights` needs A=knight and B=knave,
  `logic_river` needs the goat brought back, `logic_weigh` needs the 3/3/2 split. The old
  checks passed a restatement of the prompt and failed correct answers phrased unusually.
  `_assigns_roles()` matches case-sensitively so the subject "A" is not the article "a",
  and `_CLAUSE_GAP` keeps the wrong-role guard from reading across "and".

### llama.cpp Preset Keys (models.ini)
- `llama-server --models-preset` **rejects keys that are not its own arguments and the
  container then crash-loops.** `web/app.py` holds `LLAMA_PRESET_KEYS`; `split_preset_settings()`
  sends only those to `models.ini` and everything else to the `<section>.profile.json`
  overlay, which the proxy and `llm_benchmark_suite.py` read alongside the ini. Saving a
  profile also sweeps a stale harness key out of the section.
- `thinking` is the key that motivated this: it is a harness toggle, not a llama.cpp flag,
  and the profile editor used to write it straight into `models.ini`.
- Preset keys are long options without the dashes (`ctx-size`, `n-gpu-layers`); short names
  and `LLAMA_ARG_*` env names also work. `[*]` is the defaults section.
- Flags worth knowing, exposed in the Model Profiles editor: `cache-reuse` (min chunk
  reused from the prompt cache via KV shifting, 0 = off), `batch-size` / `ubatch-size`
  (llama.cpp defaults 2048 / 512), `swa-full`, `reasoning-effort`, `split-mode`,
  `spec-draft-n-min`, `n-cpu-moe` (**MoE layers kept in RAM - not a thread count**),
  `moe-cache-mib` (a VRAM cache for the hot experts `n-cpu-moe` keeps in RAM; 0 = off) and
  `moe-cache-cpu-share` (0-1, share of cache misses the CPU computes instead of uploading;
  **new in b12233**, so a preset carrying it crash-loops an older image).
- **`mmap` / `mlock` are not preset keys.** llama.cpp replaced `--mmap`/`--no-mmap`/`--mlock`
  with `--load-mode` (`auto|mmap|mlock|mmap+mlock`); saving a profile sweeps them out of the
  section and reindex maps an old `mlock=false` to `load-mode = mmap`.
- **Checking a new llama.cpp build for flag changes:** fetch `common/arg.cpp` at both tags and
  diff the `"--flag"` sets; also check every `LLAMA_PRESET_KEYS` entry still exists upstream
  (that is how the dead `mmap`/`mlock` were found). The image is `server-cuda12`, so
  `sudo docker compose build --pull llama-server` is what picks up a new build.
- **`settings_scan.py` KV estimate reads the attention layout** (`kv_cache_mb`): qwen35/qwen35moe
  GGUFs carry `full_attention_interval = 4` (only one layer in four keeps a KV cache), and
  sliding-window models (`muse-glimmer`: `sliding_window = 2048`, pattern 4) cap three layers in four
  at the window. Counting every layer at full context charged qwen3.8-27b ~9.3 GB of KV at 64K q8
  instead of ~2.3 GB: the scan chose 0 GPU layers, the guard blocked even that, and the 32K retry ran
  almost entirely on the CPU until the thermal stop.
- **`settings_scan.py` and MoE:** the guard and hill estimates subtract the experts of the
  `n-cpu-moe` layers that are on the GPU (`gpu_weights_mb`; llama.cpp offloads the *last* ngl
  layers, `n-cpu-moe` covers the *first* N), so ngl=99 + n-cpu-moe is no longer blocked as
  "the whole file in VRAM". A working config that offloads experts gets a `moe-cache-mib` probe
  sized from its **measured** peak VRAM (80% of the headroom, 256 MiB steps), with and without
  `moe-cache-cpu-share 0.5`. The model's pre-scan config is always probed too
  (`incumbent_candidate`, same rung only), so a scan cannot replace a tuned profile with a
  slower one, and the winner is written over a pre-hill snapshot so a losing probe's keys
  never leak into the saved profile.
- **Settings-scan results are in the dashboard:** Model Profiles → "⚙️ Settings Scan Results"
  (`GET /api/settings-scans`) reads every `data/llm_benchmarks/settings_scan*.json`, oldest first, so
  a newer scan's winner replaces an older one, and lists each model's winner with every config it
  tried. Probe responses (the generated code) stay on disk. Use a distinct `--out` per scan run:
  the file name is how the card tells runs apart.
  Also available in presets but not in the editor: `cache-ram`, `ctx-checkpoints`,
  `checkpoint-min-step`, `kv-unified-per-slot`, `cache-idle-slots`, `context-shift`,
  `chat-template-kwargs` (e.g. `{"reasoning_effort": "high"}`).
- `benchmark-configs.py` sweeps ctx x cache x flash-attn, then sweeps `batch-size` /
  `ubatch-size` at the winner. Batch sizes only change **prefill**, so that stage times a
  ~2k-token prompt with one token of output - the first stage's one-line prompt cannot
  tell the settings apart. The chosen config is applied and the backend restarted *before*
  the quality suite runs (it previously scored the restored original config).

### Image Studio / stable-diffusion.cpp
- sd-server's OpenAI-compatible routes read only `prompt`, `n`, `size`, `output_format`
  and `output_compression`. **Every quality control - steps, CFG, seed, sampler,
  scheduler, negative prompt, denoise strength, diffusion cache, highres fix - reaches the
  sampler only inside an `<sd_cpp_extra_args>` JSON block embedded in the prompt**, which
  the server parses and strips before generating. Sent as plain fields they are silently
  ignored: the request succeeds and the settings do nothing.
- The proxy translates the OpenAI-shaped fields into that block:
  `extract_sd_native_params()` maps them to the native schema (`sample_params.sample_steps`,
  `sample_params.guidance.txt_cfg`, `hires.*`, ...) and `apply_sd_native_params()` merges
  them into the prompt. A block the caller already embedded wins on conflicts.
  `_flatten_sd_native_params()` is the fallback for an sd-server too old to parse the block,
  chosen by probing `GET /sdcpp/v1/capabilities` (`get_sd_capabilities()`, cached 60s).
- `GET /v1/images/capabilities` (proxy) / `/api/sd/capabilities` (web) expose the engine's
  samplers, schedulers, upscalers, LoRAs and defaults. The Image Studio's advanced drawer
  is filled from it, so it only offers options the running build has.
- The studio rolls a concrete seed when the field is `-1` (the engine never reports which
  random seed it used), shows it on each result card with **♻ Reuse**, keeps previous
  renders in the gallery instead of wiping it, and offers **🎨 Refine** to send a result
  straight back into the Photo Editor.

### Podcast Studio (`#view-podcast` tab, `web/podcast_mixer.py` + 4 `/api/podcast/*` routes)
**Thesis: stop asking a song generator to do a bed's job.** A podcast bed is a narrow job
(sustained harmony, gentle pulse, nothing competing with speech) = textbook procedural synthesis.
MusicGen is a *song* generator: a 30 s training window, 32 kHz output, CC-BY-NC, sharing the 8 GB
card with llama-server and sd-server. Ten minutes of bed is the one thing it does worst.

1. **Bed = PROCEDURAL SYNTHESIS** at **24000 Hz to match Kokoro**, so nothing resamples. Five
   presets; three detuned partials per chord tone (±4 cents = shimmer, not wobble), an optional
   pitch-swept pulse, a one-pole lowpass, a 0.07 Hz tremolo, deterministic in `seed`.
2. **Theme sting = MusicGen at native ≤30 s** (where a generative model earns its keep). Only the
   ~10 s tail needs a 32k→24k linear resample.
3. **The encode must not peak-normalise.** `audio_server._wav_bytes` does, which would re-boost
   the quiet bed over the speech - which is why the mixer has its own encoder.

- Routes: `GET /api/podcast/status`, `GET /api/podcast/voices`, `POST /api/podcast/draft`
  (the proxy writes the script), `POST /api/podcast/render` (the mixer).
- **The mixer lives in `web/`, not `audio_server.py`**, because `web/` is bind-mounted read-only
  (an idle-safe restart is enough) whereas `audio_server.py` is COPYed into a 2 GB pytorch image.
  `Dockerfile.web` has no ffmpeg and does not need one: decode/loop/crossfade/duck/resample/sum
  are all numpy-vectorizable, and `wave` is stdlib.
- `mix_podcast` warns on a **cross-gender clone** - OpenVoice transfers timbre badly across
  gender, so the roster carries `source_gender` and the panel labels such an option rather than
  silently offering it.
- **The MusicGen 1500-token clamp is silent.** `max_new_tokens = min(duration_s*50, 1500)`, so a
  180 s request returns 200 with 30 s of audio. The only tell is
  `meta.duration_s < meta.requested_duration_s`; the render route warns when it sees that.

### Base-voice pairing (`voice_clone.py` → `POST /api/tts`, `PATCH /api/voices/{pid}`)
- **The base voice is a pitch decision, and every wrong one is audible.** `correct_pitch` runs a
  phase vocoder over the *whole* render to slide it onto the profile's `median_f0_hz`. A base
  voice 2.6 semitones off means the entire narration is sung through a vocoder, which is a worse
  artefact than the timbre the vocoder was there to fix. So with a clone in play and no
  `voice` in the request, the server picks the Kokoro voice that needs the least correction
  (`pair_base_voice`), not a hardcoded default that happens to be wrong for most speakers.
- **"Free" is a hard band, not a ranking preference.** `PAIRING_MAX_SEMITONES = 0.5` - the same
  quarter-tone `correct_pitch` refuses to act on. Any voice inside it outranks a closer one
  outside it, because inside means `correct_pitch` does not run at all. Ties break on
  |semitones|, and the sort is total so registry order can never change the answer.
- **Inside the free band, similarity decides.** A 15 s probe and a real narration disagree by
  more than the gaps between free candidates: on the enrolled profile `af_bella` probed at
  -0.03 st and `am_liam` at +0.16, but on narration `af_bella` needed -0.9 st of vocoder and
  `am_liam` none, and `am_liam` scored 0.930 similarity to `af_bella`'s 0.910. So the probe also
  records `clone_similarity` of its converted audio (cached with the F0; a cache without it is
  re-measured once), free voices rank by it, and pitch only orders voices that are not free or
  could not be compared. A voice outside the band never wins on similarity.
- **Delivery counts, not just timbre.** The converter keeps the base voice's pace and intonation.
  On the enrolled profile (3.93 words/s, 11.9 st range) `af_nicole` had the best timbre score
  (0.933) but read at 3.36 words/s with 4.8 st, slow and flat. So probes also store the base
  voice's `prosody` (words/s over speech only, pYIN 10-90% range in semitones), the speaker's is
  measured from their takes (`speaker_prosody`, cached as `prosody.json` in the profile), and free
  candidates rank on `delivery_mismatch`: timbre, pace and intonation, each in a just-noticeable
  unit, summed.
- **`clone_highband_hz` is a cleanliness/likeness trade, measured.** OpenVoice's decoder smears
  harmonics above ~2 kHz and leaves steady tones there (heard as a metallic "digital echo"; the
  20-500 ms `echo_r` check does not see it). `band_blend` keeps the converted audio below the
  cutoff and the clean Kokoro source above it (complementary linear-phase FIRs, source aligned by
  cross-correlation). It removes the artefact, but similarity to the speaker fell from 0.93 to
  0.64 (2.5 kHz) / 0.70 (3.5 kHz) against 0.49 for plain Kokoro: much of what the reference
  encoder hears as the speaker is in the band the decoder smears. Off by default.
- **The WavMark watermark is not the distortion.** Applied alone to clean speech its residual is
  53 dB below the signal.
- **`checks: true` on `/api/tts` measures the result** (`narration_checks`): `echo_r` is the peak
  speech autocorrelation at 20-500 ms lags, and `steadiness_min` / `largest_shift_at_s` is the
  lowest timbre cosine between neighbouring 3 s windows, which is where a voice change shows.
- **`clone_unit: "paragraph"` converts a paragraph in one pass.** The default (`"chunk"`)
  converts each synthesized piece separately, which lets timbre move at every seam. Paragraph
  mode joins the paragraph's Kokoro speech (with its sentence pauses) and converts it once.
- **Gender is not a filter.** `pair_base_voice` measures every voice in the 28-voice registry,
  af_* and am_* together, and prefers the one that lands nearest the speaker *after* conversion.
  A wrong guess about gender is no reason to exclude a voice that lands where the speaker is.
- **The ranking is on converted pitch, which is not the same number as base pitch.**
  `converted_voice_f0` renders a short `_SOURCE_SCRIPT` in the candidate, runs it through
  `convert` at the request's `tau`, and measures the result - because the converter does not
  preserve the base voice's register, and by how much depends on the voice in a way that is
  **not monotonic in that voice's own pitch**. Measured on a real profile, `am_echo` speaks
  9.1 Hz *above* its own 106.9 Hz base while `am_liam` speaks 20.6 Hz *below* its 125.5 Hz one,
  and `af_nicole` 49.5 Hz below its 151.2. Ranking on base pitch therefore picks the voice that
  then has to be vocoded by two semitones - worse than the hardcoded default the pairing was
  written to replace. This is the same audio `correct_pitch` measures afterwards, so the
  measurement is the one the correction is actually applied to.
- **The measurement is cached per voice *and* per profile, so the first pairing is slow.**
  `converted_voice_f0` stores `{"base_f0_hz", "converted_f0_hz"}` in
  `VOICES_DIR/_sources/{key}~{pid}.converted.f0` (`key` is the `source_se` scheme; `~pid` because
  the same voice converts differently for every speaker). Every pairing after the first is a file
  read. An unreadable cache is re-measured, an unwritable one is a warning, and neither is fatal.
  `PAIRING_PROBE_S = 15.0` bounds the probe, because a median F0 over fifteen seconds is as stable
  as one over thirty and pairing converts *every* candidate once.
- **Three overrides, in this order:** an explicit `voice` in the request wins outright (blends
  like `"af_heart,am_adam"` are still allowed and skip pairing); a profile's `pinned_base_voice`
  outranks the guess; nothing else does. Omitting `voice` is *not* a request for the default
  when a clone is in play, so an explicitly empty `voice` still 400s while the absent key does not.
- **The response says which was used.** `meta.clone.base_voice` is the voice actually rendered,
  `base_voice_source` is `"requested"` or `"paired"`, and `base_voice_pairing` is the full report
  with every candidate's `semitones_from_target` plus `base_f0_hz`, `converted_f0_hz` and
  `converter_offset_semitones`. A pairing that cannot be decided returns
  `base_voice: null` with a `reason` and the request falls back to `DEFAULT_BASE_VOICE`
  (`af_heart`) - the fallback is visible, never a surprise.
- **Pairing refuses to guess without a target.** No `median_f0_hz` on the profile means no
  ranking is possible, so `pair_base_voice` returns `base_voice: None` rather than a default.
- **`update_profile` validates every field before writing any of them.** Renaming a profile and
  pinning a base voice are one atomic `meta.json` swap (tmp + `os.replace`); half of one leaves
  a profile whose name and base voice disagree about when they were set, which makes a rollback
  ambiguous. A blend is refused as a pin: it has no pitch of its own.
- **Ranking by pitch deliberately disagrees with `identify`.** `/api/voices/identify` scores
  post-hoc timbre similarity and has better scores for worse-sounding voices; the pairing is
  chosen on the cost of fixing the voice instead.

### The converter's latent draw (`voice_clone.convert` → `POST /api/tts` `clone_seed`)
- **OpenVoice's `PosteriorEncoder.forward` samples its latent, not its mean:**
  `z = m + torch.randn_like(m) * tau * exp(logs)`. `voice_conversion` is called **once per
  sentence** (`/api/tts` loops sentences → `convert`), so with torch's global generator unseeded
  every sentence of one narration is re-timbred from a **different draw of the same
  distribution**. That is the sound of a narrator who cannot hold a voice.
- **`convert` forks the generator and seeds it**, so all sentences share one draw. The fork is
  what makes it safe: `/api/music` (MusicGen) draws from the same global generator in the same
  process, so seeding without forking would make this endpoint's internals other people's output.
  `seed=None` forks nothing and draws from the live generator - OpenVoice's own behaviour, kept as
  the escape hatch.
- **The report says whether it was pinned:** `meta.clone.seed` and `meta.clone.seeded`. A caller
  who hears instability needs to know whether the endpoint did it or they did.
- **A non-integral `clone_seed` is rejected, not truncated.** `int(1.5)` is 1: a typo answered
  with a confident wrong answer is worse than no answer.
- **How much this was worth, measured** (`seed_stability.py`, 4 renders of one 7-sentence
  paragraph, log-mel timbre vectors): mean cosine between sentences of a render **0.953**, the
  same sentence re-rendered **0.982**, across/within per-band sd ratio **0.43**. So the noise is
  real and reproducible, but it is roughly **10% of the within-render timbre spread**, not all of
  it. Fixing it makes narration deterministic and a little steadier. It is not a cure for "this
  clone sounds bad", and should not be sold as one.
- **Measured alongside, so the next person does not re-chase it** (librosa.yin, `pitch_diag.py`):
  a clone of this profile lands within **0.1 semitone** of the enrolled `median_f0_hz`, and a
  waveform autocorrelation over 20-500 ms finds **no reflection** (best lag r = 0.08-0.12, well
  under the band's own p99, and no better than plain Kokoro with no clone at all). "It sounds
  like there is an echo" is therefore not an echo in the narration: OpenVoice's decoder is a GAN
  vocoder and is characteristically a little ringy, and anything else is the music bed.

### Speaker identification (`voice_clone.py` → `POST /api/voices/identify`, `/calibrate`)
- Built on the artifact the system **already has**: cosine similarity of OpenVoice's 256-d
  reference encoder against each profile's centroid. Zero new models, zero new VRAM.
- **The threshold is derived, not chosen.** `identity_threshold(spreads) =
  clamp(1.5 * max(spread), 0.05, 0.95)` - `max` over the **worst** profile, not the average, so a
  consistent majority cannot talk the floor down and then reject their own re-records.
- `create_profile` stores `takes.pt` alongside `se.pt` and records `intraspeaker_spread` (pairwise
  over takes, or **within the take's windows** when there is only one). Reporting 0.0 there would
  claim a certainty we do not have.
- `identify` scores the **best single window**, not the mean (half a minute contains breaths and
  doors), and returns the ranking **even when nothing matches** - an agent needs to say
  "closest is X". 404 means "no voices enrolled", 422 means "not enough speech".
- **The `openvoice` package ships no `verifier/` subpackage**, so there is no WavLM verifier
  available. `identify` is the seam where one would drop in: it takes audio and returns a
  ranking, and knows nothing about how the scores are made.

### Raven / SharedLLM can drive all of it
`../SharedLLM` registers `sharedllm_podcast_render`, `sharedllm_speaker_identify` and
`sharedllm_list_voices` (`tool_registry.py`), routed to `EXECUTION_SVC`
(`/execute/podcast_render|identify|list_voices`) by `services/execution/handlers/podcast.py`,
which calls the dashboard at `ALPACA_WEB_URL` (rendering - the mixer lives there) and the audio
server at `ALPACA_AUDIO_URL` (identification - OpenVoice's reference encoder only exists in the
container that has torch). `tests/test_sharedllm_contract.py` pins alpaca's copy of the tool
vocabulary against SharedLLM's in **both** directions, so a tool added on either side without the
other fails the build.

### Vision benchmarks (`vision` group, `scripts/gen_vision_fixtures.py`)
- Eight drawn images with known answers (receipt total, device serial, table cell, which switch is
  on, a settings screen's battery level, a bar chart against gridlines, counting red circles among
  red squares, a grid cell). The generator draws them deterministically and writes them into
  `benchmark_tests.json` as `data_base64` attachments (`data/` is gitignored); edit the generator and
  re-run it with `--apply`, never the JSON. `test_the_shipped_vision_category_is_what_the_generator_draws`
  keeps the two in step. Answers are numeric or exact strings - a letter key would join the
  multiple-choice balance, and a word can pass by accident when a model lists every option.
- The runner sends image attachments as the proxy's Ollama `images` field. **A model without a
  projector is skipped, not failed**: llama-server rejects the image ("image input is not supported -
  ... mmproj"), the result carries `skipped: true`, it is left out of every score, and a group whose
  tests were all skipped emits no `category_` block at all. The direct path and online providers
  skip image tests too (no image transport there). `_get_test_benchmark_stats` ignores skipped runs.
- Dashboard: `👁 Run Vision` and `⚡ Quick Benchmark` run one group on the SELECTED models (unlike
  Run Frontier All, which selects every model: a vision run on a text model is a wasted model swap).
  Vision tests are `kind: "image"`, so the Test Browser shows each image; results show `Skipped`.
- The old `mm_image_light_switch` referred to an "attached" image it never had; it was removed and
  the switch panel is now a real vision test.

### Creative benchmark tests (all in `benchmark_tests.json`, all scored)
| Test | Category | Type | What the grader actually checks |
|---|---|---|---|
| `creative_svg_theme_pack` | creative | functional | A **seamless tile** (square viewBox with `width == height`, `fill="none"`, stroke widths 1-1.5, opacities 0.06-0.22, colours as `{{accent}}` placeholders, no glyphs) **and** a 6-`<symbol>` sprite sheet. `scripts/install_theme_pack.py` turns a passing answer into a Jarvis theme pack. |
| `creative_illuminated_manuscript_john1` | creative | ui | All 18 KJV verses present, a page-named container with **its own** border, a real inline `<svg>`, and a working page flip. See `manuscript_rubric.py`. |
| `composite_sprite_bgm_game` | gamedev_alt | composite | The 4-stage tool path: LLM authors → sd-server renders the sprite → audio-server renders the loop → `grade_code(ui=True)`. **The assembled HTML is written to `data/artifacts/`,** which is what makes it playable in the arcade. |
- **Why SVG and not diffusion:** every SD flyer preset's negative prompt bans "garbled text,
  distorted letters, bad typography", and the repo ships **zero font files**, so any `<text>` in a
  generated asset falls back to whatever the render host happens to have. Deterministic SVG is
  machine-checkable.
- **Diffusion is never asked to draw text.** The same reasoning applies to the manuscript.

### Family Games (`arcade/family.py`, `arcade/family_games/`)
- Hand-built, touch-first games for the Jarvis family hub, all on one shared kit
  (`family_games/_kit/kit.js` + `kit.css`: HUD, title/pause/game-over cards, DPR canvas,
  pointer input, WebAudio sfx, particles, the score bridge). A game's `game.html` carries a
  `<!-- @family-kit -->` marker; `family.build_html()` inlines the kit, so the served page
  has no network dependency.
- At arcade startup `family.sync()` installs/updates each one as `data/arcade/games/family-<name>/`.
  It replaces only `game.html` and the descriptive meta; **scores, ratings, bests and plays are
  never touched**, so a new version keeps its scoreboard. The arcade image COPYs `arcade/`, so
  a game change needs `sudo docker compose up -d --build alpaca-arcade`.
- Lifecycle in `data/arcade/family.json`: `testing` (new games; playable, votable, not on the
  board) → `published` ⇄ `unpublished`. `POST /api/admin/family/<slug>/state` needs
  `ARCADE_ADMIN_TOKEN` (503 naming the variable when unset). `GET /api/family/games` lists all
  with their state; **Jarvis** decides what each viewer sees (players: every published game,
  best rated first; the benchmark shelf shows a game only once it is published to the family).
- **Mobile games only.** `arcade/mobile.py` `contract_report()` is the one mobile contract (viewport,
  fills any screen, no fixed canvas, DPR, touch input, no page gestures, safe areas, pause when hidden,
  rAF loop, personal best, score bridge, no name prompt). A built-in game that fails it is not
  installed; a `mobilegames` benchmark game that fails it stays an ordinary arcade game with
  `mobile_contract_failures` in its meta; publishing re-checks and refuses (400) a failing page;
  `is_family()` requires `mobile_compatible: true`. The benchmark grader uses the same function.
- Boards are per collection: `?collection=family` on `/api/highscores`, `/api/players*`
  counts only published Family Games; the default `arcade` board excludes them.
- Ratings count each named voter once (latest vote wins), so the rating bar cannot be
  stuffed by one member.
- Score bridge: the game posts `{source:"arcade-game", type:"score", score, run_id}`; the play
  page (`ARCADE_BRIDGE`) turns it into a capture `slug:run_id`, and relays the parent's
  `{source:"arcade-host", type:"saved", rank}` back so the game card shows the rank.
- Phone check: `PLAYWRIGHT_MODULE=<node_modules>/playwright node scripts/family_games_check.mjs <name>`
  plays a game at iPhone 13 / SE / Pixel 7 / iPad sizes and fails on console errors, a missing
  bridge message or page overflow.

### Checkers (`arcade/checkers.py`, `arcade/family_games/checkers/`, `/api/checkers/*`)
- **The one game two people play against each other**, so it does not fit the party engine's phase
  machine; it is its own module with its own in-memory room store (4-char codes, two seats, a
  per-room secret, TTL/idle pruning) modelled on the party rooms. It ships as the built-in family
  game `family-checkers`; no SharedLLM change is needed - it appears on the shelf once published.
- **The server owns the board.** `arcade/checkers.py` is the single source of truth: English
  draughts rules (diagonal-forward men, captures compulsory and chained, a crowning jump ends the
  move, kings move both ways, no-piece/no-move loss, a 40-move and a threefold-repetition draw),
  a small iterative-deepening alpha-beta opponent, and the rooms. The page is a thin client that
  draws a snapshot and posts the path of a move; it can never fabricate one.
- **Routes** (`arcade/app.py`, after the family admin routes): `GET|POST /api/checkers/rooms`
  (list/create; `mode` `friend`|`quick`|`ai`, or `code` to join), `GET /api/checkers/rooms/<code>`,
  `GET .../join.svg` (the invite QR), and `POST .../{move,ai,reset,leave}`. Errors are 404 (no room),
  403 (not a player), 400 (illegal move / not your turn). Adding the computer mid-lobby (`.../ai`)
  seats `Alpaca AI` in the free seat; the person is always Red and moves first.
- **It reuses Parable Party rather than reimplementing it.** `checkers.py` imports `party` (by name,
  the same dual-mode seam `app.py` has) to share the friendly room words (`party.ROOM_WORDS`, so codes
  read `NOAH`/`LION` instead of random letters) and the one-way public id (`party.public_id`). The
  room payload also carries `invite_url` and `qr_url`, built from the party section's
  `_public_base_url()` and the party QR pattern (`GET .../join.svg`, segno, `no-store`), so a friend
  arrives by link or by scanning the lobby QR. `party.py` itself is untouched.
- **Player ids are secrets.** A snapshot carries `public_id` only (sha256 of the room secret), never
  the id; and `legal` (the moves on your turn) is filled **only** for the seat about to move, so a
  spectator or the other seat cannot read your options.
- **The computer answers inside the same request.** After a human move, `submit_move` runs
  `_maybe_ai_unlocked`, so a poll never waits on a background thread and the turn always returns to
  the person. `choose_move` is bounded by `AI_TIME_BUDGET` (1.6 s) with depth up to 6.
- The page is a single kit-built HTML (`arcade/family_games/checkers/game.html`) with a custom lobby
  overlay (friend / quick match / computer / join-by-code, the server's `invite_url` + a scannable
  `qr_url`), a canvas board with slide + capture-burst animation, and a score bridge on the result
  card (`captures × 50 + 300 win / 120 draw`). It passes `mobile.contract_report` by construction.
- Tests: `tests/test_checkers.py` (rules, AI, rooms, the HTTP surface, the Parable Party reuse, and
  that the shipped page passes the mobile contract).

### Party games on the TV (`arcade/party.py`, `/tv`, SharedLLM `tv_stage`)
- **The TV runs the game's stage page; it is never sent audio or a picture of it.** `/tv/<CODE>`
  (`arcade/templates/party_tv.html`) is a hands-free big-screen view: it polls the room as the stage
  (`?stage=1`), shows the join QR (`/api/party/rooms/<code>/join.svg` → `/join/<code>`), and plays the
  Alpaca TTS announcer on the TV's speakers. Bare `/tv` follows whichever room is active. It is ES5 + XHR
  + flexbox on purpose: TV browsers are years behind phones.
- **Phones stay quiet while a stage is live.** The room reports `stage_connected` (a stage polled in the
  last `STAGE_PRESENCE_SECONDS`) and the phone game skips its own announcer, so there is no echo.
- **No device is ever named in this repo.** Discovery and display belong to SharedLLM's execution service
  (`services/execution/handlers/tv_stage.py`), which is on the host network with Home Assistant,
  pychromecast and the LG/Samsung clients; the arcade's bridge network cannot do mDNS/SSDP. The arcade
  relays `GET /api/party/tvs` and `POST /api/party/cast` through the **Jarvis gateway** (`JARVIS_GATEWAY_URL`,
  the same `/api/arcade/{tvs,cast}` routes the Jarvis app uses) or, on the same host, straight to execution
  (`EXECUTION_SVC_URL`). Both need `JARVIS_INTERNAL_SECRET` (SharedLLM's `INTERNAL_SECRET`, acting as the
  household). Execution's :8003 is not exposed to the LAN, which is why the gateway is preferred.
  Unconfigured, discovery says so instead of guessing.
- **A caller picks the screen and the room, never the URL.** The cast route always sends this arcade's own
  `/tv/<CODE>` at the address the phone used (`X-Forwarded-*` aware, `ARCADE_PUBLIC_URL` overrides).
- How each screen shows it (SharedLLM): Google Cast (Chromecast, Google/Android TV, Cast built-in) → the
  DashCast receiver navigated to the page; LG → its browser via `webostv.command`; Samsung → its browser via
  `samsungtvws` (the TV asks once to allow "Jarvis"); Android TV without Cast → an app link via
  `androidtv_remote`; **Roku has no web engine**, so the page is rendered in headless Chromium and sent as a
  live HLS stream (`stage_stream.py`) to Media Assistant. Every result says which method worked and whether
  the TV confirmed it.
- **A streamed stage is seconds behind, and only a streamed stage waits.** The stage page knows it is being
  rendered into video (headless Chromium) and polls with `&video=1`; while such a stage is live the room's
  `stage_delay_s` is `PARTY_STREAM_DELAY_S` (default 8) or the host's own tuning
  (`POST /api/party/rooms/<code>/stream_delay`). Each phase gets that much extra time, phones show "Get ready"
  until `phase_age_s` passes it, the stage's clock runs that far ahead, and the host's phone speaks and plays
  the music (a stream carries no sound). Rooms on a live-page TV have no delay at all.
- **Game-show music is composed once by MusicGen** (`arcade/party_music.py`): a theme sting, a winner fanfare and
  loopable beds for the lobby, writing, voting and reveal, each ≤30 s, seeded, crossfaded into a seamless loop
  and stored in `ARCADE_DIR/party_music/`. Missing cues are composed in a background thread at startup (a cue
  that is not ready is just silent); `POST /api/admin/party/music/regenerate` redoes them. It plays wherever the
  announcer plays, ducks under the announcer, and the host can turn it off per room.
- **Two voices, one pipeline** (`arcade/announcer.py`, `party.synthesize_show_line`). The announcer is the
  `bm_fable,bm_lewis,bm_george` blend at 1.08x (1.15 was too fast in play) with per-sentence pauses, chosen by ear over three audition
  rounds; its chain is deliberately clean (high-pass, presence, one compressor, no echo or drive), and everything but
  an exclamation sits `SHOUT_LIFT_DB` under it so the shouts land. `depop()` fades out the ~2 ms burst Kokoro puts at
  the start of a clip that opens on a vowel ("Ooh."), which was heard as a click after a pause. A line that opens with
  "Extra! Extra!" is split: a paper boy shouts "Extra! Extra! Read all about it!" first. No Kokoro voice is a boy
  (every male voice measured 88-166 Hz), so he is `am_eric,af_kore` (~170 Hz) resampled up 1.32x to ~235 Hz,
  band-limited, driven and given a street slap echo. Stdlib only, ~0.2 s a line, cached with the clips.
- **243 writing prompts in 9 round types, plus 275 race clues and boards** (`PROMPT_BANK` + `arcade/party_prompts.py`, validated by
  `tests/test_party_prompts.py`, including that every Truth or Bluff truth trips `_matches_truth` and its bluffs do
  not). A match is an easy Blank-O-Matic opener, five different round types drawn per match (always one Truth or
  Bluff), and the double-points Survey Says finale. Prompts are chosen least-recently-used across all rooms.
- **AI players only at a small table, and the trap is always there.** Scribes sit in only while 2 or fewer people play
  (to make three seats). With no AI players at all, the house plays a second Scribe's Trap drawn from the prompt's
  `bot_answers` (`opt_house_extra`, `_house_extra_answer`); both cost 250 x multiplier, like the first.
- **Pronunciation is measured, not guessed.** `announcer.PRONUNCIATIONS` holds misaki IPA (misaki's own symbols:
  `O A I ʤ`, never `oʊ eɪ aɪ dʒ`) for every Bible word misaki says wrong, checked in the audio-server container with
  `KPipeline(lang_code="a", model=False).g2p`; `SAID_RIGHT` lists the ones it already gets right. Possessives go
  inside the mark, "Job" is the man unless the words around it make it a task, and `unshout()` lower-cases words in
  capitals (headlines, "WE ASKED 100 PARENTS"), which misaki would otherwise spell letter by letter.
- **A changed music recipe recomposes itself.** Each cue's file has a `.json` sidecar with the prompt, seed and length
  it was made from; `party_music.is_current()` treats a mismatch like a missing file.
- **Race rounds** (`arcade/party_race.py`, content in `arcade/party_race_content.py`): Quick Draw (5 clues, 4 choices),
  Lightning (8 true/false), Pair Up (match 5 pairs) and In Order (sort 5). Everyone answers at once; the first correct
  answer scores most (1000/750/600...), a wrong tap in Quick Draw or Lightning costs points, and boards score per right
  part plus a speed bonus for a perfect board. The phase is `racing`; clues advance inside it (`step` in the snapshot
  changes per clue and per answer flash, so screens redraw and the announcer speaks), and a streamed TV's lag applies
  per clue (`phase_age_s` is per clue). Answers go to `POST /api/party/rooms/<code>/race` `{player_id, item, choice}`.
  Every match has one race round, sometimes two, in place of a writing round.
- **Reveal, then scoreboard.** The answers screen shows only the answers; at `scores_at_s` (5 s after the result is
  called, 8 s into a race reveal) the TV and phones switch to a full-screen scoreboard with "Up next", for about 8 s.
  The music sits well under the announcer: bed volumes 0.15-0.2 and ducked to 22% while a line plays.
- **The announcer's chain must never add a click.** `compress()` smooths its gain and looks ahead 5 ms (it used to
  clamp down inside a hard vowel onset after a pause, which was heard as a click), fades are S-curves, and
  `test_the_chain_adds_no_clicks` holds it. The remaining sharp edges in a line are Kokoro's own plosives.
- **Lines are timed, jokes fitted.** Every line the announcer says is measured from its audio (`_spoken_seconds`);
  a reveal's result-and-joke line is made in the background as the reveal starts, and the reveal is held exactly that
  long (the clip is then cached, so the TV speaks at once). Jokes are chosen at random, least recently told in any
  room first and never twice in a room, and a writing-phase joke is only one that ends before the ten-second call.
- **Emoji on TVs without an emoji font** (a 2020 LG drew boxes): the stage measures a smiley on a canvas and, if it
  has no colour, swaps every emoji for its Twemoji image in `arcade/static/twemoji/` (CC-BY 4.0; `?emoji=img` forces it).
- **The host never repeats himself in a room.** `_pick()` chooses each moment's line once, never one the room has
  heard until it has heard the whole list, least recently said anywhere first. Lists are the `HOST_*` pools plus
  `arcade/party_host_lines.py` (523 more: openers, round intros per kind, standings talk, race quips...).
  `standings_line()` comments on the scoreboard after a result (runaway, close, tied, comeback, last round).
  **No fixed sentence in a result.** The voted-round winner (`HOST_ROUND_WINNER`, with `{who}`/`{take}`/`{has}`/`{was}`),
  the truth found, the room trapped, the race winner/perfect/nobody and the lead-in to the between-rounds joke
  (`HOST_JOKE_LEADIN`, `HOST_THINK_LEADIN`), each race clue's answer (`HOST_FLASH_FIRST`/`_NOBODY`, salted per clue) and
  the race intro (`RACE_INTROS` + `ROUND_INTROS[kind]`) all go through `_pick()`; "A triumph of taste. Or at least, of
  votes." once closed every round and "got it first. Naturally." every clue. Lines in a pool should not share a
  sentence either (two "Ten seconds left." openings sound like a repeat): simulate a few matches and count sentences.
- **A race clue is closed while it is read.** `open_after()` holds answers until the TV has said the clue (its
  measured length, capped at 12 s); phones grey their buttons ("Listen to the question...") and every clue's line
  is made as the race starts so the TV speaks it at once.
- **The TV stage is a 16:9 board that fits any screen.** Every size in `party_tv.html` is in `rem`, and the page sets
  the root to a tenth of the largest 16:9 frame the screen holds (`sizeFrame`), so a 4:3 or portrait screen letterboxes
  instead of reflowing; `body` gets its own small size because the root is a layout unit, not a text size. A screen with
  more on it than fits is scaled down inside `.fitbox` (`fitMain`), never clipped. Write new sizes in `rem`
  (`1rem` = `10vh` on a 16:9 screen), never `vh`/`vw`, and no `inset` (Chrome < 87 on webOS ignores it).
- **Visual check:** a stand-in arcade on :5099 can stage every screen and photograph it at TV and phone sizes
  (see the session tooling pattern in `scripts/family_games_check.mjs`); long names, a long unbroken word and seven
  options are the cases that used to clip. Phones give names two lines and keep a round's change under the score.
- **Duplicate answers share one option** (`_build_voting_options_unlocked` merges by `_normalize_compare`; typing the
  trap's words counts as playing it). A player who leaves mid-race takes their taps with them, and a race's answer
  flash is held until its "X got it first" line has been said (`_hold_flash_for_the_line_unlocked`).
- **Every screen goes everywhere at once** (`_gate_unlocked`). A new screen (`step_key`: match:round:phase[:step])
  starts closed: every client gets `waiting` and no content (prompt, ballot, reveal, clue are withheld; answers,
  votes and race taps are refused). Each device reports it is on it (`?ready=<stepKey>` on its poll); a live TV
  reports it has the host's line in hand (`POST .../said {key, event: "ready"}`); with no TV the room makes the line
  itself first. When all are ready the screen goes `GO_LEAD_S` later (`go_in`), the TV starts its line then, and
  the phase clock starts from that moment (`_start_from_unlocked`). Clients poll at `poll_ms` (200 ms while
  waiting). With a TV, a race clue opens `OPEN_LEAD_S` after the TV reports `"end"` of reading it (every client is
  told `opens_in` in advance, so polling order gives nobody a head start) and an answer flash ends on its `"end"`.
  Nothing waits on a guessed time: a device or TV that stops polling (`PLAYER_AWAY_SECONDS`,
  `STAGE_PRESENCE_SECONDS`) is simply no longer waited for. A video stage keeps its own fixed delay and no gate.
  **While a screen waits, its round is paused** (`_check_auto_advance_unlocked` does nothing while `gated`): the
  first deploy let writing, voting and the reveal expire behind "Get ready". **Only devices that report are waited
  for** (`reports_ready`, set by any poll carrying `ready`, even empty): a page left open across a deploy never
  reported and froze the room, showing an empty answer box. Snapshots carry `build` (a hash of the screens' code);
  an open page that sees it change reloads itself. TV timers (reveal steps, chatter) also wait for the go.
  **Each round's opening line is made before the round starts** (`_prepare_round_unlocked`): during the previous
  reveal, and for round 1 while the room waits in the lobby (`prepared_schedule` and `prepared_match_id` are kept
  apart, since the opener is chosen per match and the match on screen still reads `schedule`). Without it the TV sat on
  "Get ready" ~15 s while the long opening (banter, story, prompt) was made. Race rounds premake their own lines.
  **The party game page is served `no-cache`** (`meta.party`), never `stale-while-revalidate`: a browser kept an old
  copy and needed a hard refresh. Single-player family games keep the fast cached launch.
  **Every screen counts down to the same server moment**: snapshots carry `server_time`, `deadline_at`, `go_at`
  and a race's `open_at`; each page keeps its offset from the poll with the shortest round trip (`noteClock`) and
  counts to `localAt(...)`. Counting from "seconds left" on arrival made a slow TV run a second or more behind. A race
  clue being read sends `listening` and its full `seconds`, held still on every screen until answers open.
  Gate events and TV reports are logged (`arcade.party`: "go ...", "tv ready/end ...") so a live game can be
  followed with `docker logs alpaca-arcade`.
  **Nothing waits on one message:** the stage snapshot says what the room still needs from the TV (`tv_needs`:
  "ready"/"end") and the TV re-sends it (and retries a failed POST), so a lost report cannot hold a screen or a
  locked round. Only a TV page that reports (`tv_reports`, set by `rtt` on its polls) sets the pace; an old page
  left open across a deploy is ignored. Writing/voting open on the TV's "end" of the question line (`open_at`,
  `listening`), race clues get their full `ITEM_SECONDS` from when they open, an answer flash stays at least
  `FLASH_MIN_S`, and the opening line is chosen on the word estimate so measuring its clip never changes it.
  Party tests count devices as ready via the `party_devices_ready` fixture; `party_gate` tests keep the gate.
  Measured in browsers: TV, phone and desktop showed a question within 70 ms of each other.
- **Taps belong to the screen they started on** (`freshTap` in the phone game): a finger that went down before the
  screen changed does not press the new button under it. The answer box is never rebuilt while focused, and Return
  closes the keyboard instead of sending (it sent half-written answers).
- **Only winning answers are signed at the reveal**; the rest stay anonymous (`is_mine` still tells a phone its own).
- **Every writing prompt's story is told first** (`party_story_context.STORY_CONTEXT`, by id, one or two sentences
  naming the chapter; `party_scripture.story_context` falls back to "This one comes from <book> chapter N." for an
  AI prompt). A Truth or Bluff story stops before the answer (tested with `_matches_truth`). The banter is dropped
  when the opening would pass `OPENING_LINE_MAX_S`, and the writing clock grants the reading time (up to
  `PROMPT_READ_MAX_S`). **The verse itself is shown, not read:** under the prompt while writing and voting (never in
  Truth or Bluff, where it is the answer), at the reveal, and under each race answer.
- **Ids on screens are public.** A player's own id authorises their actions, so snapshots carry
  `public_id(room, pid)` only and tell each phone which one is `you`. Host actions must name the host; `/next`
  carries the screen it was tapped on (`expect`) and is ignored if the room has already moved on.
- **Room snapshots never carry credentials.** An earlier version stored the HA URL + token on the room and
  serialized it to every phone; `test_room_snapshots_never_carry_tv_credentials` keeps it out.

### Error Types & Fixes
| Symptom | Cause | Fix |
|---------|-------|-----|
| `/slots` 400 Bad Request | Missing `?model=` param | Always pass `model` query param |
| `/slots` 400 with `--` in model ID | (Old bug - was fixed by passing full ID) | Use `backend_model` as-is with `--` |
| Proxy unreachable from telemetry | Proxy on `host` network, telemetry on compose | Add `extra_hosts` + `PROXY_URL=http://host.docker.internal:11434` |
| Stop marker blocks new pulls | Leftover `.alpaca-stop/{model}` file | Clean up on pull failure/completion |
| Browser shows stale search results | Cached `dashboard.js` | Add `Cache-Control: no-store` via `@app.after_request` |
| Benchmark returns empty for thinking models | Model outputs `<think>` block with no content | Add `"think": False` + `strip_thinking()` |
| Download hangs/stuck | `readline()` blocks forever | Use `select.select()` with 1s timeout |
| CLI games crash with EOFError in sandbox | `input()` gets no stdin | `run_code_once` redirects `/tmp/stdin.txt` for non-SQL langs |
| `outdated_only` runs ALL tests | Empty `test_ids` list is falsy at `if test_ids:` | Return `{"status":"No outdated benchmarks"}` early instead of falling through |
| `_outdated_test_ids` misses models | Sanitized filename vs public model name | Compare both `model` and `re.sub(r"[/:.]","_",model)` forms |
| Every "Apply to Profile" click is a no-op | The sanitiser stripped `-` and `_` but not `:`, so a public name never matched the router GGUF stem | Strip `:` and `.` too (web/app.py `apply_telemetry_recommendations`) |
| A near-blank page scores as a rendered UI | `_screenshot_has_content` only tested pixel variance, and anti-aliasing alone produces hundreds of greys | Ink-coverage floor, calibrated against the real screenshots in `docs/screenshots/` |
| A test can never pass | `num_predict: 0` survives `turn_budget`'s `max(0, …)` and reaches llama-server verbatim | `num_predict >= 1`, pinned per test by `test_benchmark_tests_schema.py` |
| A podcast bed is 30 s long | MusicGen's positional table is 1500 frames = 30.0 s; `min(duration_s*50, 1500)` clamps silently | Synthesize the bed procedurally; use MusicGen only for a short sting |
| The podcast bed drowns the speech | `_wav_bytes` peak-normalises before encoding | `podcast_mixer.encode_wav` never normalises |
| Voice ID rejects a speaker's own re-record | A hand-picked threshold | `identity_threshold()` derived from the worst profile's intra-speaker spread |
| A cloned voice sounds buzzy, "singing" or like it has an echo | A base voice far from the profile's `median_f0_hz` forces `correct_pitch` to phase-vocoder the whole render | Omit `voice` so the server pairs one within `PAIRING_MAX_SEMITONES`, or pin it on the profile |
| A narration changes voice from sentence to sentence | `voice_conversion` is called per sentence and samples its latent, so each sentence got a different draw | `convert` now seeds and forks the generator; `clone_seed: null` to opt out |
| `clone_seed: 1.5` is accepted and behaves like `1` | `int()` truncates | Reject non-integral values; a seed the caller did not mean is worse than no seed |
| Sphinx: ruff reformatted 28 unrelated lines | `ruff check --fix .` on a whole source tree | Scope `--fix` to new files; check `git diff` immediately after |

## Testing
**4,143 tests, all green on a clean machine** (`EXIT=0`, zero FAILED/ERROR). `pytest` from the repo
root: `testpaths = ["tests"]`, `pythonpath = ["."]`, `asyncio_default_fixture_loop_scope = function`,
and **`addopts = "-q -m 'not live'"`** - the `live` marker is deselected by default, so a bare
`pytest` is hermetic and a missing service is never a failure.

### `tests/conftest.py` - what it guarantees
- `pytest_report_header` prints the **live target** (`ALPACA_BASE_URL` / `ALPACA_PROXY_URL`,
  default `localhost:5000` / `localhost:11434`) and a `missing_facilities()` line
  (node / numpy / docker / playwright / sibling SharedLLM checkout), so a skip is never mistaken
  for coverage.
- An autouse `no_network` fixture replaces `socket.socket` with a subclass whose `connect` /
  `connect_ex` raise `RuntimeError` naming the test. **Anything not marked `live` physically
  cannot open a socket.** That is why a mock that is accidentally not applied shows up as a
  loud error rather than a 30-second connect timeout.
- Markers registered in `pyproject.toml`: `live`, `needs_docker`, `needs_node`, `needs_gpu`,
  `needs_sibling_repo`, `slow`.

### Test-file map
| File | Covers |
|---|---|
| `test_proxy_unit.py` | Slot allocation, queueing, VRAM budgeting, MTP/OOM ladders, keep-alive, SD native params |
| `test_web_integration.py` | Dashboard REST surface, profiles, pull, arcade, sandbox proxy, vision |
| `test_web_audio_routes.py` | All 7 `/api/audio/*` bridge routes (upstream path/verb/timeout per route) |
| `test_web_sd_routes.py` | All 7 `/api/sd/*` routes + `/api/companions` + QR burn-in |
| `test_web_sandbox_routes.py` | `serve` / `serve_ui` / `stop_serve` / `ui/{status,exec,restart,screenshot}` |
| `test_web_misc_routes.py` | Routing matrix, request mutations, telemetry recommendations, model lifecycle, ratings, pulls |
| `test_arcade.py` | Arcade service, publish pipeline, scores, achievements |
| `test_audio_server_unit.py` | The 606-line TTS/music service: validation order, model eviction, clone path |
| `test_voice_clone.py` + `test_voice_clone_identity.py` | Voice-clone analysis; **speaker identification and calibration** |
| `test_voice_clone_pairing.py` | Base-voice pairing: the free band, the all-costly fallback, the F0 cache, pins |
| `test_voice_clone_seed.py` | The pinned latent draw: repeat calls, a narration being one voice, the RNG not leaking, the `seed=None` opt-out |
| `test_audio_server_speaker_id.py` | `POST /api/voices/identify` and `/calibrate` |
| `test_podcast_mixer.py` + `test_podcast_routes.py` + `test_podcast_frontend.py` | Podcast Studio end to end |
| `test_telemetry_monitor_unit.py` / `test_analyzer_unit.py` / `test_imageops_unit.py` | The daemons and the deterministic image editor |
| `test_dashboard_frontend.py` / `test_podcast_frontend.py` | `dashboard.js` via the **node-slice** technique |
| `test_benchmark_tests_schema.py` | All 307 tests + `_compute_test_hash` semantics + the answer-key balance |
| `test_manuscript_rubric.py` / `test_screenshot_ink_coverage.py` | The illuminated-manuscript rubric; the blank-page gate |
| `test_composite_sprite_game.py` | The `composite` tool-test path end to end, through the real publisher |
| `test_svg_theme_pack_grader.py` / `test_install_theme_pack.py` | The SVG theme-pack grader and its installer |
| `test_sharedllm_contract.py` | alpaca's `_CANONICAL_TOOLS` vs SharedLLM's `ALLOWED_TOOLS` (both directions) |
| `test_scripts_entrypoints.py` | Every `scripts/` entry point imports; `rebalance_choice_keys` / `gen_reasoning_estimates` are idempotent |
| `test_cli_verifier_unit.py` | `tests/test-alpaca.py` (its hyphen filename means pytest never collects it) |
| `test_ui_bugfix_regressions.py` | The four dashboard bugs, so they stay fixed |
| `test_live_services.py`, `test_dashboard_e2e.py`, `test_smoke_playwright.py` | `live`-marked only; need a running stack |

### Running them
```bash
pytest                     # the hermetic suite (live deselected)
pytest -m live             # needs a running stack; see .github/workflows/live.yml
pytest -k podcast -q       # one area
```
CI (`.github/workflows/test.yml`) runs `pip install -r requirements-dev.txt` + mypy, `ruff check .`,
`pytest -m "not live"`, and **mypy as a real gate** on the three files below. It used to run 2 of
21 files (12.5% of the suite).

**Long runs:** a full `pytest` takes ~100 s, which exceeds a 120 s tool timeout, so background it
(`nohup bash -c '… & disown'`) and poll the log. Note that with `-q` the summary line is **not
always emitted** - assert on the exit code plus a zero `grep -cE "^FAILED|^ERROR"` count, never on
the absence of a "passed" line.

## Linting & Type Checking
```bash
# Ruff linting (scope --fix to NEW files: a whole-file --fix also repairs unrelated debt)
ruff check .

# mypy (selective strict mode) - these three are the CI gate
mypy web/app.py llm_benchmark_suite.py   # strict (disallow_untyped_defs = True)
mypy web/shared_llm_benchmark.py         # lenient (web.* exempt)
```
`mypy.ini` pins `python_version = 3.12` - modern numpy ships `type X = ...` stubs, and on 3.10
mypy aborts before checking anything. Ruff's `RUF001/002/003` are per-file-ignored for
`tts_text.py` and `manuscript_rubric.py`: the en-dash scripture ranges, IPA stress marks and curly
quotes are those modules' **input domain**, and ASCII-folding them breaks the tests.

## Common Workflows

### Rebuild & Restart Services (Use `sudo` for full model directory & docker socket access)
```bash
sudo docker compose up -d --build alpaca-web       # rebuild web
sudo docker compose up -d --build alpaca-telemetry # rebuild telemetry
sudo docker compose up -d --build alpaca-proxy     # rebuild proxy
sudo docker compose up -d                          # all services
```

### Safe Restart (never kills a running benchmark)
Direct `docker compose restart` kills any active benchmark run. After `git pull`,
use the idle watcher instead - it polls `/api/status` and restarts only when idle:
```bash
./scripts/restart-when-idle.sh [service] [max_wait_s] [poll_s]
# e.g. nohup ./scripts/restart-when-idle.sh alpaca-web > .tmp/restart-when-idle.log 2>&1 &
```
Defaults: `alpaca-web`, 24h max wait, 60s poll. Logs to `.tmp/restart-when-idle.log`.
Pure bind-mounted Python changes (e.g. `web/`, `llm_benchmark_suite.py`,
`online_providers.py`) need only a restart, not a rebuild.

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
1. **Search returns 0 results**: API works (curl test), browser has stale `dashboard.js` - hard-refresh or check cache headers
2. **Pull fails with "Interrupted"**: Check for leftover `.alpaca-stop/{model}` marker files
3. **400 on `/slots`**: Always pass `?model=` parameter, use full backend_model ID with `--`
4. **Telemetry container can't reach proxy**: Proxy is on `host` network; use `host.docker.internal:11434`
5. **Benchmark empty results for thinking models**: Ensure `"think": False` and `strip_thinking()` applied
6. **Download stuck**: Check if `select.select()` timeout is working, verify `_terminate_process()` is called on cancel
7. **Model not appearing in list**: Run `python alpaca-puller.py reindex` to rebuild router symlinks

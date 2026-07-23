#!/bin/bash
# sd-entrypoint.sh - Hardware-aware startup and compilation script for stable-diffusion.cpp
set -e

echo "[sd-entrypoint] Starting stable-diffusion.cpp container entrypoint..."

# Ensure model and router directories exist with full access permissions
mkdir -p /router-models/companions/lora /models/companions 2>/dev/null || true
chmod -R 777 /router-models /models 2>/dev/null || true

# 1. Hardware Detection
HARDWARE="CPU"
CMAKE_FLAGS=""

echo "[sd-entrypoint] Running hardware detection..."

# Check CUDA/NVIDIA GPU
if lspci 2>/dev/null | grep -q -i nvidia || command -v nvidia-smi >/dev/null 2>&1 || [ -d /usr/local/cuda ] || [ -f /proc/driver/nvidia/version ]; then
    echo "[sd-entrypoint] NVIDIA GPU / CUDA detected."
    HARDWARE="CUDA"
    CMAKE_FLAGS="-DSD_CUDA=ON -DGGML_CUDA=ON"
# Check Apple Silicon (Darwin/arm64)
elif uname -m | grep -q "arm64" && (sysctl -a 2>/dev/null | grep -i -q "Apple" || uname -s | grep -q "Darwin"); then
    echo "[sd-entrypoint] Apple Silicon detected."
    HARDWARE="METAL"
    CMAKE_FLAGS="-DSD_METAL=ON -DGGML_METAL=ON"
# Check AMD/ROCm
elif lspci 2>/dev/null | grep -q -i -E "amd|radeon" || [ -d /opt/rocm ] || command -v rocminfo >/dev/null 2>&1; then
    echo "[sd-entrypoint] AMD/ROCm detected."
    HARDWARE="ROCM"
    CMAKE_FLAGS="-DSD_ROCM=ON -DGGML_HIPBLAS=ON"
# Check Vulkan fallback
elif command -v vulkaninfo >/dev/null 2>&1; then
    echo "[sd-entrypoint] Vulkan hardware detected."
    HARDWARE="VULKAN"
    CMAKE_FLAGS="-DSD_VULKAN=ON -DGGML_VULKAN=ON"
else
    echo "[sd-entrypoint] No hardware acceleration detected. Using CPU only."
    HARDWARE="CPU"
    CMAKE_FLAGS=""
fi

# ── SD_DEBUG: optional startup diagnostic dump ────────────────────────────────
# Set SD_DEBUG=true to print detected hardware, CUDA capability, and the final
# placement arguments before sd-server starts. Invaluable when tuning VRAM on
# cards like the RTX 4060 (8 GB) where layer counts matter a lot.
if [ "${SD_DEBUG:-false}" = "true" ]; then
    echo "[sd-entrypoint][DEBUG] ── Hardware Detection Report ──────────────────"
    echo "[sd-entrypoint][DEBUG]   Detected backend : $HARDWARE"
    echo "[sd-entrypoint][DEBUG]   CMake flags      : ${CMAKE_FLAGS:-<none>}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "[sd-entrypoint][DEBUG]   nvidia-smi GPU   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unavailable')"
        echo "[sd-entrypoint][DEBUG]   CUDA version     : $(nvidia-smi | grep 'CUDA Version' | awk '{print $NF}' || echo 'unavailable')"
    fi
    echo "[sd-entrypoint][DEBUG]   SD_GPU_LAYERS    : ${SD_GPU_LAYERS:-<not set, default=40 for Qwen>}"
    echo "[sd-entrypoint][DEBUG]   SD_CPU_THREADS   : ${SD_CPU_THREADS:-<not set, auto>}"
    echo "[sd-entrypoint][DEBUG] ──────────────────────────────────────────────"
fi

# 2. Build Verification & Dynamic Compilation
BUILD_DIR="/app/stable-diffusion.cpp/build"
BINARY_PATH="$BUILD_DIR/bin/sd-server"

# Prefer a prebuilt sd-server baked into the image (e.g. ghcr.io/leejet/
# stable-diffusion.cpp:master-cuda ships it at /sd.cpp/bin/sd-server).
if [ -x "/sd.cpp/bin/sd-server" ]; then
    echo "[sd-entrypoint] Using prebuilt binary at /sd.cpp/bin/sd-server. Skipping compilation."
    BINARY_PATH="/sd.cpp/bin/sd-server"
elif [ -f "$BINARY_PATH" ]; then
    echo "[sd-entrypoint] Found pre-compiled binary at $BINARY_PATH. Skipping compilation."
else
    COMPILE_SUCCESS=false
    if [ "$HARDWARE" != "CPU" ]; then
        echo "[sd-entrypoint] Compiling stable-diffusion.cpp with $HARDWARE acceleration flags: $CMAKE_FLAGS"
        cd "$BUILD_DIR"
        if cmake .. $CMAKE_FLAGS && cmake --build . --config Release --target sd-server; then
            if [ -f "$BINARY_PATH" ]; then
                echo "[sd-entrypoint] Compilation with $HARDWARE acceleration successful."
                COMPILE_SUCCESS=true
            else
                echo "[sd-entrypoint] sd-server target not found after compilation."
            fi
        else
            echo "[sd-entrypoint] Acceleration build failed. Cleaning build directory..."
        fi
    fi

    # Fallback to CPU compilation if acceleration failed or wasn't detected
    if [ "$COMPILE_SUCCESS" = false ]; then
        echo "[sd-entrypoint] Falling back to CPU-only compilation..."
        cd /app/stable-diffusion.cpp
        rm -rf "$BUILD_DIR"
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        if cmake .. && cmake --build . --config Release --target sd-server; then
            if [ -f "$BINARY_PATH" ]; then
                echo "[sd-entrypoint] CPU compilation successful."
            else
                echo "[sd-entrypoint] ERROR: CPU compilation completed but binary $BINARY_PATH is missing!"
                exit 1
            fi
        else
            echo "[sd-entrypoint] ERROR: CPU-only compilation failed!"
            exit 1
        fi
    fi
fi

# 3. Model Loading & Config Parsing
CONFIG_FILE="/router-models/sd_active_model.json"
MODEL_PATH=""
VAE_PATH=""
CLIP_L_PATH=""
T5XXL_PATH=""
LLM_PATH=""
LISTEN_IP="0.0.0.0"
LISTEN_PORT="8081"
EXTRA_ARGS=""
OFFLOAD_ARGS=""

if [ -f "$CONFIG_FILE" ]; then
    echo "[sd-entrypoint] Reading active model configuration from $CONFIG_FILE"
    # Helper to safely parse JSON keys using python
    MODEL_PATH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('model_path') or '')")
    VAE_PATH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('vae_path') or '')")
    CLIP_L_PATH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('clip_l_path') or '')")
    T5XXL_PATH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('t5xxl_path') or '')")
    LLM_PATH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('llm_path') or '')")
    MODEL_FAMILY=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('model_family') or '')")
    LISTEN_IP=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('host') or '0.0.0.0')")
    LISTEN_PORT=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('port') or '8081')")
    EXTRA_ARGS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('extra_args') or '')")
    GPU_LAYERS_CONF=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('gpu_layers') or '')")
    THREADS_CONF=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('threads') or '')")

    # Qwen-Image / qwen-image-edit models use a Qwen2.5-VL LLM text encoder and a
    # separate diffusion model. They can be large (Q4_K_M ≈ 12 GB total), but the
    # diffusion backbone itself is usually well under 8 GB so the RTX 4060's 8 GB
    # VRAM can hold a significant portion.
    #
    # SD_GPU_LAYERS controls how many transformer layers are placed on the GPU:
    #   - Higher value  → more VRAM used, faster generation
    #   - Lower value   → less VRAM used, more CPU fallback
    # Start with 40 on an RTX 4060 and increase if generation is slow and VRAM
    # headroom permits (check with: docker exec sd-server nvidia-smi).
    # To restore the old fully-automatic behaviour, set SD_GPU_LAYERS=0 and add
    # --auto-fit via the model profile's extra_args key instead.
    #
    # If the user has configured profile-specific GPU layers (via models.ini/profile.json),
    # use that; otherwise fall back to the Qwen heuristics / SD_GPU_LAYERS environment default.
    if [ "$MODEL_FAMILY" = "qwen-image" ] || [ -n "$LLM_PATH" ] || echo "$MODEL_PATH" | grep -qi "qwen"; then
        if [ -n "$GPU_LAYERS_CONF" ]; then
            OFFLOAD_ARGS="--qwen-image-layers $GPU_LAYERS_CONF"
        else
            OFFLOAD_ARGS="--qwen-image-layers ${SD_GPU_LAYERS:-40}"
        fi
    fi
else
    echo "[sd-entrypoint] No configuration file found at $CONFIG_FILE"
fi

if [ -z "$MODEL_PATH" ]; then
    echo "[sd-entrypoint] Scanning for models in /router-models..."
    MODEL_PATH=$(find /router-models -maxdepth 1 \( -name "*.safetensors" -o -name "*.gguf" \) | head -n 1)
fi

# If still no model, wait in an idle loop
if [ -z "$MODEL_PATH" ] || [ ! -e "$MODEL_PATH" ]; then
    echo "[sd-entrypoint] WARNING: No stable-diffusion model file found. Entering idle loop. Please pull a model to start."
    while true; do
        sleep 10
    done
fi

# ── Config check (diagnostics ONLY — never changes launch behaviour) ────────
# Surface the resolved model configuration and warn (without failing) about
# missing companions, so a bad load is easy to diagnose from `docker logs`.
echo "[sd-entrypoint][check] ── Model Load Configuration Check ──────────────"
echo "[sd-entrypoint][check]   Model path : $MODEL_PATH"
if [ -e "$MODEL_PATH" ]; then
    echo "[sd-entrypoint][check]   Model file : present ($(du -h "$MODEL_PATH" 2>/dev/null | cut -f1))"
else
    echo "[sd-entrypoint][check]   Model file : MISSING — sd-server load will fail"
fi
echo "[sd-entrypoint][check]   Family     : ${MODEL_FAMILY:-<undetected>}"
if [ "$MODEL_FAMILY" = "qwen-image" ] || echo "$MODEL_PATH" | grep -qi "qwen"; then
    echo "[sd-entrypoint][check]   Qwen-Image load path:"
    echo "[sd-entrypoint][check]     - standalone diffusion model (--diffusion-model)"
    echo "[sd-entrypoint][check]     - GPU layers (--qwen-image-layers): ${GPU_LAYERS_CONF:-${SD_GPU_LAYERS:-40}}"
    if [ -n "$LLM_PATH" ]; then
        if [ -f "$LLM_PATH" ]; then
            echo "[sd-entrypoint][check]     - LLM text encoder: $LLM_PATH (present)"
        else
            echo "[sd-entrypoint][check]     - LLM text encoder: $LLM_PATH (MISSING — load may fail)"
        fi
    else
        echo "[sd-entrypoint][check]     - LLM text encoder: <none specified>"
    fi
    if [ -n "$VAE_PATH" ]; then
        if [ -f "$VAE_PATH" ]; then
            echo "[sd-entrypoint][check]     - VAE companion: $VAE_PATH (present)"
        else
            echo "[sd-entrypoint][check]     - VAE companion: $VAE_PATH (MISSING — load may fail)"
        fi
    else
        echo "[sd-entrypoint][check]     - VAE companion: <none specified>"
    fi
fi
echo "[sd-entrypoint][check]   Extra args : ${EXTRA_ARGS:-<none>}"
echo "[sd-entrypoint][check] ────────────────────────────────────────────────"

echo "[sd-entrypoint] Loading model: $MODEL_PATH"
if [ "$MODEL_FAMILY" = "qwen-image" ] || [ -n "$LLM_PATH" ] || [ -n "$VAE_PATH" ]; then
    echo "[sd-entrypoint] Loading as standalone diffusion model..."
    CMD=("$BINARY_PATH" "--diffusion-model" "$MODEL_PATH" "--listen-ip" "$LISTEN_IP" "--listen-port" "$LISTEN_PORT")
else
    CMD=("$BINARY_PATH" "-m" "$MODEL_PATH" "--listen-ip" "$LISTEN_IP" "--listen-port" "$LISTEN_PORT")
fi


if [ -n "$VAE_PATH" ] && [ -f "$VAE_PATH" ]; then
    echo "[sd-entrypoint] Using VAE companion file: $VAE_PATH"
    CMD+=("--vae" "$VAE_PATH")
fi

if [ -n "$CLIP_L_PATH" ] && [ -f "$CLIP_L_PATH" ]; then
    echo "[sd-entrypoint] Using CLIP L companion file: $CLIP_L_PATH"
    CMD+=("--clip_l" "$CLIP_L_PATH")
fi

if [ -n "$T5XXL_PATH" ] && [ -f "$T5XXL_PATH" ]; then
    echo "[sd-entrypoint] Using T5XXL companion file: $T5XXL_PATH"
    CMD+=("--t5xxl" "$T5XXL_PATH")
fi

# qwen-image / qwen-image-edit use a Qwen2.5-VL text encoder passed via --llm.
if [ -n "$LLM_PATH" ] && [ -f "$LLM_PATH" ]; then
    echo "[sd-entrypoint] Using LLM (text encoder) companion file: $LLM_PATH"
    CMD+=("--llm" "$LLM_PATH")
fi

if [ -d "/router-models/companions/lora" ]; then
    echo "[sd-entrypoint] Using LoRA directory: /router-models/companions/lora"
    CMD+=("--lora-model-dir" "/router-models/companions/lora")
fi

# Limit CPU threads so diffusion doesn't peg every core at 100% (which can
# thermally shut down small-form-factor / weak-cooled hosts during long
# CPU-bound generations). Default: leave 2 cores free for the OS. Override with
# the SD_CPU_THREADS env var.
if [ -n "$THREADS_CONF" ]; then
    THREADS="$THREADS_CONF"
elif [ -n "$SD_CPU_THREADS" ]; then
    THREADS="$SD_CPU_THREADS"
else
    THREADS=$(( $(nproc) - 2 ))
    [ "$THREADS" -lt 1 ] && THREADS=1
fi
echo "[sd-entrypoint] Using $THREADS CPU threads for sd-server (of $(nproc) available)."
CMD+=(--threads "$THREADS")

if [ -n "$OFFLOAD_ARGS" ]; then
    # Log whether we are using GPU layers or a CPU-offload fallback so it is
    # immediately obvious in `docker logs` which path was chosen.
    if echo "$OFFLOAD_ARGS" | grep -q "\-\-qwen-image-layers"; then
        echo "[sd-entrypoint] GPU layer placement enabled: $OFFLOAD_ARGS  (RTX 4060: adjust SD_GPU_LAYERS to tune VRAM usage)"
    else
        echo "[sd-entrypoint] CPU offload mode: $OFFLOAD_ARGS"
    fi
    # shellcheck disable=SC2206
    CMD+=($OFFLOAD_ARGS)
fi

if [ -n "$EXTRA_ARGS" ]; then
    echo "[sd-entrypoint] Appending extra arguments: $EXTRA_ARGS"
    # Word splitting is intended here for extra flags
    # shellcheck disable=SC2206
    CMD+=($EXTRA_ARGS)
fi

# Final command-line summary (always shown; also shown by SD_DEBUG for completeness).
echo "[sd-entrypoint] Executing: ${CMD[*]}"
if [ "${SD_DEBUG:-false}" = "true" ]; then
    echo "[sd-entrypoint][DEBUG] CUDA enabled : $( [ "$HARDWARE" = 'CUDA' ] && echo yes || echo no )"
    echo "[sd-entrypoint][DEBUG] GPU layers   : ${OFFLOAD_ARGS:-<none set>}"
fi
# Run sd-server in the foreground but trap its exit so a failed model load
# (e.g. incompatible GGUF) drops the container into an idle loop instead of
# exiting and crash-looping under `restart: always`.
set +e
"${CMD[@]}" &
SD_PID=$!
wait "$SD_PID"
SD_RC=$?
set -e
if [ "$SD_RC" -ne 0 ]; then
    echo "[sd-entrypoint] sd-server exited with code $SD_RC. Entering idle loop (model may be incompatible or missing)."
    while true; do
        sleep 30
    done
fi

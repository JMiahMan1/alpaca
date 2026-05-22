#!/bin/sh
# Entrypoint for llama-server: auto-detect model type and apply safe flags.
set -e

FLAGS=$(python3 /app/llama-server-flags.py)
echo "[entrypoint] llama-server flags: $FLAGS"

exec llama-server $FLAGS

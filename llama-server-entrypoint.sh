#!/bin/sh
# Entrypoint for llama-server: auto-detect model type and apply safe flags.
set -e

# Install python3 if not present (minimal image doesn't include it)
if ! command -v python3 >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq python3 > /dev/null 2>&1
fi

FLAGS=$(python3 /scripts/llama-server-flags.py)
echo "[entrypoint] llama-server flags: $FLAGS"

exec /app/llama-server $FLAGS

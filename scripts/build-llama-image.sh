#!/bin/bash
# Build a llama.cpp server image from source, for a model newer than the newest pinned official image.
#   scripts/build-llama-image.sh <llama.cpp tag, e.g. b12708> [CUDA arch, default 89 = RTX 40xx]
# Tags the result localhost/llama.cpp:server-cuda12-<tag>-sm<arch>; set LLAMA_BASE_IMAGE to that in
# .env and rebuild llama-server (docker compose up -d --build --no-deps llama-server) while idle.
set -euo pipefail
TAG="${1:?llama.cpp tag, for example b12708}"
ARCH="${2:-89}"
SRC="${LLAMA_SRC_DIR:-$HOME/build}/llama.cpp-$TAG"
IMAGE="localhost/llama.cpp:server-cuda12-$TAG-sm$ARCH"
mkdir -p "$(dirname "$SRC")"
[ -d "$SRC" ] || git clone --quiet --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
REV=$(git -C "$SRC" rev-parse --short HEAD)
docker build -f "$SRC/.devops/cuda.Dockerfile" --target server \
  --build-arg CUDA_DOCKER_ARCH="$ARCH" --build-arg APP_VERSION="$TAG" --build-arg APP_REVISION="$REV" \
  -t "$IMAGE" "$SRC"
docker builder prune -af >/dev/null  # the CUDA devel layers are ~10 GB
echo "built $IMAGE (llama.cpp $TAG $REV). Set LLAMA_BASE_IMAGE=$IMAGE in .env, then rebuild llama-server."

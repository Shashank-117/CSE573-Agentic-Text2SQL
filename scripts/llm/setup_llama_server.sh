#!/usr/bin/env bash
# One-time setup of a local, pinned Qwen2.5-Coder-7B-Instruct (Q4_K_M) server.
#
# - llama-cpp-python (CUDA 12.4 wheel) in its own venv, no sudo needed
# - CUDA runtime + cuBLAS from NVIDIA's pip wheels
# - official Qwen GGUF at a pinned revision, verified by SHA-256
#
# Env overrides: LLM_VENV (default ~/.venvs/llama-server), LLM_MODEL_DIR (default ~/models)
set -euo pipefail

LLM_VENV="${LLM_VENV:-$HOME/.venvs/llama-server}"
LLM_MODEL_DIR="${LLM_MODEL_DIR:-$HOME/models}"

LLAMA_CPP_VERSION="0.3.16"
WHEEL_INDEX="https://abetlen.github.io/llama-cpp-python/whl/cu124"
HF_REPO="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
HF_REVISION="13fb94bfda8c8cf22497dc57b78f391a9acb426a"
MODEL_FILE="qwen2.5-coder-7b-instruct-q4_k_m.gguf"
MODEL_SHA256="509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }

if [ ! -x "$LLM_VENV/bin/python" ]; then
  uv venv "$LLM_VENV" --python 3.10
fi
uv pip install --python "$LLM_VENV/bin/python" \
  "llama-cpp-python[server]==$LLAMA_CPP_VERSION" \
  "nvidia-cuda-runtime-cu12==12.4.*" "nvidia-cublas-cu12==12.4.*" \
  --extra-index-url "$WHEEL_INDEX" --index-strategy unsafe-best-match

mkdir -p "$LLM_MODEL_DIR"
target="$LLM_MODEL_DIR/$MODEL_FILE"
if [ ! -f "$target" ] || ! echo "$MODEL_SHA256  $target" | sha256sum --check --status; then
  echo "Downloading $MODEL_FILE (~4.7 GB) at revision ${HF_REVISION:0:12}..."
  curl -L --fail --retry 3 -C - -o "$target.part" \
    "https://huggingface.co/$HF_REPO/resolve/$HF_REVISION/$MODEL_FILE"
  mv "$target.part" "$target"
fi
echo "$MODEL_SHA256  $target" | sha256sum --check
echo "Ready. Start the server with: bash scripts/llm/start_llama_server.sh"

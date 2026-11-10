#!/usr/bin/env bash
# Serve Qwen2.5-Coder-7B-Instruct (Q4_K_M) on an OpenAI-compatible endpoint.
#   http://127.0.0.1:${LLM_PORT:-8080}/v1/chat/completions
#
# Defaults sized for an 8 GB GPU: all layers on GPU, 12k context, 1 request
# at a time (the harness is sequential). Override via env vars.
set -euo pipefail

LLM_VENV="${LLM_VENV:-$HOME/.venvs/llama-server}"
LLM_MODEL_DIR="${LLM_MODEL_DIR:-$HOME/models}"
LLM_PORT="${LLM_PORT:-8080}"
LLM_CTX="${LLM_CTX:-12288}"
LLM_GPU_LAYERS="${LLM_GPU_LAYERS:--1}"
MODEL="$LLM_MODEL_DIR/qwen2.5-coder-7b-instruct-q4_k_m.gguf"

nvidia_dir="$LLM_VENV/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$nvidia_dir/cuda_runtime/lib:$nvidia_dir/cublas/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

[ -f "$MODEL" ] || { echo "Model missing; run scripts/llm/setup_llama_server.sh"; exit 1; }

exec "$LLM_VENV/bin/python" -m llama_cpp.server \
  --model "$MODEL" \
  --model_alias qwen2.5-coder-7b-instruct \
  --n_gpu_layers "$LLM_GPU_LAYERS" \
  --n_ctx "$LLM_CTX" \
  --flash_attn true \
  --chat_format chatml \
  --host 127.0.0.1 --port "$LLM_PORT"

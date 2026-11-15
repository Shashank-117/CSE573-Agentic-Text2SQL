#!/usr/bin/env bash
# Stop the local llama.cpp server started by start_llama_server.sh (frees ~7 GB VRAM).
set -euo pipefail
if pkill -f "[l]lama_cpp.server"; then
  for _ in $(seq 1 20); do
    pgrep -f "[l]lama_cpp.server" >/dev/null || { echo "llama server stopped"; exit 0; }
    sleep 0.5
  done
  echo "server did not exit; sending SIGKILL"; pkill -9 -f "[l]lama_cpp.server" || true
else
  echo "no llama server running"
fi

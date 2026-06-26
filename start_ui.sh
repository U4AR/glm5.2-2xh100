#!/usr/bin/env bash
# Launch the GLM-5.2 test chat UI (zero deps, stdlib only).
# Serves a chat page on :8080 and proxies to the model server on :8000.
set -euo pipefail
PORT="${1:-8080}"
cd "$(dirname "$0")"
echo "Starting chat UI on port $PORT (model expected on :8000)…"
echo "Open the forwarded port $PORT in your browser (VS Code forwards it automatically)."
exec python3 chat_ui.py "$PORT"

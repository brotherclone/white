#!/usr/bin/env bash
set -euo pipefail

# Load .env if present
if [ -f .env ]; then
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue
    export "$line"
  done < <(grep -v '^#' .env | grep '=')
fi

if [ -z "${SHRINKWRAP_OUTPUT_DIR:-}" ]; then
  echo "Error: SHRINKWRAP_OUTPUT_DIR is not set. Add it to your environment or .env file." >&2
  exit 1
fi

cleanup() {
  echo ""
  echo "Stopping servers…"
  kill "$API_PID" "$UI_PID" 2>/dev/null || true
  wait "$API_PID" "$UI_PID" 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM

echo "Starting API server on :8000 …"
python -m white_api.candidate_server \
  --shrink-wrapped-dir "$SHRINKWRAP_OUTPUT_DIR" \
  --no-open &
API_PID=$!

echo "Starting Next.js dev server on :3000 …"
(cd packages/client && source ~/.nvm/nvm.sh && nvm use 20 && npm run dev) &
UI_PID=$!

echo "Both servers running. Press Ctrl+C to stop."
wait

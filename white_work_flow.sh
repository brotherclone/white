#!/usr/bin/env bash
set -euo pipefail

# Load .env if present — set -a exports every assignment, set +a stops it
if [ -f .env ]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

if [ -z "${SHRINKWRAP_OUTPUT_DIR:-}" ]; then
  echo "Error: SHRINKWRAP_OUTPUT_DIR is not set. Add it to your environment or .env file." >&2
  exit 1
fi

API_PID=""
UI_PID=""

cleanup() {
  echo ""
  echo "Stopping servers…"
  [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null || true
  [ -n "$UI_PID" ] && kill "$UI_PID" 2>/dev/null || true
  [ -n "$API_PID" ] && wait "$API_PID" 2>/dev/null || true
  [ -n "$UI_PID" ] && wait "$UI_PID" 2>/dev/null || true
}
trap cleanup EXIT

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

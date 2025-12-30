#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"

cd "$ROOT_DIR/modelverse"

if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
fi

exec uvicorn main:app --host "$HOST" --port "$PORT"


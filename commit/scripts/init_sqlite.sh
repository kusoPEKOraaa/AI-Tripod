#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DB_PATH="${DB_PATH:-$ROOT_DIR/modelverse/modelverse.db}"
SCHEMA_PATH="${SCHEMA_PATH:-$ROOT_DIR/commit/db/sqlite_schema.sql}"

if ! command -v sqlite3 >/dev/null 2>&1; then
  echo "sqlite3 not found in PATH; please install sqlite3 first." >&2
  exit 1
fi

mkdir -p "$(dirname "$DB_PATH")"
sqlite3 "$DB_PATH" < "$SCHEMA_PATH"

echo "Initialized SQLite DB at: $DB_PATH"


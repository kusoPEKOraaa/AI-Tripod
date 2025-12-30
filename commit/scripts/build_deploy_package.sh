#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT_DIR/commit/deploy"
PKG_PATH="$OUT_DIR/modelverse-src.tar.gz"
SUMS_PATH="$OUT_DIR/SHA256SUMS.txt"

mkdir -p "$OUT_DIR"

tar \
  --exclude=".git" \
  --exclude=".claude" \
  --exclude="modelverse/.claude" \
  --exclude="modelverse/__pycache__" \
  --exclude="**/.DS_Store" \
  --exclude="modelverse/hf_cache" \
  --exclude="modelverse/uploads" \
  --exclude="modelverse/*.log" \
  --exclude="modelverse/nohup.out" \
  --exclude="*.pdf" \
  -czf "$PKG_PATH" \
  -C "$ROOT_DIR" \
  modelverse requirements-all.txt DEPLOYMENT.md .gitignore

(
  cd "$OUT_DIR"
  sha256sum "$(basename "$PKG_PATH")" > "$SUMS_PATH"
)

echo "Built package: $PKG_PATH"

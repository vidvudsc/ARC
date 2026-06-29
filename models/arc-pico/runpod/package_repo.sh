#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-/tmp/arc_pico_runpod.tgz}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

COPYFILE_DISABLE=1 tar \
  --no-xattrs \
  --exclude='.DS_Store' \
  --exclude='._*' \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='src/**/__pycache__' \
  --exclude='scripts/__pycache__' \
  --exclude='data' \
  --exclude='downloads' \
  --exclude='*.tgz' \
  --exclude='runs' \
  --exclude='checkpoints' \
  -czf "$OUT" .

ls -lh "$OUT"

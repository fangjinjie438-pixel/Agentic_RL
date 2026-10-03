#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATA_DIR="${DATA_DIR:-data/synthetic}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m procurement_rl.retrieval --corpus "$DATA_DIR/train/corpus.jsonl" "$DATA_DIR/val/corpus.jsonl" "$DATA_DIR/test/corpus.jsonl" --host "${RETRIEVER_HOST:-127.0.0.1}" --port "${RETRIEVER_PORT:-8000}" "$@"

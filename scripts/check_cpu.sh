#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python scripts/bootstrap_search_r1.py --check
python -m pytest -q
mkdir -p outputs
CHECK_DIR="$(mktemp -d "$ROOT/outputs/cpu-check-XXXXXX")"
python -m procurement_rl prepare --output "$CHECK_DIR/data" --train-size 24 --val-size 6 --test-size 6 --seed 42
python -m procurement_rl export --data-dir "$CHECK_DIR/data"
python -m procurement_rl demo --data-dir "$CHECK_DIR/data" --split test --output "$CHECK_DIR/demo"
for profile in school_single school_smoke autodl_3b autodl_3b_dual4090; do
  python scripts/preflight.py --profile "configs/$profile.yaml" --config-only
done
python scripts/launch_train.py --profile configs/school_single.yaml --algorithm igpo --data-dir "$CHECK_DIR/data" --dry-run > "$CHECK_DIR/launch-dry-run.txt"
python -m pip freeze > "$CHECK_DIR/pip-freeze.txt"
echo "CPU verification artifacts: $CHECK_DIR"
echo 'This check does not run a language model or GPU training.'

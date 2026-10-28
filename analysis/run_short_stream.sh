#!/usr/bin/env bash
# One stream of reduced-budget runs. Usage: bash run_stream.sh [--wait-for TAG] "VARIANT SEED" ...
PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../results/analysis/train_logs"
cd "$HERE"
if [ "$1" = "--wait-for" ]; then
  until grep -q "^done" "$OUT/$2.stdout" 2>/dev/null; do sleep 60; done
  shift 2
fi
for job in "$@"; do
  set -- $job
  echo "start $1 $2 $(date)"
  $PY train_variant.py --variant "$1" --seed "$2" --short --no-grad-ckpt > "$OUT/${1}_s${2}_short.stdout" 2>&1
  echo "end $1 $2 $(date) rc=$?"
done
echo "stream finished $(date)"

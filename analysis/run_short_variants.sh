#!/usr/bin/env bash
# Reduced-budget controlled ablation (5 + 10 epochs, otherwise the original protocol).
PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../results/analysis/train_logs"
mkdir -p "$OUT"
cd "$HERE"
for job in "A2 42" "anatOnly 42" "simpleEmb 42" "segOnly 42" "A2 7"; do
  set -- $job
  echo "start $1 $2 $(date)"
  $PY train_variant.py --variant "$1" --seed "$2" --short --no-grad-ckpt > "$OUT/${1}_s${2}_short.stdout" 2>&1
  echo "end $1 $2 $(date) rc=$?"
done
echo "queue finished $(date)"

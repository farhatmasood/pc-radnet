#!/usr/bin/env bash
# Sequential training queue for the controlled ablations.
PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../results/analysis/train_logs"
mkdir -p "$OUT"
cd "$HERE"
run() {
  local v=$1 s=$2
  if [ -f "$HERE/../checkpoints/variants/${v}_s${s}_best.pt" ] && grep -q "done" "$OUT/${v}_s${s}.stdout" 2>/dev/null; then
    echo "skip $v $s"; return
  fi
  echo "start $v $s $(date)"
  $PY train_variant.py --variant "$v" --seed "$s" --no-grad-ckpt > "$OUT/${v}_s${s}.stdout" 2>&1
  echo "end $v $s $(date) rc=$?"
}
run noSDA 42
run anatOnly 42
run segOnly 42
run simpleEmb 42
run A2 7
run A2 123
run levelTrue 42
run clinOnly 42
echo "queue finished $(date)"

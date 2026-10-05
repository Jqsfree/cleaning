#!/usr/bin/env bash
# Phase 0.2 pilot：对 clean 子集跑 CLIP 以生成 clip_fail 供误杀审计（5000 条）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python3"
CLEAN="${ROOT}/data/runs/exo_agriculture/machine_0818/05_clean/run01/农业采集_0813-0818_clean_0828.csv"
OUT="${ROOT}/data/runs/exo_agriculture/machine_0818/06_tools/v1_risk_audit/clip_pilot"
mkdir -p "$OUT"
echo "[clip_pilot] sample=5000 → $OUT"
PYTHONPATH=02_脚本 "$PY" 02_脚本/tools/run_exo_agriculture_cascade_clip.py \
  "$CLEAN" -o "$OUT" \
  --sample 5000 --seed 42 \
  --skip-metadata-filter \
  --cache-dir qc_thumb_cache/exemplar_sim \
  --batch-rows 2000

#!/usr/bin/env bash
# 断点续跑 parent_child CLIP+LR 全量打分；进程退出后自动重试直至 APPLY_DONE。
set -euo pipefail

REPO="${REPO:-/home/jqs/projects/clean_DATASET}"
CLIP_DIR="${CLIP_DIR:-$REPO/data/runs/parent_child/machine_0818_lt50/06_tools/clip_lr_v1}"
PASS="${PASS:-$REPO/data/runs/parent_child/machine_0818_lt50/06_tools/minilm_v3/亲子互动_<50%_run02_minilm_v3_pass_0820.csv}"
PY="${PY:-/home/jqs/miniconda3/envs/data_cleaning/bin/python}"
LOG="$CLIP_DIR/apply.log"

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7897}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7897}"
export PYTHONPATH="$REPO/02_脚本"

cd "$REPO"
mkdir -p "$CLIP_DIR"

attempt=0
while ! grep -q 'APPLY_DONE' "$LOG" 2>/dev/null; do
  attempt=$((attempt + 1))
  echo "[$(date '+%F %T')] attempt=$attempt start apply" >> "$LOG"
  "$PY" "$REPO/02_脚本/tools/apply_parent_child_thumb_clip.py" \
    "$PASS" -o "$CLIP_DIR/" \
    --model "$REPO/models/parent_child_thumb_clip_lr.pkl" \
    --cache-dir "$CLIP_DIR/thumb_cache" \
    --chunksize 5000 --thumb-workers 32 --batch-size 64 \
    >> "$LOG" 2>&1 || true
  if grep -q 'APPLY_DONE' "$LOG" 2>/dev/null; then
    break
  fi
  echo "[$(date '+%F %T')] attempt=$attempt exited early; sleep 10" >> "$LOG"
  sleep 10
done

echo "[$(date '+%F %T')] all done" >> "$LOG"

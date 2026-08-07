#!/usr/bin/env bash
# 负例 neg_sim ckpt 跑完后：按「高通过率可误杀」重写出 keep（不重下缩略图）
set -euo pipefail
cd "$(dirname "$0")/../.."
BATCH=data/runs/live_sell/machine_0805
CKPT=$BATCH/06_tools/纯直播机采_0805_exemplar_keep_high_0805_neg_sim.ckpt.csv
POOL=$BATCH/06_tools/纯直播机采_0805_exemplar_keep_high_0805.csv
N=$(python -c "import pandas as pd; print(len(pd.read_csv('$CKPT')))" 2>/dev/null || echo 0)
P=$(python -c "import pandas as pd; print(len(pd.read_csv('$POOL')))" 2>/dev/null || echo 0)
echo "ckpt=$N pool=$P"
if [[ "$N" -lt "$P" ]]; then
  echo "neg_sim 未完成 ($N/$P)，请等 filter_exemplar_neg 跑完或续跑后再执行"
  exit 1
fi
exec python -u 02_脚本/tools/filter_exemplar_neg.py \
  --labels "$BATCH/06_tools/neg_labels_human_0806.csv" \
  --pool "$POOL" \
  --pos-sim "$BATCH/06_tools/纯直播机采_0805_records_quality_0805_exemplar_sim_0805.csv" \
  --pos-bank data/assets/exemplars/yt_live_scene/ \
  --neg-bank data/assets/exemplars/yt_live_scene_neg_human_f/ \
  -o "$BATCH/06_tools/" \
  --objective pass_rate \
  --min-keep-labels 5 \
  --min-sim 0.70 \
  --batch-rows 5000 \
  --thumb-workers 24

# exemplars — CLIP 样例原型 bank

域内正/负视觉原型（`prototypes.npy`，ViT-B-32 / openai）。完整复用规则见 [`../CLIP_ASSETS.md`](../CLIP_ASSETS.md)。

## 现有 bank

| 目录 | 规模 | 用途 |
|------|------|------|
| [`yt_live_scene/`](yt_live_scene/) | 10×512 + 抽帧 | YouTube「直播场景」正例；见同目录 README |
| [`yt_live_scene_neg_human_f/`](yt_live_scene_neg_human_f/) | 224×512 | 人工 F 缩略图负例，margin 二滤 |

**勿混域：** 直播 ≠ 带货 ≠ 农业/户外。带货请另建 bank。

## 工具

```bash
# 样例 mp4 → 原型
02_脚本/tools/build_exemplar_bank.py \
  --video-dir /path/to/sample_mp4s \
  -o data/assets/exemplars/<bank_name>/

# 候选打分
02_脚本/tools/score_exemplar_sim.py candidates.csv \
  --bank data/assets/exemplars/yt_live_scene/ \
  -o $BATCH/06_tools/

# 负例二滤
02_脚本/tools/filter_exemplar_neg.py \
  --pool $BATCH/06_tools/*_exemplar_keep_*.csv \
  --pos-bank data/assets/exemplars/yt_live_scene/ \
  --neg-bank data/assets/exemplars/yt_live_scene_neg_human_f/ \
  -o $BATCH/06_tools/
```

缩略图缓存统一用 `qc_thumb_cache/exemplar_sim/`（与 cascade CLIP 共用）。

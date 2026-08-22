# clean_DATASET

YouTube 视频元数据清洗管道（Python 3.13+ / DuckDB）。  
生产入口：`02_脚本/`。环境：`conda activate data_cleaning`。

详细 SOP、双路径（人工采 / 机采）、品类策略见 [AGENTS.md](AGENTS.md)。

## 本仓库重点：两路编码

清洗中后期常用 **文本语义（MiniLM）** + **缩略图视觉（CLIP）** 做 certain-noise / 场景过滤。完整说明见：

- **[docs/ENCODING_CLIP_MINILM.md](docs/ENCODING_CLIP_MINILM.md)** — 编码骨干、落盘约定、命令模板
- **[data/assets/CLIP_ASSETS.md](data/assets/CLIP_ASSETS.md)** — CLIP 资产复用清单（store / exemplar / cascade）

```text
title/channel ──► MiniLM（句向量 / 文本 clf）──► 高置信 drop
video_id 缩略图 ──► CLIP ViT-B-32 ──► embeddings.npy 或零样本 margin
```

| 层 | 骨干 | 入口 |
|----|------|------|
| 文本 | `paraphrase-multilingual-MiniLM-L12-v2` | `experiments/*_text_classifier.py` 训 → `02_脚本/tools/apply_small_model.py` 打分 |
| 视觉 | open_clip `ViT-B-32` / `openai` | `ClipEncoder`（`02_脚本/core/exemplar_sim.py`）→ cascade / `export_clip_embeddings.py` |

**大体积向量（`*.npy`）、`data/runs/` 批次表、`models/*.pkl` 默认不入库**；本地按文档导出即可。

## 快速开始

```bash
conda activate data_cleaning
cd /path/to/clean_DATASET
export PYTHONPATH=02_脚本

# 1) MiniLM 小模型打分（需本地 models/*.pkl）
02_脚本/tools/apply_small_model.py keep.csv -o $BATCH/06_tools/minilm/ \
  --model models/<category>_text_clf_f.pkl \
  --text-fields title,channel

# 2) 缩略图 → CLIP embedding store（可续跑）
02_脚本/tools/export_clip_embeddings.py candidates.csv \
  -o data/assets/embeddings/<store_name>/ \
  --model ViT-B-32 --pretrained openai \
  --cache-dir qc_thumb_cache/exemplar_sim \
  --batch-rows 5000 --encode-batch 128 --thumb-workers 24

# 3) 品类零样本 CLIP（例：农业；可顺带落 store）
02_脚本/tools/run_exo_agriculture_cascade_clip.py candidates.csv \
  -o $BATCH/06_tools/ \
  --sample 0 --batch-rows 5000 \
  --save-embeddings $BATCH/06_tools/embeddings/
```

## 目录速览

```text
02_脚本/          生产管道（pipeline / qc / tools / categories）
experiments/      MiniLM 文本分类器、CLIP+LR 等实验训模
data/assets/      CLIP 资产说明与 exemplar README（向量本地生成）
docs/             专题说明（编码等）
tests/            单测
AGENTS.md         完整仓库指南与 SOP
```

## 环境与密钥

- Python 3.13+，conda env `data_cleaning`
- 文本 LLM QC：`DASHSCOPE_API_KEY`
- YouTube 清晰度（可选）：`YOUTUBE_API_KEY`
- 勿提交 `.env`、cookies、原始大批 CSV

## 测试

```bash
conda activate data_cleaning
python -m pytest tests/test_exo_agriculture_cascade_clip.py tests/test_exo_fitness_cascade_clip.py -q
```

## License / 远程

GitHub：`git@github.com:Jqsfree/cleaning.git`（默认分支 `main`）。

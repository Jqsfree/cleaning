# embeddings — CLIP 缩略图向量 store

Shared 图像 embedding（float16，`ViT-B-32` / `openai`）。完整清单与复用规则见 [`../CLIP_ASSETS.md`](../CLIP_ASSETS.md)。  
编码命令总览见仓库 [`docs/ENCODING_CLIP_MINILM.md`](../../docs/ENCODING_CLIP_MINILM.md)。

**Git：** 只跟踪本 README（及可选 `meta.json` 说明）；`embeddings.npy` / `thumb_ok.npy` 体积大，已在 `.gitignore`，需本地用 `export_clip_embeddings.py` 或 cascade `--save-embeddings` 生成。

## Shared stores（本地生成后目录名）

| 目录 | Shape（约） | 用途 |
|------|-------------|------|
| `exo_service_hair_human/` | 261×512 | 理发 L3 人工标校准/检索 |
| `exo_agriculture_0817_semantic_remain/` | 63,160×512 | 农业 0817 semantic_remain |
| `exo_agriculture_0814_semantic_remain/` | 1,294,085×512 | 农业 0814 semantic_remain |

约定文件：`embeddings.npy`、`index.csv`、`meta.json`、`thumb_ok.npy`、`progress.json`。

## 导出 / 续跑

```bash
02_脚本/tools/export_clip_embeddings.py input.csv \
  -o data/assets/embeddings/<store_name>/ \
  --model ViT-B-32 --pretrained openai \
  --cache-dir qc_thumb_cache/exemplar_sim \
  --batch-rows 5000 --encode-batch 128 --thumb-workers 24
```

同一输入 id 集合续跑：勿加 `--overwrite`。

Cascade 同步落盘：

```bash
02_脚本/tools/run_exo_agriculture_cascade_clip.py input.csv \
  -o $BATCH/06_tools/ --save-embeddings $BATCH/06_tools/embeddings/ ...
```

改 prompt 免重编码：

```bash
02_脚本/tools/score_exo_agriculture_from_embeddings.py \
  --embeddings data/assets/embeddings/exo_agriculture_0817_semantic_remain/ \
  --config 02_脚本/categories/exo_agriculture/rules/cascade_harvest_clip.toml \
  -o $BATCH/06_tools/rescored_from_emb.csv
```

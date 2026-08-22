# CLIP 与 MiniLM 编码说明

本仓清洗在规则/文本 QC 之后，常再接两层语义过滤：

1. **MiniLM（文本）**：标题/频道句向量 → 小模型或原型对比 → 只丢高置信噪声  
2. **CLIP（视觉）**：YouTube 缩略图 → ViT-B-32 图像向量 → 零样本 margin 或线性探针  

二者正交：MiniLM 管「文案像不像目标品类」；CLIP 管「画面像不像目标场景」。

---

## 1. CLIP（缩略图视觉）

### 骨干

| 项 | 值 |
|----|-----|
| 实现 | `02_脚本/core/exemplar_sim.py` → `ClipEncoder` |
| 模型 | open_clip **ViT-B-32** |
| 权重 | **openai** pretrained |
| 输出维 | 512（L2 归一化；store 存 float16） |
| 缩略图 | `qc_thumb_cache/exemplar_sim/{video_id}.jpg`（hqdefault） |

### Store 落盘（可续跑）

```text
<data/assets/embeddings/<name>/ 或 $BATCH/06_tools/embeddings/>
  embeddings.npy   # float16 (N, 512)
  index.csv        # row, video_id
  thumb_ok.npy     # bool (N,)
  progress.json    # done / ids_sha256
  meta.json        # model / rows / dim
```

同输入 id 集合续跑：**不要**加 `--overwrite`。

### 导出编码

```bash
conda activate data_cleaning
export PYTHONPATH=02_脚本

02_脚本/tools/export_clip_embeddings.py input.csv \
  -o data/assets/embeddings/<store_name>/ \
  --model ViT-B-32 --pretrained openai \
  --cache-dir qc_thumb_cache/exemplar_sim \
  --batch-rows 5000 --encode-batch 128 --thumb-workers 24
```

### 零样本 cascade（边打分边可落 store）

品类 CLI 共用 `categories/exo_agriculture/cascade_clip.py` 的 `run_harvest_clip`（农业 / 畜牧 / 健身 / 户外等）：

```bash
02_脚本/tools/run_exo_agriculture_cascade_clip.py remain.csv \
  -o $BATCH/06_tools/ \
  --sample 0 --batch-rows 5000 --thumb-workers 24 --batch-size 128 \
  --config 02_脚本/categories/exo_agriculture/rules/cascade_harvest_clip.toml \
  --save-embeddings $BATCH/06_tools/embeddings/
```

决策：`margin = cos(img, mean(pos)) − cos(img, mean(neg))`；`require` 键全部过阈值才 `clip_pass`；`no_thumb` 不自动当 drop。

### 改 prompt 免重编码

有 store 后只重算文本原型 + matmul：

```bash
02_脚本/tools/score_exo_agriculture_from_embeddings.py \
  --embeddings data/assets/embeddings/exo_agriculture_0817_semantic_remain/ \
  --config 02_脚本/categories/exo_agriculture/rules/cascade_harvest_clip.toml \
  -o $BATCH/06_tools/rescored_from_emb.csv
```

仅改阈值、已有 `clip_q*`：可用 outdoor 的 reapply 工具同类脚本。

### 资产文档

见 [data/assets/CLIP_ASSETS.md](../data/assets/CLIP_ASSETS.md)、[data/assets/embeddings/README.md](../data/assets/embeddings/README.md)。

**注意：** `embeddings.npy` 体积大（百万行约 GB 级），**不进 Git**；只提交 README / 可选 `meta.json` 样例说明。

---

## 2. MiniLM（标题文本）

### 骨干

| 项 | 值 |
|----|-----|
| 模型 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 训模脚本 | `experiments/*_text_classifier.py`（按品类） |
| 生产打分 | `02_脚本/tools/apply_small_model.py` |
| 权重落盘 | `models/<category>_text_clf_*.pkl`（**不进 Git**） |

常见特征字段：`title` + `channel`（或 `title,keyword`，与训模对齐）。

### 训模（实验）

```bash
conda activate data_cleaning
PYTHONPATH=02_脚本:experiments python experiments/exo_agriculture_text_classifier.py --train \
  --labels data/runs/.../03_qc/.../human_qc.csv
# 写出 models/exo_agriculture_text_clf_f.pkl 等
```

农业等品类可能是「原型对比 / 否决器」而非朴素 LR，以各脚本 docstring 为准。

### 生产打分

```bash
02_脚本/tools/apply_small_model.py keep.csv \
  -o $BATCH/06_tools/minilm/ \
  --model models/<category>_text_clf_f.pkl \
  --text-fields title,channel \
  --drop-threshold 0.005
```

契约：**只自动 drop 高置信负例**；uncertain 不扔；`ml_score` ≠ 交付 KPI。

### 与 CLIP 的典型顺序

| 品类示例 | 顺序 |
|----------|------|
| 农业 0814/0817 | MiniLM semantic → CLIP |
| 农业 0818 计划 | CLIP（clean keep）→ MiniLM |
| unbox recipe | MiniLM → CLIP → vision_thumb |
| parent_child | MiniLM pass → 缩略图 CLIP+LR |

以各品类 `categories/*/recipe.toml` 为准。

---

## 3. 依赖（编码相关）

- `open_clip_torch` / `torch` / `Pillow`（CLIP）
- `sentence-transformers` / `scikit-learn`（MiniLM clf）
- `numpy` / `pandas` / `duckdb`

环境名：`data_cleaning`。

---

## 4. 相关测试

```bash
python -m pytest tests/test_exo_agriculture_cascade_clip.py \
  tests/test_exo_fitness_cascade_clip.py -q
```

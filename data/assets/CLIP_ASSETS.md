# CLIP 编码资产复用清单

全仓视觉骨干统一为 **open_clip `ViT-B-32` + `openai` pretrained**（`02_脚本/core/exemplar_sim.py` → `ClipEncoder`）。  
命令与 MiniLM 对照见仓库根目录 [`docs/ENCODING_CLIP_MINILM.md`](../../docs/ENCODING_CLIP_MINILM.md)。  
同一**模型指纹**下，可复用层级从高到低：

1. 缩略图 JPEG 缓存（跨品类、跨批，`video_id` 相同即可）
2. 图像 embedding store（同模型 + 同预处理，可反复打分/校准）
3. 样例原型 bank（域内正/负例）
4. 已写好的 margin 分数 CSV（只改阈值可重判，无需重编码）
5. 零样本文本 prompt TOML（文本编码便宜，可随时重算）

农业 / 健身 / 户外 cascade CLIP **早期只落 `clip_*.csv` 分数**；农业 0814/0817 已回填 `embeddings.npy`，且 `run_exo_agriculture_cascade_clip.py` 支持 `--save-embeddings`。改 prompt：用 store 重打分；改阈值：reapply。

```text
thumbs JPEG ──► embeddings.npy ──► Tip-Adapter 式 cache 决策
exemplar prototypes.npy ─────────► 原型匹配
cascade_*_clip.toml ─────────────► CLIP zero-shot / pos−neg margin
clip_q* 分数 CSV ─────────────────► 仅改阈值重判
```

---

## 论文映射

| 范式 | 代表工作 | 本仓对应 |
|------|----------|----------|
| 多模态零样本 | Radford et al., *CLIP* (ICML 2021) | `ClipEncoder`；`cascade_*_clip.toml` 正负 prompt |
| 缓存特征再决策 | Tip-Adapter (Zhang et al., ECCV 2022) | `export_clip_embeddings.py` + store；`sample_active_learning.py` |
| 原型 / few-shot 匹配 | Prototypical Networks (Snell et al., NeurIPS 2017) | `build_exemplar_bank.py` → `prototypes.npy` → `score_exemplar_sim.py` |
| 正负描述对比 | Visual Classification via Description 等；工程上 pos−neg score | `margin = cos(img, mean(pos)) − cos(img, mean(neg))` |

决策公式（cascade）：

```text
margin(q) = cos(img, mean(pos_q)) − cos(img, mean(neg_q))
clip_pass  ⟺  所有 require 键的 margin ≥ 阈值
no_thumb   ⟺  无缩略图（保留交下游 VL，不当 drop）
```

---

## A. 共享基础设施（几乎总可复用）

| 资产 | 路径 | 说明 |
|------|------|------|
| 编码器 | `ClipEncoder` / open_clip ViT-B-32 | cascade / exemplar / parent_child thumb 共用 |
| 缩略图缓存 | `qc_thumb_cache/exemplar_sim/` | 按 `video_id` 命中跳过下载；约数百万 jpg |
| 导出工具 | `02_脚本/tools/export_clip_embeddings.py` | 标准 float16 store |

**复用条件：** 不换 `model` / `pretrained` / 预处理；缩略图仍为 YouTube `hqdefault` 口径（`exemplar_sim.THUMB_URL`）。

**store 落盘约定：**

```text
<store>/
  embeddings.npy   # float16, (N, 512)
  index.csv        # row, video_id
  thumb_ok.npy     # bool, (N,)
  progress.json    # 续跑断点 + ids_sha256
  meta.json        # model / pretrained / rows / dim
```

---

## B. Embedding store（向量可复用；域绑定）

### Shared（`data/assets/embeddings/`）

| Store | Shape | 用途 | 复用范围 |
|-------|-------|------|----------|
| `exo_service_hair_human/` | 261×512 f16 | 理发 L3 人工标 | 仅 `exo_service` hair 校准/检索 |
| `exo_agriculture_0817_semantic_remain/` | 63,160×512 f16 | 0817 semantic_remain 缩略图向量 | 农业改 prompt 重打分 / 主动学习 |
| `exo_agriculture_0814_semantic_remain/` | 1,294,085×512 f16 | 0814 semantic_remain 缩略图向量 | 同上 |

### Runs 内可复用、未晋升 shared

路径前缀：`data/runs/live_sell/machine_0805/06_tools/visual_filter/`

| Store | Shape | 用途 |
|-------|-------|------|
| `labels_embeddings/` | 262×512 | 标签集 v1 |
| `labels_embeddings_v2/` | 543×512 | 标签集 v2 |
| `labels_embeddings_v3/` | 1079×512 | 标签集 v3 |
| `pool_embeddings/` | **300,271×512** | 0805 候选池；池内重阈值 / 主动学习最值钱 |
| `smoke1000/embeddings/` | 1000×512 | 冒烟测试 |

不默认把 `pool_embeddings` 迁入 `data/assets/`（体积大）；需要跨会话复用时在本清单引用即可。

---

## C. Exemplar / prototype bank（域内复用）

见 [`exemplars/README.md`](exemplars/README.md)。

| Bank | Shape | 来源 | 工具链 |
|------|-------|------|--------|
| `exemplars/yt_live_scene/` | 10×512 + 12 帧/样例 | 本地「直播场景」mp4 | `build_exemplar_bank` → `score_exemplar_sim` |
| `exemplars/yt_live_scene_neg_human_f/` | 224×512 | 人工 F 缩略图负例 | `filter_exemplar_neg` margin 二滤 |

**不可混用：** 直播场景 ≠ 带货场景 ≠ 农业/户外。带货须另建 bank。  
农业 / 户外尚无样例 mp4 bank（Phase2 待样例目录）。

---

## D. 零样本 prompt + margin（配置可复用）

品类 TOML（运行时 `encode_text`，**不**持久化 text embedding）：

| 品类 | 配置 |
|------|------|
| exo_agriculture | `02_脚本/categories/exo_agriculture/rules/cascade_harvest_clip.toml` |
| exo_livestock | `…/exo_livestock/rules/cascade_livestock_clip.toml` |
| exo_fitness | `…/exo_fitness/rules/cascade_fitness_clip.toml` |
| exo_outdoor | `cascade_outdoor_clip.toml` 及 person / l1_strict / negative 变体 |
| exo_service | `cascade_l3_hair_person.toml` 等 |

**已有分数、可免重编码重判：**

| 场景 | 做法 |
|------|------|
| outdoor | `02_脚本/tools/run_exo_outdoor_reapply_l1_clip.py` |
| agriculture 等 | `*_clip.ckpt.csv` / `*_clip_scored_*.csv` 含 `clip_q*` — **改阈值可重判；改 prompt 必须重跑** |

---

## E. 非 CLIP 向量（勿混谈）

| 资产 | 说明 |
|------|------|
| `data/assets/rejects/` | 文本/画面 reject 注册表 + CSV，**不是** CLIP embedding；可作负例种子去**重建** thumb 原型 |
| MiniLM 等文本 clf（`models/*_text_clf_*.pkl`） | 文本语义层，与 CLIP 图像向量正交 |

---

## 复用规则

1. **指纹一致才复用向量：** `model=ViT-B-32` + `pretrained=openai` + 缩略图源；换模型或换 maxres/hq 一律作废。
2. **JPEG 缓存最通用：** 任何品类 CLIP / exemplar 优先共用 `qc_thumb_cache/exemplar_sim`。
3. **原型 bank 按域：** live ≠ 带货 ≠ 农业；负例 bank 只服务同域 margin 二滤。
4. **分数 CSV vs embedding：** 只动阈值 → reapply；动 prompt / 模型 → 需要 embedding 或重编码。
5. **跨批 `video_id` 交集：** 可按 id join 复用已有向量；新批未编码 id 仍需补编码。

---

## 主要缺口

| 缺口 | 影响 | 状态 |
|------|------|------|
| 农业 cascade **未写** `embeddings.npy` | 收紧 prompt 后全量重跑成本高 | **0814/0817 已回填 store**；CLI 已加 `--save-embeddings`（0818 起建议打开） |
| 农业无 exemplar bank | 缺样例 mp4 | 仍缺 |
| `data/assets/embeddings/` 正式资产很少 | live_sell 大池仍在 runs 下 | 农业两批已晋升 shared |
| 无机器可读 registry | 靠本清单人工维护 | 仍缺 |

### 农业：从 store 改 prompt 重打分

```bash
02_脚本/tools/score_exo_agriculture_from_embeddings.py \
  --embeddings data/assets/embeddings/exo_agriculture_0817_semantic_remain/ \
  --config 02_脚本/categories/exo_agriculture/rules/cascade_harvest_clip.toml \
  --merge-csv data/runs/exo_agriculture/machine_0817/06_tools/农业采集_0817_semantic_remain_clip_scored_0817.csv \
  -o data/runs/exo_agriculture/machine_0817/06_tools/农业采集_0817_clip_rescored_from_emb.csv
```

### 0818 CLIP 建议带上落盘

```bash
02_脚本/tools/run_exo_agriculture_cascade_clip.py "$KEEP" \
  -o $BATCH/06_tools/ \
  --sample 0 --batch-rows 10000 --thumb-workers 32 --batch-size 128 \
  --stem 农业采集_0818_clean \
  --save-embeddings $BATCH/06_tools/embeddings/
```

---

## Cascade 后落 embeddings（推荐规范）

**优先：** 跑 cascade 时直接加 `--save-embeddings DIR`（与 `export_clip_embeddings` 同结构，续跑对齐 checkpoint）。

**补跑：** 对已 CLIP 过的 CSV（只需 `video_id`）用导出工具回填：

```bash
conda activate data_cleaning
cd /home/jqs/projects/clean_DATASET
export PYTHONPATH=02_脚本

# 例：农业 0814 scored（缩略图多已在 qc_thumb_cache）
IN=data/runs/exo_agriculture/machine_0814/06_tools/农业采集_0814_semantic_remain_clip_scored_0818.csv
OUT=data/assets/embeddings/exo_agriculture_0814_semantic_remain

02_脚本/tools/export_clip_embeddings.py "$IN" -o "$OUT" \
  --model ViT-B-32 --pretrained openai \
  --cache-dir qc_thumb_cache/exemplar_sim \
  --batch-rows 5000 --encode-batch 128 --thumb-workers 24
```

续跑：同一 `-o`、同一输入 id 集合，**不要**加 `--overwrite`。

改 prompt 后免重编码：

```bash
02_脚本/tools/score_exo_agriculture_from_embeddings.py \
  --embeddings data/assets/embeddings/exo_agriculture_0814_semantic_remain/ \
  --config 02_脚本/categories/exo_agriculture/rules/cascade_harvest_clip.toml \
  -o $BATCH/06_tools/rescored_from_emb.csv
```

仅改阈值、已有 `clip_q*` 列时：

```bash
# outdoor 示例
02_脚本/tools/run_exo_outdoor_reapply_l1_clip.py \
  $BATCH/06_tools/*_clip_scored_*.csv \
  --config 02_脚本/categories/exo_outdoor/rules/cascade_outdoor_clip.toml \
  -o $BATCH/06_tools/
```

---

## 相关工具速查

| 工具 | 作用 |
|------|------|
| `tools/export_clip_embeddings.py` | 候选缩略图 → embedding store |
| `tools/score_exo_agriculture_from_embeddings.py` | store + TOML → 重打 clip margin（免重编码） |
| `tools/build_exemplar_bank.py` | 样例 mp4 → `prototypes.npy` |
| `tools/score_exemplar_sim.py` | 候选 vs 正例 bank |
| `tools/filter_exemplar_neg.py` | 负例 bank margin 二滤 |
| `tools/run_exo_*_cascade_clip.py` | 品类零样本 cascade（`--save-embeddings`） |
| `tools/run_exo_outdoor_reapply_l1_clip.py` | 已有分数重判 |
| `tools/sample_active_learning.py` | 读 embedding store 做主动学习抽样 |

索引：[`embeddings/README.md`](embeddings/README.md)、[`exemplars/README.md`](exemplars/README.md)。

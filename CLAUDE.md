# CLAUDE.md

YouTube 视频元数据清洗管道，**exo 家族独立仓库**。

## 背景

- 从 `~/projects/clean_DATASET` 提取 exo 家族（2026-08-23），只做 8 个 exo 品类：`exo`、`exo_agriculture`、`exo_factory`、`exo_fitness`、`exo_livestock`、`exo_medical`、`exo_outdoor`、`exo_service`。其他品类（film_tv、live_sell 等）留在原仓库。
- 环境：**`conda activate data_cleaning`**（Python 3.13+，DuckDB）。

## 目录

```
02_脚本/                  生产管道
├── pipeline/             01_quality / 02_clean / 03_sample / run / orchestrate
├── qc/                   text / vision_thumb
├── tools/                exo 专用工具（run_*_cascade_clip 等）
├── categories/exo*/      品类插件（recipe.toml + cleaner + rules）
└── core/                 共享库
data/runs/{exo*}/{source}_{batch}/   批次产物
raw/{exo*}/               Bronze（只读）
models/exo*               模型权重
data/assets/embeddings/exo_*   CLIP 语义嵌入缓存
```

## 数据现状

`raw/`、`data/runs/`（10 个 exo 批次）、`models/`、`data/assets/embeddings/exo_*` 已全量从原仓库迁移。

## 质量治理原则（对齐人工标准）

- **人工标注是唯一可信的 pass_rate 来源**（`tools/ingest_human_qc.py` 入库 `03_qc/{labeled,pass,fail,train_export}.csv`）。交付/验收只认人工合格率，勿用 keep% / LLM-QC% / ml_score 冒充质量指标。
- **自动机检（文本 QC / CLIP / 小模型）= certain-noise only**：只自动丢「确定噪声」；`U` / 中间带 → 交人工或小模型，勿直接当 drop。
- **overturn 验证筛检器**：机检 drop 的「人工 T 误杀率（t_hurt）」与「F 召回（f_recall）」是校准机检阈值的依据；`apply_*` 支持 `--eval-labels` + `--halt-on-high-t-hurt`，超 calibration 上限即中止。
- **禁止无验证自训**：小模型/阈值校准必须以人工 gold 为准（`eval_exo_agriculture_clip.py` 网格校准 → 写 `calibration.json`）。
- 农业（exo_agriculture）是唯一已建立「人工标注 → 对齐量化」闭环的品类，方法论见 `docs/农业机检对齐人工标准评估.md`。

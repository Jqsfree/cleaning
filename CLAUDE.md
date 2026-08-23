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

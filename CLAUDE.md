# CLAUDE.md

@AGENTS.md

This file provides Claude Code-specific guidance for this repository.

## Dual stack（生产 vs WIP）

| 路径 | 角色 |
|------|------|
| `02_脚本/` | **生产**：`pipeline/` / `qc/` / `tools/` |
| `src/dataclean/` | 包化 WIP；勿当主入口 |

## 环境

**`conda activate data_cleaning`**（主用）。

## 双路径（勿默认 quality→clean）

- **人工采** `human`：quality → sample → 人工质检 → 合格交付；不合格才 `02_clean`
- **机采** `machine`：quality → sample → 文本 QC → 改 TOML → **再** `02_clean`
- 清晰度：`tools/fetch_yt_definition.py`（工具，交付前可选）
- 目录：`data/runs/{category}/{source}_{batch}/{01_quality…07_deliver}/`；禁止 `data/runs/` 根放数

详见 AGENTS.md。

## 分品类

各品类数据与主链不同，见 AGENTS 品类策略表。`core/sop.py` **只写 run_log**。

**有 cleaner：** `language_teaching`（+scorer）/ `beauty` / `welding` / `film_tv`。

**仅 QC：** `ego_repair` / `lila_outdoor` — 勿 `02_clean --category`。

```
categories/<name>/
  cleaner.py       # 若有
  scorer.py        # 仅 language_teaching 等需要
  rules/
    blacklist.toml
    whitelist.toml
    qc.toml / vision_thumb.toml / …  # 按需
```

## Cleaner 形态

- **language_teaching**：SQL 黑名单 → UDF 评分 → high/medium/drop。
- **beauty / film_tv / welding**：主要为 SQL 黑名单 keep/drop。

## film_tv

- 初筛：`pipeline/01_quality.py`
- 清洗：`pipeline/02_clean.py --category film_tv --source …`（门禁见 AGENTS；输出 **CSV**）
- 画面：`qc/vision_thumb.py -c film_tv`
- ≥720：`tools/fetch_yt_definition.py … --filter-ge720`
- 新跑落盘：`data/runs/film_tv/{human|machine}_{batch}/`
- **闭环**：人工 `ingest_human_qc` → 规则/小模型只丢确定噪声 → `sample_drop_for_reqc` 回流；详见 AGENTS「正反馈闭环」
- **排除类资产**：`reject_registry` 可演进；`accumulate_reject_assets` / `cascade_reject_propose` 双模态累计；`reject_source_metrics` + `suggest_reject_opt`（建议闸门，禁无验证自训）。种子批：`data/runs/film_tv/_reject_seed_0727/` → `data/assets/rejects/`

## Data Flow

```
raw/{category}/*.csv
  → 01_quality
  → （人工｜机采分流，见 AGENTS）
  → 可选 06_tools definition/resolution / 小模型
  → 07_deliver
```

不做：`05_evaluate`、`07_merge`、`monitor.py`。无 GUI。

## Adding a New Category

**完整清洗插件：** `cleaner.py` + `rules/blacklist.toml`；注册到 `pipeline/02_clean.py` `_CLEANERS`。  
**仅 QC：** `rules/qc.toml`（+ 可选 vision toml）。

## Workflow (EXEC / ASK)

### 直接执行

- 数据查看：`ls` / `find` / `head` / `grep` / `wc`、Read/Grep/Glob
- 「检查/看下/对比/对齐」→ 直接探索、统计、出表

### 需确认或给命令

- `conda activate` / `pip install`
- 修改 `rules/` TOML
- 删除数据、`git commit`、管道改动数据时
- 交付合并

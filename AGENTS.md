# Repository Guidelines

## Project Overview

YouTube 视频元数据清洗管道。目标 Python 3.13+，主引擎 DuckDB。

**生产入口：`02_脚本/`。** `src/dataclean/` 是包化 WIP，**日常勿当主路径**。

**环境：`conda activate data_cleaning`**（主用；勿依赖文档里已过时的「仅 .venv」说法）。

**没有跨品类统一 Phase0–7**：流程由 **采集来源（人工采 / 机采）+ 品类** 决定，不是全局 `quality→clean→sample`。

LLM / vision：DashScope（`DASHSCOPE_API_KEY`）。清晰度：YouTube Data API（`YOUTUBE_API_KEY`）或 yt-dlp。

---

## 双路径（默认 SOP）

```text
raw → 01_quality（基础质量）
        ├─ 人工采 → 02_sample → 人工质检
        │              ├─ 合格 → 07_deliver（可选 06_tools 如 ge720）
        │              └─ 不合格 → 05_clean → 07_deliver
        └─ 机采   → 02_sample → 03_qc 文本质检 → 改/确认 TOML 规则
                       → 05_clean（有规则依据后）→ 07_deliver（可选 tools）
```

| 来源 | quality 后 | `02_clean` 时机 | 交付 |
|------|------------|-----------------|------|
| **人工采** `human` | 抽样 → **人工质检** | **仅不合格**进入清洗 | 合格可对 quality 全集（或约定基表）交付 |
| **机采** `machine` | 抽样 → **文本质检 → 规则** | **有规则依据后**再 clean | 按品类交付要求 |

**禁止：** 无抽样质检依据就对机采直接 `02_clean`；对人工合格集默认跑 clean。  
**薄编排** [`pipeline/run.py`](02_脚本/pipeline/run.py) 必须带 `--source`；默认 **只跑 quality**，不自动 clean。

清晰度（`fetch_yt_definition` / `fetch_resolution`）是 **横切工具**，挂在交付前按需调用，不是 clean 前置阶段。

### 品类策略（交付差异）

| Category | cleaner | 默认 QC | 需要 ge720 | 交付格式 | 典型路径 / 备注 |
|----------|---------|---------|------------|----------|-----------------|
| `language_teaching` | 有 + scorer | 文本 | 否 | parquet | 机采：sample→text QC→规则→clean |
| `beauty` | 黑名单 | 文本 | 否 | parquet | whitelist 未接入 |
| `welding` | 轻量 | storyboard | 否 | parquet | quality→（规则）→storyboard |
| `film_tv` | 黑名单 | thumb / text | **是** | **CSV** | 人工/机采分流；交付前挂 definition |
| `ego_repair` / `lila_outdoor` | 无 | thumb | 否 | 按 QC 产出 | 勿 `02_clean --category` |
| `tow-person` | 无 | two_person | 否 | 按 QC 产出 | 仅画面 QC |

机采 `sd` / 抽检子集才考虑慢路径 `fetch_resolution`；**不要**默认全量 yt-dlp。

### film_tv 文本 vs 画面

- **文本** `categories/film_tv/rules/qc.toml`：仅对**确定**目标打 `t`、**确定**噪声打 `f`；无法确认必须 `u`（勿当 f）。
- **画面** `vision_thumb.toml`：缩略图 T/F。默认 `qc/vision_thumb.py -c film_tv`。
- **KPI**：交付只认**人工合格率**（`ingest_human_qc`）；勿用 keep% / LLM-QC% / ml_score 当交付指标。
- 自动文本 QC / 黑名单 = **certain-noise only**；`u` / 中间带 → 人工或小模型，勿直接当 drop。

---

## 正反馈闭环（人工锚定）

```text
人工 QC → ingest → 训小模型 / 收紧规则
  → 自动只丢「确定噪声」→ drop 抽样回流再 QC
```

```bash
# 人工入库（唯一有意义的 pass rate）
02_脚本/tools/ingest_human_qc.py labels.csv -o $BATCH/ \
  --category film_tv --source human --batch 0724

# drop 回流抽样 → 人工复检 → 再 ingest
02_脚本/tools/sample_drop_for_reqc.py $BATCH/05_clean/run01/drop.csv -o $BATCH/ --n 200

# 小模型（仅高置信负例可 drop；uncertain 交人工）
02_脚本/tools/apply_small_model.py input.csv -o $BATCH/06_tools/ \
  --model models/film_tv_text_clf_svm.pkl

# 找交付
02_脚本/tools/run_manifest.py list --runs-root data/runs
02_脚本/tools/run_manifest.py find-deliver --category film_tv
```

### 共享排除类资产（可演进 registry + 双模态）

正例因客户而异难复用；**不要的细类**跨批可积累。  
Registry（`categories/_shared/reject_registry.toml`）是**版本化命名空间**，非冻死枚举。  
同一 `reject_tag`，用 **`modality=text|thumb|fusion`** 区分通道（勿平行造 `*_thumb` id）。

- **人工 = 验证**（pass/fail + 可选抽样 confirm/correct），不是全量打细类  
- **累计 = 批次后一条命令**（非默认 pipeline、非常驻静默）  
- **自动优化 = 度量 + 建议 + 人工闸门**；禁止无验证自训（确认偏误）  
- 来源准确度靠 **overturn**（`reject_source_metrics`），不信 raw score  
- **资产落盘**：`data/assets/rejects/{tag}/`；度量：`…/_metrics/by_source.json`  
- **初版种子**（2026-07-27）：`data/runs/film_tv/_reject_seed_0727/`（存量 drop+thumb → proposed；尚无 human_validated）  
  复跑：`PYTHONPATH=02_脚本 python scripts/_seed_reject_assets_once.py`

```bash
# 一条命令：text drop + thumb_qc → proposed → export
02_脚本/tools/accumulate_reject_assets.py -o $BATCH/ --category film_tv \
  --text-input $BATCH/05_clean/run01/drop.csv \
  --thumb-input $BATCH/03_qc/xxx_thumb_qc.csv
# 旧路径 005_clean 需显式 --text-input / --thumb-input（不会自动扫 005_*）

# 级联：高把握提案；冲突/中间带 → 03_qc/reject_sample_for_validate.csv
02_脚本/tools/cascade_reject_propose.py -o $BATCH/ --category film_tv
# 或：accumulate … --cascade

# 来源准确度账本
02_脚本/tools/reject_source_metrics.py --assets-root data/assets/rejects

# 优化建议（默认不改配置；--apply --i-understand 只写 shadow 副本）
02_脚本/tools/suggest_reject_opt.py --assets-root data/assets/rejects
```

配置：`categories/_shared/reject_cascade.toml`、`reject_modality_map.toml`、`reject_registry.toml`。  
训练：**human_validated 优先**；proposed 仅弱监督。交付 KPI 仍只认人工合格率。

---

## 命名与目录（快速定位）

### 路径骨架（新跑强制）

```text
data/runs/{category}/{source}_{batch}/
  01_quality/
  02_sample/
  03_qc/                 # 人工结果表或 LLM QC
  04_rules/              # 机采：本轮规则说明（可链到 categories/*/rules）
  05_clean/runNN/        # 仅需清洗时
  06_tools/              # yt_definition / resolution 等
  07_deliver/            # 对外唯一交付口
  manifest.json          # 批次索引（推荐）
```

- `{category}`：英文品类，如 `film_tv`
- `{source}`：`human` | `machine`（全库统一英文）
- `{batch}`：批号，如 `0724`、`723`

例：`data/runs/film_tv/human_0724/` — **不要**再在 `data/runs/影视剧人工724/` 与 `data/runs/film_tv/` 两套并行新开。

**禁止**在 `data/runs/` **根目录**放 CSV/报告；存量中文根目录只读兼容，新交付进 `07_deliver/`。

### 文件名契约

```text
{batch}_{stage}[_{variant}]_{MMDD}.csv
```

例：`0724_quality.csv`、`0724_yt_definition.csv`、`0724_deliver_ge720.csv`、`0724_clean_run01_keep.csv`。

### 批次索引

```bash
02_脚本/tools/run_manifest.py init -o data/runs/film_tv/human_0724/ \
  --category film_tv --source human --batch 0724 --input raw/.../xxx.csv
02_脚本/tools/run_manifest.py show -o data/runs/film_tv/human_0724/
```

`pipeline/run.py` 在批次根目录会自动维护 `manifest.json`。

### raw/

- 只读；`raw/{category}/`；文件名下划线、无空格括号。
- 管道产物不得写回 `raw/`。

### 历史对照（只读兼容）

详见 [`data/runs/film_tv/README_迁移对照.md`](data/runs/film_tv/README_迁移对照.md)。

| 旧路径 | 新约定 |
|--------|--------|
| `data/runs/影视剧人工724/` | `data/runs/film_tv/human_0724/`（根上可为 symlink） |
| `data/runs/影视剧人工724-2/` | `data/runs/film_tv/human_0724_2/` |
| `005_clean` / `002_audit` | `05_clean` / `02_sample`+`03_qc` |
| `data/runs/` 根上合并 CSV | `film_tv/07_deliver/` 或批次 `07_deliver/` |

---

## Project Structure

```
02_脚本/                   ★ 生产管道
├── pipeline/
│   ├── 01_quality.py        初筛（双路径共同入口）
│   ├── 02_clean.py          规则清洗（有门禁；见 --source）
│   ├── 03_sample.py         抽样（clean 之前）
│   ├── 04_analyze.py        污染分析（助改规则）
│   ├── 06_dedup.py          跨批次去重
│   └── run.py               按 --source 薄编排（默认仅 quality）
├── qc/                      text / vision_*
├── tools/                   fetch_yt_definition / batch_deliver_ge720 / run_manifest / …
├── categories/              品类插件
└── core/                    共享库（含 run_manifest）
experiments/  .archive/  src/dataclean/  raw/  data/runs/
```

**不做：** `05_evaluate` / `07_merge` / `monitor.py`。

---

## 开一批（可复制 SOP）

路径一律：`BATCH=data/runs/film_tv/{human|machine}_{批号}`。先 `conda activate data_cleaning`。

### 人工采 `human`

```bash
BATCH=data/runs/film_tv/human_0727
RAW=raw/film_tv/xxx.csv          # 改成实际输入
export YOUTUBE_API_KEY='...'     # 若本批要 ge720

# 1) 初筛 + manifest（默认不 clean）
02_脚本/pipeline/run.py "$RAW" --category film_tv --source human -o "$BATCH/"

# 2) 抽样 → 02_sample/
Q=$(ls -t "$BATCH"/01_quality/*quality*.csv 2>/dev/null | command grep -v drop | head -1)
02_脚本/pipeline/03_sample.py "$Q" -o "$BATCH/02_sample/" -n 385

# 3) 人工质检：结果表放入 03_qc/（外置流程）
mkdir -p "$BATCH/03_qc"

# 4) 仅不合格集才 clean（合格跳过本步）
# 02_脚本/pipeline/02_clean.py "$BATCH/03_qc/fail.csv" \
#   --category film_tv --source human --allow-clean \
#   -o "$BATCH/05_clean/run01/" -r run01

# 5) 交付前 ≥720（对 quality 全集或 clean keep）
KEEP="$Q"   # 或 "$BATCH/05_clean/run01/"*_clean*.csv（非 drop）
02_脚本/tools/batch_deliver_ge720.py "$KEEP" --batch-root "$BATCH" --batch-id 0727

# 6) 交付只从 07_deliver/ 取
ls "$BATCH/07_deliver/"
02_脚本/tools/run_manifest.py show -o "$BATCH/"
```

### 机采 `machine`

```bash
BATCH=data/runs/film_tv/machine_0727
RAW=raw/film_tv/xxx.csv
export YOUTUBE_API_KEY='...'

02_脚本/pipeline/run.py "$RAW" --category film_tv --source machine -o "$BATCH/"
Q=$(ls -t "$BATCH"/01_quality/*quality*.csv 2>/dev/null | command grep -v drop | head -1)
02_脚本/pipeline/03_sample.py "$Q" -o "$BATCH/02_sample/" -n 385

# 文本 QC → 改 categories/film_tv/rules/*.toml → 在 04_rules/ 留一句说明
S=$(ls -t "$BATCH"/02_sample/*sample*.csv 2>/dev/null | head -1)
02_脚本/qc/text.py "$S" --category film_tv -w 20 -o "$BATCH/03_qc/"
mkdir -p "$BATCH/04_rules"
echo "规则依据: 本轮 text QC $(date +%F)" > "$BATCH/04_rules/NOTES.md"

# 有规则后再 clean
02_脚本/pipeline/02_clean.py "$Q" \
  --category film_tv --source machine --rules-ready \
  -o "$BATCH/05_clean/run01/" -r run01

CK=$(ls -t "$BATCH"/05_clean/run01/*clean*.csv 2>/dev/null | command grep -v drop | head -1)
02_脚本/tools/batch_deliver_ge720.py "$CK" --batch-root "$BATCH" --batch-id 0727
```

`02_clean` **必须**带 `--source`；遗留脚本临时加 `--legacy`（将移除）。

## Key Commands（摘录）

```bash
# 一键：definition + 过滤 + 写入 07_deliver + 更新 manifest
02_脚本/tools/batch_deliver_ge720.py path/to/keep.csv \
  --batch-root data/runs/film_tv/human_0727 --batch-id 0727

# 仅过滤（已有 definition）
02_脚本/tools/fetch_yt_definition.py keep.csv --filter-only \
  --definition "$BATCH/06_tools/0727_yt_definition.csv" \
  -o "$BATCH/07_deliver/0727_deliver_ge720.csv"

# 精确 max_height（慢；可选，建议仅 sd 子集）
02_脚本/tools/fetch_resolution.py sd_subset.csv --bank-size 10000 -w 2 \
  --sleep-interval 1.0 -o "$BATCH/06_tools/res_banks/"
```

`fetch_yt_definition`：`hd`≈≥720，`sd`≈&lt;720；约 50 id/请求。  
`fetch_resolution`：cookies 场景须 deno；见 EJS wiki。

### QC 并发

| 脚本 | 建议 |
|------|------|
| `qc/text.py` | `-w 20` 起 |
| `qc/vision_storyboard.py` | `-w 2`，先 `--benchmark` |
| `qc/vision_thumb.py` | `-t 12` 起 |

### `-o` 约定

| 类型 | `-o` | 示例 |
|------|------|------|
| pipeline / 多数 QC | **目录** | `…/05_clean/run01/` |
| `fetch_yt_definition` / `06_dedup` 单文件 | **文件** | `…/06_tools/0724_yt_definition.csv` |
| `fetch_resolution --bank-size` | **bank 目录** | `…/06_tools/res_banks/` |
| `run.py` / manifest | **批次根目录** | `data/runs/film_tv/human_0724/` |

---

## Environment

- Python 3.13+，**`conda activate data_cleaning`**
- 常用包：`duckdb`, `pandas`, `openai`, `tqdm`, `requests`, `yt-dlp`
- `DASHSCOPE_API_KEY`、`YOUTUBE_API_KEY`（勿提交）
- 可选：`YT_DLP_COOKIES_FILE`；cookies 在 `.gitignore`
- 带 cookies 的 yt-dlp 需 **deno**（或 node）+ 必要时 `yt-dlp-ejs` / `--remote-components ejs:github`

## Coding Conventions

- Python 3.13+；`sys.path` 插入 `02_脚本/` 访问 `core/`、`categories/`
- 4 空格；argparse CLI；中文日志/注释，英文标识符与路径
- 规则：`categories/<name>/rules/*.toml`

## Testing

```bash
conda activate data_cleaning
python -m pytest tests/test_adaptive_api.py tests/test_fetch_resolution_height.py tests/test_fetch_yt_definition.py -q
```

管道阶段以小样本跑通 + 行数/留存对照为准。

## Commit

短中文说明阶段变更。勿提交 raw 大数据、密钥、cookies。

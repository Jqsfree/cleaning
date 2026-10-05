# CLAUDE.md

YouTube 视频元数据清洗管道，**exo 家族独立仓库**。

## 背景

- 从 `~/projects/clean_DATASET` 提取 exo 家族（2026-08-23），迁入 8 个品类：`exo`、`exo_agriculture`、`exo_factory`、`exo_fitness`、`exo_livestock`、`exo_medical`、`exo_outdoor`、`exo_service`；本仓后续新建 `exo_cook`（烹饪餐饮，0916）、`exo_parent`（亲子互动，0917）、`exo_团队协作`（乒乓球双打 + 网球双打，0922；子运动见 `categories/exo_团队协作/{pingpong,tennis}/`）、`exo_construction`（建筑施工，0923）。其他品类（film_tv、live_sell 等）留在原仓库。
- 环境：仓库自带 **`.venv`（Python 3.9.6）**，依赖见 `requirements.txt`（安装：`.venv/bin/python3 -m pip install -r requirements.txt`）。源码按 3.11+ 写法，故 `tomli` 提供 `tomllib` 兜底；`pandas` 的 `read_parquet`/`to_parquet` 依赖 `pyarrow`。

## 目录

```
02_脚本/                  生产管道
├── pipeline/             01_quality / 02_clean / 04_analyze / 06_dedup / run / orchestrate
├── qc/                   text / vision_thumb
├── tools/                exo 专用工具（score_*_text / apply_* / run_*_cascade_clip / ingest_human_qc）
│   ├── batch_ops/        跨品类批次运维（merge_csvs 合并 / sample_qc 抽样）
│   └── agri_v1/          农业 v1 工具套件
├── categories/exo*/      品类插件（recipe.toml + cleaner + rules/blacklist.toml）
└── core/                 共享库（certain_noise_clean / rules_loader / sql_builder / human_qc …）
experiments/              MiniLM 文本分类器训练脚本（{category}_text_classifier.py）
data/runs/{exo*}/{source}_{batch}/   批次产物
  └ 01_quality/ 02_sample/ 03_qc/ 05_clean/run0N_*/ 06_tools/ 07_deliver/ README.md
work/                     清洗中间产物（scratch，gitignore，随时可清空）
raw/{exo*}/               Bronze（只读）
models/exo*               模型权重 + `{name}_calibration.json`
data/assets/embeddings/exo_*   CLIP 语义嵌入缓存
/Users/muse/data/deliver/ 对外交付落地目录（仓库外）
```

脚本工程约定见 `02_脚本/tools/README.md`（纯函数 + `main()` 薄 CLI 分层，路径由 `_SCRIPT`/`_REPO` 推导，禁止硬编码绝对路径与解释器路径）。

## 现行清洗流程（文本阶段，全品类同款）

1. **Bronze 合并** `tools/batch_ops/merge_csvs.py`（DuckDB，`video_id` 去重）→ `raw/{category}/`
2. **初筛** `pipeline/01_quality.py` → `01_quality/`
3. **黑名单清洗** `categories/{category}/cleaner.py`（走 `core/certain_noise_clean.py`，DuckDB pass2/r2）→ `05_clean/run0N_v0M/` + `clean_summary.json`
4. **抽样** `tools/batch_ops/sample_qc.py --confidence 90 --margin 0.05`（SRS，c90/±5%）→ `02_sample/{用途}_c90/…_labeled_template.csv`；**默认不加 `--stratify`**（分层会丢小层、需加权，见「质量治理原则」）
5. **人标入库** `tools/ingest_human_qc.py` → `03_qc/{labeled,pass,fail,train_export}.csv`；同目录写 `analysis.md` 记 T/F 特征与人工合格率
6. **扩黑名单 v(N+1)**：只收金标 **T_hurt=0** 的 certain-noise；回滤时重跑第 3 步，产物落新 `run0N`，旧 run 保留为快照
7. **MiniLM**：`experiments/{category}_text_classifier.py` 训练 → `models/{category}_text_clf_f.pkl` + `_calibration.json`；`tools/score_{category}_text.py` 按 `calibration.t_like` 的 τ 过滤 → `06_tools/text_gov_v0N/`
8. **交付**：拷 `/Users/muse/data/deliver/`，批次内归档 `07_deliver/{方案}_{日期}.{md,json}`，回填 `清洗进度汇总.md`、`项目记录.md`、批次 `README.md`

**增量批次**：新机采单独建 `data/runs/{category}/{source}_{新batch}/` 走 2–3 步，再按 `video_id` 与老批 keep 做 union（重叠优先保留老批行），产物落老批 `05_clean/run0N_union_*/` + `union_summary.json`。

**大表口径**：百万~千万行一律 DuckDB 落盘库（`SET memory_limit` / `threads=2` / `temp_directory`），读表用 `read_csv_auto(..., all_varchar=true, ignore_errors=true)` 兜住 NUL 与坏行；**不要用 pandas 读整表**（千万行 OOM，且 pandas 重写 CSV 会破坏多行字段）。时长统计口径：`SUM(TRY_CAST(duration_seconds AS DOUBLE))/3600`。

## 临时文件约定

清洗过程中的**中间文件、试跑产物、一次性分析输出一律落 `work/`**（`work/{category}_{主题}_{MMDD}/`），不要写到仓库根目录或 `02_脚本/` 下；`work/` 是 gitignore 的 scratch 区，任务收尾后把结论/正式产物归档进 `项目记录.md`、`docs/`、`data/runs/`，其余删掉。详见 `work/README.md`。

## 产物命名规范

批次内产物统一 **`{输入文件名 stem}_{操作}_{阶段日期}.{csv,parquet}`**（即「原文件名 + 操作」），品类由批次路径 `data/runs/{category}/{source}_{batch}/` 推断（`core/batch_layout.infer_category`），中文品类名见 `core/category_labels.py`。例：合并产物 `raw/exo_service/商业服务_merged_0813.csv`，初筛产物 `商业服务_merged_0813_quality_0915.csv`，抽样产物 `exo_service_sample_0915.csv`，人工标注模板为 `exo_service_sample_0915_labeled_template.csv`。脚本另提供 `--name-tag`（抽样）/ `--stem`（合并）覆盖。规则迭代用 `05_clean/run0N_{版本}/` 区分，不覆盖旧 run。

## 数据现状

- 本机现存批次：`exo_agriculture`（0814 / 0818）、`exo_cook`（0916）、`exo_livestock`（0818）、`exo_parent`（0818 / 0917）、`exo_service`（0813）、`exo_团队协作`（0922：`machine_0922_pingpong` / `machine_0922_tennis`）、`exo_construction`（0923）；`raw/exo_agriculture`、`raw/exo_livestock` 已清空（大表回收），其余品类历史批次与指标见 `清洗进度汇总.md`。
- 已交付：农业 0818（72,003 h）、渔牧 0818（33,922 h）、医疗 0813。
- 各品类现行黑名单版本：农业 v0.6、渔牧 v0.5、烹饪 v0.3、亲子 **v0.4**、团队协作·乒乓 **v0.5** / 网球 **v0.7**（见 `categories/{category}/rules/blacklist.toml` 的 `[meta]`）。
- 进行中：`exo_parent`（v0.5 合格率 **18.4%**；text_gov_v02：`t_like` τ=0.82 **17,377 / 4,156 h**，实用备选 τ=0.51 **2,865,250 / 804,087 h**）、`exo_cook`（blacklist v0.3 + MiniLM τ=0.36，keep 751,722 / 135,207 h，`02_sample/text_gov_v02_c90/` 待标）、`exo_团队协作`（乒乓 text_gov_v05 ml_keep **8,303 / 8,606.6 h**；网球 τ≥0.6 抽样待标）、`exo_construction`（blacklist **v0.2.4**：金标**四批累积** 1,022 条，纯 F 频道最小化 **470** + 全 F keyword 硬闸 **35** 词 + `pass2` **39** 条，**四批 T_hurt=0**，规则 keep **134,914 / 27,156.6 h**（`05_clean/run09_v024`）；MiniLM `exo_construction_text_clf_f` OOF AUC 0.688，`t_like` τ=0.45 → `text_gov_v08` ml_keep **60,060 / 12,038.7 h**；CLIP 视觉层 `06_tools/clip_v01` **119,226 / 23,897.6 h**（τ=0.9926，基于 v0.2.3 池待重跑）；v0.2.4 池待抽样，`07_deliver/` 暂空，见 `docs/exo_construction_text_governance_v1.md`）。
 - ⚠ 该品类**标题层已到极限**：新金标内部 OOF AUC 仅 **0.595**，τ 从 0.45→0.70 合格率仅 50.5%→60.0%，交付 pass_rate 期望上限约 **60–65%**；进一步提质需视觉层（CLIP/缩略图判「有没有真人现场作业」）。
 - ⚠ v0.2.2 经验（可复用）：**精确 title 拉黑在样本内无意义**（construction 536 条金标标题 100% 唯一，生产上恒不匹配）；应用「F 专属短语」替代。**keyword 硬闸是有效杠杆**（池内 1,285 个采样词全部非空），但小样本（n_f 仅 2~9）整词拉黑风险高 → 必须用 **MiniLM 留存率**做独立旁证，高于池基准者剔除（construction 剔了 9 个真施工词）。
 - ⚠ v0.2.1 另修复一处**全仓通用缺陷**：`core/certain_noise_clean.py` 对 channel 只 `lower()` 不 `TRIM()`，精确锚点 `^…$` 会因频道名首尾空格漏匹配（construction 侧已用 `^\s*…\s*$` 规避，其它品类未修）。

## 质量治理原则（对齐人工标准）

- **人工标注是唯一可信的 pass_rate 来源**（`tools/ingest_human_qc.py` 入库 `03_qc/{labeled,pass,fail,train_export}.csv`）。交付/验收只认人工合格率，勿用 keep% / LLM-QC% / ml_score 冒充质量指标。
- **抽样：默认 SRS，分层只在池内 keyword 不碎片化时用**。`sample_qc.py --stratify` 的配额是 `floor(N_h×n/N)`，**`N_h < N/n` 的小层配额为 0 → 整层不进样本**，且层抽样率不等（需按 `audit_strata.csv` 的 `weight` 加权）。故分层结果只代表「参与分层」的子总体：factory 95–97%（可接受）、agriculture 发现后改走 SRS、**construction 仅 77.9%（993 层 / 6 万行，层均 61 行，掉 22.1%）**。判据：**层均行数 ≪ N/n 就别分层**；工具现已输出「分层覆盖 x%」并在 <95% 时 `[WARN]`。SRS 命名沿用 `{tag}_srs`（见 `exo_agriculture/.../v1_human_qc_c90_srs`）。⚠ FPC（`n_adj` / `effective_margin`）**只缩方差、不修偏差**，别拿它当无偏的保证。
- **文本治理三层（全品类现行口径）**，母版 `docs/exo_agriculture_text_governance_v2.md`，另见 `docs/exo_livestock_text_governance_v1.md`、`docs/exo_cook_text_governance_v1.md`：
  - **黑名单** = 非本品类主题闸门（游戏/音乐/卡通/明显串台等）；品类内 howto/日常实拍等不进黑名单，交 MiniLM。扩规则的硬闸门是**金标 T_hurt=0**，禁止裸词（如 `recipe`/`tutorial`/`haul`/`unboxing`/`workout`）。
  - **MiniLM** = 像不像人标 T 的排序；**τ 越高 → 留下越像 T**（默认读 `calibration.t_like`）。
  - 旧「certain-noise + t_hurt≤10% 宁杀勿漏」作历史对照，见 `calibration.recall` / `recall_aggressive`。
- **overturn 验证筛检器**：机检 drop 的「人工 T 误杀率（t_hurt）」与「F 召回（f_recall）」仍可用于选 τ / 报告；`apply_*` 支持 `--eval-labels` + `--halt-on-high-t-hurt`。
- **禁止无验证自训**：小模型/阈值校准必须以人工 gold 为准。
- 农业有效样本定义：`docs/exo_agriculture_agent_spec_v1.md` §1；文本治理以 v2 为准；旧 T/F 对齐评估见 `docs/农业机检对齐人工标准评估.md`。

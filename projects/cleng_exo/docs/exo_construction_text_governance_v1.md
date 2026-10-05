# exo 建筑施工 · 文本阶段治理（现行规则 **v0.2.4**；视觉层见文末 2026-09-28 CLIP 段）

> v1（v0.1.2）由数据剖析得规则闸门；v2 起由 **人工金标** 校正规则并接 MiniLM 排序；
> **v0.2.1** 合并第二批人标（`建工_6c91274f`，≥0.45 池）并修复 v0.2.0 三处实现缺陷。
> 文本-only 阶段依据本文档；视觉 Stage 暂不上（但见文末「标题层已到极限」）。

## 分层

| 层 | 职责 | 约束 |
|----|------|------|
| **规则过滤（频道）** | 1) `channel_construction_allowlist.csv` 正向白名单（频道施工词命中率 ≥ τ_ch）<br>2) `channel_pass2` 金标纯 F 频道（n_t=0）整频道丢弃 | 命中率不足但本条标题自带施工词 → 放行（OR） |
| **正则过滤（标题）** | 丢**与建筑施工无关**的确定串台（游戏/模拟器/影视/综艺/新闻/带货/玩具模型/播客/软件/考试/房产…） | 每条规则须**两个金标**都 **T_hurt=0** |
| **MiniLM** | 「像不像人标 T」——正类口径 = **有人的建筑施工**（现场劳作/工序），非宽泛「建筑/施工」主题词 | τ 越高越像 T；默认 `calibration.t_like` |
| **人工** | 唯一可信 pass_rate | keep% / 词表命中率 / ml_score 不当交付 KPI |

## 操作要点

1. 规则：`02_脚本/categories/exo_construction/rules/blacklist.toml`（**v0.2.4**，`pass2`=39 条 + `keyword_pass2`=35 词 + `channel_pass2`=470 频道）
2. 正向白名单：`rules/channel_construction_allowlist.csv`（**38,427** 个施工频道；step0 闸门）
   - 频道量级（22.8 万）远超 TOML 正则可承载范围，故 step0 用白名单 JOIN，不入 `blacklist.toml`
3. 纯 F 频道审计表：`rules/channel_blacklist_pure_f.csv`（全量 **280** 行，`decision` 列区分 `blacklist` / `pass2_covers`）
4. QC prompt：`rules/qc.toml`
5. 规则复现：`work/exo_construction_0924/build_v021.py`（`TAU_CH = 0.5`）
6. MiniLM：`experiments/exo_construction_text_classifier.py` → `models/exo_construction_text_clf_f.pkl` + `_calibration.json`
7. 打分：`02_脚本/tools/score_exo_construction_text.py`
8. 抽样：`02_脚本/tools/batch_ops/sample_qc.py … --confidence 90 --margin 0.05 --stratify`

## v0.2.1 修复的三处 v0.2.0 实现缺陷

| # | 缺陷 | 影响 | 修法 |
|---|------|------|------|
| ① | `build_construction_v2.py` 的 `hit_p2` 把 **SQL 文本当正则**再嵌一层 `regexp_matches` → `pass2` **从未生效** | v0.2.0 keep（183,649 行）里 **15,760 行**命中 pass2 却未丢；旧金标 F 实际只被频道黑名单清了 125 条 | 展开为真布尔 `regexp_matches(...) OR …` |
| ② | `hit_ch_bl` 只 `lower()` **不 TRIM**（`core/certain_noise_clean.py` 亦如此） | 6 个频道因尾部空格漏匹配 → **369 行 / 63.7 h** 漏放 | `channel_pass2` 模式两端加 `^\s*…\s*$` 容错锚点 |
| ③ | `PREMIER CONSTRUCTION & DEVELOPERS` 由旧金标 1F/0T 判为纯 F | 新金标 2T/1F → **真误杀** | 从黑名单剔除（合并金标 n_t>0 自动排除） |

> ② 是**全仓通用缺陷**：其它品类若用 `channel_pass2` 精确锚点（`^…$`）而未 TRIM，
> 同样会漏匹配。本仓已在 construction 侧用 `\s*` 锚点规避；`certain_noise_clean.py`
> 本身仍未 TRIM，建议后续统一改列侧 `TRIM(channel)`。

## machine_0923 现行操作点（v0.2.4）

| 项 | 值 |
|----|-----|
| 输入 | `建筑施工-01_65de9016_records(1)/` 12 个 CSV（1,562,296 行） |
| Bronze | `raw/exo_construction/建筑施工_merged_0923.csv`：**1,495,851 行 / 474,276.2 h**（去重 −66,445） |
| 初筛 | `01_quality/`：**1,327,266 行 / 370,115.6 h（88.7%）** |
| **现行 keep（规则 v0.2.4）** | `05_clean/run09_v024/`：**134,914 行 / 27,156.6 h（10.2%）** |
| **现行 keep（+MiniLM）** | `06_tools/text_gov_v08/`（v0.2.4 池重打分，τ=0.45） |
| 前一轮 keep（+CLIP 视觉层） | `06_tools/clip_v01/建筑施工_clip_pass.csv`：**119,226 行 / 23,897.6 h**（τ=0.9926；基于 run08_v023，**待 v0.2.4 池重跑**） |
| τ_ch | **0.5** |
| 待标样本 | 见 `03_qc/analysis.md`「待标」；v0.2.4 池尚未抽样 |

### 规范位置

| 内容 | 路径 |
|------|------|
| 规则依据 | `04_rules/NOTES.md` |
| 金标原始结果（四批） | `03_qc/gold_raw/` |
| 文本打分 | `06_tools/text_gov_v01…v08/` |
| CLIP 视觉层 | `06_tools/clip_v01/`（pass/drop/review/scored + summary） |
| CLIP 有效性评估 | `06_tools/clip_eval_v01/` |
| 标题「有人作业」层 | `06_tools/clip_v02_title_human_work/` |
| 配方 | `02_脚本/categories/exo_construction/recipe.toml`（`clip_filter` 为可选阶段） |

> v0.2.3 的 `run08_v023`（140,811）已作废：其金标集丢了 `建工_6c91274f` 一整批，
> 导致 keyword/channel 清单缩水 + T 误杀，见文末「金标回归」。仍保留为历史 run。

### 两批人标口径

| 金标 | 来源 | 抽样域 | T / F | 合格率 |
|------|------|--------|-------|--------|
| **old** | `03_qc/labeled.csv`（`const_v012_c90_strat`） | 规则 keep 池**全分数区间** | 90 / 177 | **33.7%**（加权 38.4%） |
| **new** | `Downloads/建工/建工_6c91274f_qc_result.csv` | 旧模型 **`ml_score≥0.45`** 区域 | 126 / 141 | **47.2%**（加权 49.5%） |

- 金标 T 定义：真人实拍施工：工地现场、结构/装修施工、机具作业、工序教学。
- 金标 F 主因：玩具/模型工程车、机械展会 walkabout、播客口播、软件/BIM 演示、考试理论、
  房产中介、影视游戏、工厂产线；**new 批次额外暴露**：产品/厂家推广、器材测评、
  非现场木工手作、维护保养、CAD 设计图文（**这些标题看上去都像施工**，见下节）。

### 丢量拆解（v0.2.1）

| step | 口径 | drop |
|------|------|------|
| step0 | 频道命中率 < 0.5 且标题无施工词 | 1,131,612 行 / 329,079.0 h |
| step1 | 金标纯 F 频道（`channel_pass2`，236 个） | 20,083 行 / 3,790.7 h |
| step2 | 标题硬噪声正则（`pass2`，29 条） | 15,394 行 / 4,693.9 h |
| **keep** | | **160,177 行 / 32,552.0 h** |

**两个金标 T_hurt=0**（脚本断言）。F 命中：old **171/177（97%）**、new **129/141（91%）**。

### 规则贡献拆解（区分「可泛化」与「循环论证」）

| 规则 | 派生自 | 旧金标 F 剔除 | 新金标 F 剔除 | T_hurt |
|------|--------|---------------|---------------|--------|
| `pass2`（29 条） | **仅旧金标** | **48/177（27%）** | 3/141（2%） | 0 |
| `channel_pass2` 旧派生（113 ch） | 仅旧金标 | 125/177 | 1/141 | 0（剔除 PREMIER 后） |
| `channel_pass2` 新增（123 ch） | **含新金标自身** | — | 126/141 | 0（**同源，循环**） |

> **只有 `pass2` 的增益是独立可验证的**（旧金标派生 → 新金标上仍 T_hurt=0）。
> 频道黑名单从 114 扩到 236 所剔除的 F，正是**用来派生这些频道的同一批 F**（131 个频道各仅 1 条人标），
> 故「F 命中 91%」不可当作泛化指标。代价实测很小：**多丢 7,599 行 / 1,499.0 h**
> （相对旧派生口径 167,534 / 34,031.2 h）；但风险已知：旧派生 114 频道的人工抽检 FP ≈ 15–20%。

### 黑名单最小化（280 → 236）

280 个纯 F 频道里，**44 个**的 F 样本已被 `pass2` 内容规则解释（Expo/口播/软件/考试等），
属内容型噪声而非频道型 → 不拉黑整频道。审计见 `channel_blacklist_pure_f.csv` 的 `decision` 列。

> 风险已知并接受：236 个中绝大多数**仅 1 条人标**支撑，且新增 123 个的 T_hurt=0 属同源验证。

## MiniLM 校准

| 项 | 值 |
|----|-----|
| 模型 | `models/exo_construction_text_clf_f.pkl`（MiniLM-L12 + LR，仅 title） |
| 训练 | n=533（T=215 / F=318）＝ old 266 + new 267，OOF **AUC 0.688 / AP 0.588** |
| 默认 τ | **`t_like.drop_threshold = 0.45`**（keep = `ml_score ≥ τ`，**不开启救援**） |
| keep | **63,326 行 / 12,363.6 h** |
| 高质量档 | **`t_precision` = 0.60** → 25,741 行 / 5,003.2 h |
| 待标样本 | `02_sample/text_gov_v06_srs`（269 条 SRS） |

### τ 曲线（OOF，近无偏；keep = `oof ≥ τ`，无救援）

| τ | 旧金标 合格率 / T_hurt | 新金标 合格率 / T_hurt | keep（池内） |
|---|----------------------|----------------------|--------------|
| 0.35 | 45.2% / 18 | 48.3% / 0 | 94,951 / 18,669.0 h |
| 0.40 | 47.8% / 25 | 48.2% / 7 | 78,752 / 15,511.8 h |
| **0.45** | **51.3% / 31** | **50.5% / 19** | **63,326 / 12,363.6 h** |
| 0.50 | 56.8% / 35 | 50.6% / 39 | 49,319 / 9,630.6 h |
| 0.55 | 65.1% / 46 | 52.8% / 59 | 36,736 / 7,156.6 h |
| **0.60** | **70.5% / 58** | **58.5% / 78** | **25,741 / 5,003.2 h** |
| 0.65 | 73.3% / 67 | 62.7% / 94 | 16,825 / 3,210.0 h |
| 0.70 | 79.0% / 74 | 60.0% / 111 | 9,800 / 1,841.4 h |

### 模型局限（关键结论）

**标题层已到极限。** 新金标（覆盖交付主区间 `ml_score≥0.45`）上：

1. **τ 抬不动了**：τ 从 0.45 升到 0.70，合格率只从 50.5% → 60.0%；τ=0.85 仍到不了 80%。
2. **残留 F 与 T 在标题上不可分**：用该金标自身做 5-fold OOF，AUC 仅 **0.595**（≈随机）。
   逐条标题信号检验，命中率全部回落到基准率 52.8%：
   例如 `how to/tutorial` 命中 62 条 F 率 53.2%（未命中 52.7%）；`品牌/公司频道名` 命中 147 条 F 率 55.8%。
   仅 `review/expo/product`（n=4）与 `news`（n=3）偏离，样本量无意义。
3. 语料上看，F 侧多为 *产品/厂家推广、器材测评、非现场木工手作、维护保养、CAD 图文*，
   与 T 侧「真人现场作业」的差别**不在标题**（两者都写着 bricklaying / concrete / plastering）。

> **下一步只能上视觉层**（`data/assets/embeddings/exo_*` CLIP / `qc/vision_thumb.py`）：
> 判「画面里有没有真人在现场作业」——这是标题无法承载的信息。
> 在此之前，交付 pass_rate 的期望上限约 **60–65%**（t_precision 档）。

### 救援规则：默认关闭

`should_rescue_labor`（「劳作动词 + 施工对象」二元短语）在 τ=0.45 下会救回 **4,158 行 / 1,134.9 h**，
但两个金标合计**只多保住 6 条 T**（各 3 条），救回的样本以游戏/微缩模型/软件/产品广告为主
（如 `Lord of the Rings: Return to Moria - Building Tutorial`、`DIY Mini Brick House | Miniature`、
`buildingEXODUS Construction Site Validation case`、`Construction Mason Worker PowerPoint Template`）。
→ **τ≥0.45 交付档默认不开启**；确需保 T 时再显式打开（见 `calibration.v021_scoring.rescue_tradeoff`）。

## 历史对照

| 版本 | 规则 | 说明 | keep |
|------|------|------|------|
| `run01_v010` | v0.1.0 | 初版 | 215,032 / 47,197.3 h |
| `run02_v011` | v0.1.1 | 增补 `building` 裸词误召（LEGO/桌游/建站/健身/模型/口播） | 212,917 / 46,714.6 h |
| `run03_v012` | v0.1.2 | 正/负向词表移除裸词 `building`/`builder`/`construct` | 187,639 / 38,526.0 h |
| `run05_v020` | v0.2.0 | 金标纯 F 频道（最小化 114）+ `pass2` 29 条 —— **但 pass2 因实现缺陷未生效** | 183,649 / 38,898.5 h |
| `text_gov_v03` | v0.2.0 + MiniLM | τ=0.25 + 二元短语窄救援 | 162,970 / 33,586.4 h |
| `run06_v021` | v0.2.1 | 修复 pass2/TRIM/PREMIER 三缺陷 + 合并新金标（236 频道） | 160,177 / 32,552.0 h |
| `text_gov_v05` | v0.2.1 + MiniLM | τ=0.45、无救援 | 63,326 / 12,363.6 h |
| **`run07_v022`** | **v0.2.2** | **新增全 F keyword 硬闸（231 频道 + 21 词 + 34 条 pass2）** | **147,007 / 29,576.5 h** |
| **`text_gov_v06`** | **v0.2.2 + MiniLM** | **τ=0.45、无救援（现行）** | **60,074 / 11,687.0 h** |

## v0.2.2：全 F 频道 / 标题 / keyword 三级拉黑

### 1. 频道（沿用 v0.2.1）

合并金标纯 F 频道（`n_t=0`）：280 个 → **最小化 231 个**（49 个 F 已被 `pass2` 覆盖，
无需拉黑整频道）。清单 `rules/channel_blacklist_pure_f.csv`。

### 2. keyword 硬闸（新增，本版主要收益）

`keyword` 是采样检索词，池内 **1,285 个全部非空**，量级大 → 整词丢弃是有效杠杆。

- 派生口径：合并金标 `n_t=0` 且 `n_f≥2` → 30 个候选
- **独立旁证（关键）**：金标只有 2~9 条/词，单靠人标整词拉黑风险高。
  故对每个候选再看它在 v0.2.1 keep 池里的 **MiniLM τ=0.45 留存率**（与金标无关的独立信号）。
  池基准 39.5%，**高于基准者剔除**（说明这批内容「像 T」，拉黑会砍正样本）：
  30 → **采纳 21，剔除 9**。
  被剔除的 9 个（恰是 `brick wall workers building` 75.3%、`real bricklaying construction work` 67.6%、
  `bridge workers building` 63.5%、`workers constructing brick wall` 59.8%、`foundation building process` 55.7%、
  `professional masonry building playlist` 51.4%、`building construction skills techniques playlist` 49.3%、
  `professional engineering construction channel` 48.8%、`rcc construction` 47.1%）
  看着就是真·现场作业词 —— 只靠 2 条人标就拉黑会把正样本砍掉。
- 规则形态：精确锚点 `^\s*(?:…|…)\s*$` + `'i'`（`certain_noise_clean` 读 `keyword_pass2`，硬闸）
- 池内影响：丢 14,861 行 / 3,477.1 h

### 3. 标题（新增，但收益很小）

**精确标题拉黑在本品类无意义**：金标 536 条标题 **100% 唯一**（536 distinct / 536 行），
按精确 title 拉黑在生产池上恒不匹配 —— 等于把金标背下来。

故改用**F 专属短语**：F 标题命中 ≥4 次、且 T 标题 **0 次**的 bigram，再经语义清洗
（只保留测绘/设计制图/设备器材/安全培训/清洁维护/结构讲解/儿童类，剔掉
`god work`/`com wall`/`cover pvt` 这类碎片），并入 `pass2`（title||channel 软闸）：

| 新增 pass2 类别 | 依据（F 标题支撑） | keep 池内 τ0.45 留存 |
|---|---|---|
| `surveyor / surveying / land survey / auditorium drawing` | `land surveyor` F=12、`surveyor work` F=12 | 0.0%（1 行） |
| `cat excavator / construction excavator / construction mega / blade concrete` | `cat excavator` F=5、`construction excavator` F=8 | 0–8.1% |
| `construction safety / safety hazards / precautions safety` | `construction safety` F=7 | 1.5%（136 行） |
| `clean grout` | F=4 | 16.7% |
| `bridges suspension` | `bridges suspension` F=6 | 0 行 |

⚠ 被**排除**的短语（τ0.45 留存率高，是像 T 的内容，不拉黑）：`design tiles`(F=12，留存 100%)、
`arch design`(77.4%)、`construction tips`(63.1%)、`concrete cutting`(56.4%)、`building surveyor`(40.0%)。

> 结论：标题层已被 v0.2.1 的频道/短语规则基本清空 —— 上述新增短语在 keep 池内合计只命中
> 约 **250 行 / 27 h**。它们的价值是**防新批次回归**，而非提升当前交付质量。

### 4. 金标自检

四层（channel_pass2 + keyword_pass2 + pass2）叠加后：**两个金标 T_hurt = 0**；
F 命中 old **171/177 (97%)**、new **129/141 (91%)**。

> 口径说明：keyword/标题清单由同一批金标派生，故「金标合格率提升」属 in-sample，**不作无偏报告**；
> 本版的无偏证据是上面那条「9 个词因 MiniLM 留存率高于池基准而被剔除」。



## 2026-09-28 人标增量更新（v0.2.3）

新增人工标注 `建筑施工_c961381c_qc_result.csv`：T=81、F=188、未标 2 条。合并既有两批标注后，对 T 的规则误删仍为 0。按新标注中的 F 模式增加微缩/动画施工、安全提示/培训、焊接爱好手作、设备运输装卸、设备论坛/厂商证言五组标题规则；F 命中旧批 172/177、新批 176/188。新增 28 个纯 F keyword 中采纳 19 个，另将纯 F 频道表扩至 266 个（规则去重后）。

重清洗版本为 `work/exo_construction_0924/build_v023.py`，输出目录 `data/runs/exo_construction/machine_0923/05_clean/run08_v023/`：保留 140,811 条 / 28,477.2 小时（10.6%），较 v0.2.2 减少 6,196 条 / 1,099.3 小时。新增样本仍有 12 条 F 未被文本规则命中；没有足够安全的 title/channel 模式可泛化，故保留这些内容。具体证据和各过滤层数量见 `work/exo_construction_0928/建筑施工_人标增量与过滤更新报告.md`。

## 2026-09-28 视觉层：CLIP 缩略图风险过滤（可选 Stage）

标题层到极限后接视觉层，判「画面里有没有真人现场作业」。评估与执行产物已归档进批次：

| 内容 | 路径 |
|------|------|
| 有效性评估 | `data/runs/exo_construction/machine_0923/06_tools/clip_eval_v01/建筑施工CLIP有效性评估.md` |
| 逐条 OOF 预测 / 指标 / 样本 | 同目录 `group_oof_predictions.csv` / `cv_metrics.json` / `labeled_sample.csv` |
| 全量过滤结果 | `06_tools/clip_v01/`（`建筑施工_clip_pass.csv` / `_drop` / `_review` / `_scored` + `summary.json`） |
| Keep 质检样本 | `06_tools/clip_v01/建筑施工_CLIP_keep抽样质检_90CI_05error_全量.csv`（270 条，按 keyword 前 25 层加权） |
| 工具 | `02_脚本/tools/run_exo_construction_clip_filter.py`（默认停在 `work/…`，归档时拷入 `06_tools/clip_v01/`） |

### 评估口径（269 条独立人标，248 频道，5-fold StratifiedGroupKFold 按频道分组）

| 特征 | 加权 ROC-AUC (F) | 加权 AP (F) |
|------|------------------|-------------|
| CLIP 图片 | **0.6661** | 0.8170 |
| title + channel（TF-IDF） | 0.5378 | 0.7573 |
| 文本 + CLIP | 0.6622 | **0.8413** |

### 全量执行（输入 = run08_v023 keep 140,811）

| 项 | 值 |
|----|-----|
| 阈值 | τ = **0.9926**（5-fold 频道分组 OOF CLIP-only；加权 T_hurt 1.28%、F recall 7.84%） |
| pass | **119,226 行 / 23,897.6 h** |
| drop | 21,585 行 / 4,579.7 h（高风险 7,604 + 无缩略图 13,981） |
| 策略 | 只留 `clip_pass`；`clip_review` 与 `no_thumbnail` 一并 drop，源 CSV 不覆盖 |

> ⚠ 阈值仍是小样本分组 OOF 校准；**未标注前 pass 集不等于实际准确率**。
> 评估文档明确建议：CLIP 用于排序/人工复核优先级或与文字规则联合，**不要单独作为自动剔除门槛**。

### 标题「有人作业」层（实验档，未进交付）

`06_tools/clip_v02_title_human_work/`：在 CLIP pass 上再按标题「有施工作为 + 施工语境、排除教程/演示/非施工手作」过滤，
119,226 → **196 行 / 106.0 h**（keep_share 0.16%）。过严，仅作方向验证。

## v0.2.4：金标改四批累积（修复 v0.2.3 回归，2026-09-28）

### 缺陷（v0.2.3）

`work/exo_construction_0924/build_v023.py` 的金标合并只读 `labeled.csv` + `建筑施工_c961381c`（536 行），
**丢掉了 v0.2.2 用过的 `建工_6c91274f`（267 行）**，另 `建工2_050a0200`（269 行）从未使用。后果：

1. `construction site` / `construction work` / `house construction` 等 keyword 在缩小后的金标里 `n_t=0`
   → 被整词硬闸；但套到 `建工` 上**误杀 11 条 T**、`建工2` 上**误杀 6 条 T** → v0.2.3 的 T_hurt ≠ 0。
2. 15 个 v0.2.2 已闸 keyword 因 `n_f` 掉到 1（< MIN_KW=2）掉出候选 → **重新放行 11,432 行**
   （run08 vs run07：+11,432 / −17,628 / 净 −6,196）。
3. ML 独立旁证仍读 `run06_v021` keep + `text_gov_v05` scored（v0.2.1 口径），未随 run 更新。

### 修复（v0.2.4）

| 项 | 改法 |
|----|------|
| 金标 | **四批累积合并**：`labeled.csv` + `建工_6c91274f` + `建工2_050a0200` + `建筑施工_c961381c`，按 `video_id` 去重 → **1,022 条**（去重后 T=397 / F=625） |
| 原始结果归档 | `03_qc/gold_raw/`（+ `README.md` 记来源/口径） |
| ML 旁证池 | 改为**上一轮最新** `text_gov_v07/…ml_scored.csv`；基准留存率**池内实测**（42.2%），不再硬编码 0.395 |
| 断言 | **四批**金标 `T_hurt = 0`（脚本内断言） |

### 结果

| 项 | v0.2.3 | v0.2.4 |
|----|--------|--------|
| keyword_pass2 | 19 | **35** |
| channel_pass2 | 266 | **470** |
| keep | 140,811 / 28,477.2 h | **134,914 / 27,156.6 h（10.2%）** |
| run | `run08_v023` | **`run09_v024`** |
| T_hurt（四批） | 建工 11 + 建工2 6 ✗ | **0 ✓** |

F 命中：old 172/177（97%）、jg 129/141（91%）、jg2 95/119（80%）、c961 169/188（90%）。

> 复现：`work/exo_construction_0928/build_v024.py`。**`run08_v023` 已作废，勿作交付依据。**
> 遗留：CLIP 视觉层（`clip_v01`）与 `text_gov_v08` 的前序 `v07` 均基于旧池，按需在 v0.2.4 池重跑。

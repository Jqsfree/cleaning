# exo 单人舞蹈 · 文本阶段治理（黑名单 v0.2.1 + MiniLM v0.1 + CLIP solo v0.1）

> 品类：`exo_dance`（单人舞蹈）；批次 `data/runs/exo_dance/machine_0923/`。
> 本文档记录由人标金标 `kpop_0df26300` 校正的黑名单 v0.2.0 / v0.2.1，
> 首个 MiniLM 排序器（v0.1）的**诚实结论：本品类标题层已到极限**，
> 以及缩略图 CLIP 零样本闸门（solo ∧ is_dance）的首轮全量结果。
> 母版口径见 [`exo_agriculture_text_governance_v2.md`](exo_agriculture_text_governance_v2.md)。

## 分层

| 层 | 职责 | 约束 |
|----|------|------|
| **规则过滤（频道）** | `channel_pass2` 整频道硬闸：金标纯 F 频道（`n_f≥2` 且 `n_t=0`）+ 乐器台/电视台/无版权音乐台/游戏台 | 一律 `^\s*…\s*$` 精确锚点 |
| **正则过滤（标题）** | `pass2` 软闸（串台主题，可被 `rescue` 豁免）；`r2` 硬闸（观看向舞蹈 / 双人舞 / 非舞健身，不可豁免） | 每条规则须金标 **T_hurt=0** |
| **MiniLM** | 「像不像人标 T」排序；正类口径 = 单人舞蹈（教学/练习/表演/即兴） | τ 越高越像 T；本品类默认 `calibration.apply` |
| **CLIP（缩略图）** | 封面是否「单人在跳舞」：`q1_solo_dancer` ∧ `q2_is_dance` | 阈值初值 0；`no_thumb` 单列、不入 pass |
| **人工** | 唯一可信 pass_rate | keep% / ml_score / clip_pass% 不当交付 KPI |

## 批次与现行操作点

| 项 | 值 |
|----|-----|
| 输入 | `~/Downloads/exo单人舞蹈-机采1.csv`：**62,282 行 / 8,238.9 h**（50 个舞蹈 playlist 检索词） |
| 初筛 | `01_quality/`：**55,577 行 / 8,025.2 h**（89.2%） |
| **现行 keep（规则）** | `05_clean/run03_v021/`：**41,367 行 / 6,094.1 h（74.4%）** |
| **现行 keep（+MiniLM）** | `06_tools/text_gov_v01/…_ml_keep.csv`：**40,125 行 / 5,986.8 h**（τ=0.26） |
| **现行 keep（+CLIP solo）** | `06_tools/clip_solo_v01/…_pass.csv`：**28,448 行 / 4,489.6 h**（阈值 0/0） |
| 待标样本（文本出货池） | `02_sample/text_gov_v01_ml_c90_strat/`：269 条（c90/±5%，50 keyword 层，seed 42，**从 ml_keep 抽**） |
| 旧样本（已作废） | `02_sample/text_gov_v021_c90_strat/`：从**规则 keep**抽，内含 7 条已被 MiniLM 判 drop 的行，勿再送标 |
| 金标 | `~/Downloads/建工/kpop_0df26300_qc_result.csv` → `03_qc/{labeled,pass,fail}.csv` |

## 人标金标（0df26300）

269 条分层样本 → 有效 **265**（T=117 / F=148 / U=3），**人工合格率 44.2%**（未加权）。

| 口径 | 内容 |
|------|------|
| **T（保留）** | 单人舞蹈影像：教学（how to / lesson / tutorial / breakdown / step by step / drills）、表演与即兴（belly dance solo、lyrical solo、freestyle、单人 battle）、单人跟练（单人 mirrored 练习、单人跟练拉伸） |
| **F（剔除）** | 团体·多人舞（团体 kpop 练习室 / KPOP IN PUBLIC crew / line dance / 民俗团体）、双人舞（ballroom·latin·swing·salsa 伴侣套路）、观看向直拍（직캠·fancam·facecam / BE ORIGINAL / Performance ver. / MV）、舞蹈健身课（zumba·cardio）、非舞蹈串台（英语语法 / 吉他 / 圣经 / 羽毛球 / 美发 / 游戏 / 纯音乐发行） |

## 黑名单 v0.2.0 / v0.2.1

### v0.2.0（首轮金标校正）

金标揭示本品类口径 =「单人舞蹈」，故新增 `channel_pass2 = gold_pure_f_ch`（原 127 个纯 F 频道）
与 11 条 `r2` 观看向 / 双人舞 / 非舞健身硬闸（`view_dance_practice` / `view_be_original` /
`view_performance` / `view_dance_along` / `view_official_video` / `view_busking` /
`partner_couple` / `partner_line_dancers` / `fitness_rehab` / `music_release` / `misc_worship`）。

> 观看向舞蹈**不能**走 `pass2`：`rescue` 里的舞蹈信号会把它放行，故一律 `r2` 不可豁免。

**金标验收：T_hurt = 0/117，F_caught = 140/149（94.0%）**；keep 39,687 行。

### v0.2.1（薄支撑不立闸）

按「1 条人标不足以支撑整频道硬闸」剔除 **116 个仅 1 条 F 支撑的频道**（127 → 11），
仅回收 F 支撑 ≥2 且 T_hurt=0 的 CJK 观看向词（`view_performance` 并入 `퍼포먼스`）。

| step | 口径 | drop |
|------|------|------|
| step0 | `channel_pass2` 频道硬闸 | 6,551 行 |
| step0b | `keyword_pass2` | **0 行**（本品类未启用） |
| step1 | `pass2` 软闸（16 组串台） | 1,172 行（另有 863 行被 `rescue` 豁免） |
| step2 | `r2` 硬闸（观看向 / 双人 / 健身） | 6,487 行 |
| **keep** | | **41,367 行 / 6,094.1 h** |

**代价显著**：整闸金标 **F 召回 94.0% → 41.2%**（T_hurt 仍为 0），keep 39,687 → 41,367 行。
残余 F 主要落在「有 T 有 F 的混合频道」（ADTC Dance Camps / Kiseki Mirrored Dances /
Marius & Elena Official / M2）与韩日葡西音译的观看向标题。

## MiniLM v0.1

| 项 | 值 |
|----|-----|
| 模型 | `models/exo_dance_text_clf_f.pkl`（MiniLM-L12 + LR，**仅 title**） |
| 训练 | n=265（T=117 / F=148）＝ 全部有效金标；**OOF AUC 0.668 / AP 0.613** |
| keep 池子集 OOF AUC | **0.691**（生产口径；池内金标 204 = T117 / F87） |
| 样本内 keep 池 AUC | 0.863 —— **不可作报告**（模型拟合过这 265 条） |
| 默认 τ | **`apply.drop_threshold = 0.26`**（`pick_apply_threshold`：保 T 上限 t_hurt_rate≤3.2% 内最大化 f_caught） |
| keep | **40,125 行 / 5,986.8 h**（丢 1,242 行 / 107.3 h） |
| 金标验收（实际产物核对） | 见下「端到端核验」：**T_hurt = 0/117**，池内 F_caught 7/148 |
| 分层加权估计（OOF） | drop 精度 **0.86**、池内 F 召回 **3.4%** |
| 打分 | `02_脚本/tools/score_exo_dance_text.py`（默认读 `apply`；`--legacy-t-like` 见下） |

### 端到端核验（实际产物 × 金标，非 OOF）

核验脚本 `work/exo_dance_0924/gold_end2end_eval.py`（金标 265 行 → 规则闸 → MiniLM）：

| 层 | n_t | n_f | T_hurt | F_caught | F 召回 | drop 精度 |
|----|-----|-----|--------|----------|--------|-----------|
| 规则 v0.2.1（run03_v021） | 117 | 148 | **0** | 61 | 41.2% | 1.00 |
| MiniLM τ=0.26（叠加） | 117 | 148 | **0** | 7 | 4.7% | 1.00 |
| **端到端（规则 ∪ MiniLM）** | 117 | 148 | **0** | **68** | **46.0%** | **1.00** |
| MiniLM 在规则 keep 池内 | 117 | 87 | **0** | 7 | 8.1% | 1.00 |

分层加权（keyword 层权重）：MiniLM 在池内 drop 精度 **1.00**（加权 F 1057.8 / 加权 T 0.0），
加权 F 召回 8.0%。**实际部署模型比 OOF 估计更干净**（OOF 曾估 t_hurt=1、精度 0.86）——
符合预期：OOF 是「没见过的数据」，部署模型见过全部 265 条金标。

<details><summary>τ=0.26 实际抓到的 7 条（全部 F，零 T 误伤）</summary>

```
[0.170] F | Tokyo Cyber Detective Brigade 東京電脳探偵団 [Mirrored] | Kiseki Mirrored Dances
[0.196] F | [ODOTARA] K-POP IN PUBLIC JAPAN | IZ*ONE - PANORAMA | G-Project
[0.215] F | [KPOP IN PUBLIC] ITZY - Not Shy (Male Ver.) Dance Cover | Fossil and Bones F&B
[0.233] F | [MPD직캠] 방탄소년단 정국 직캠 4K 'FAKE LOVE' | M2
[0.239] F | Emerald Planet Short Version Mirror | SakuraShounen
[0.252] F | [페이스캠4K] 지민 'Like Crazy' | SBSKPOP ZOOM
[0.259] F | Folkeswing | sverred
```

</details>

> ⚠️ 打分用的 `--resume` 会校验 `scoring_code_sha256`：脚本改一行注释都会导致旧 checkpoint 失配。
> 本次因修 docstring 重新生成了 `text_gov_v01/`（重跑前已在 `work/` 下比对，`ml_score` 最大差 0、`ml_auto_drop` 逐行一致）。

### drop 集构成（τ=0.26 实际丢掉的 1,242 行 / 107.3 h）

| Top 频道 | 行数 | | Top keyword | 行数 |
|---|---|---|---|---|
| M2（打歌直拍） | 49 | | `kpop solo choreography breakdown` | 390 |
| KBS Kpop | 36 | | `camera facing dance practice solo` | 284 |
| SBSKPOP ZOOM | 24 | | `street style dance breakdown fast` | 56 |
| MBCkpop | 15 | | `waacking dance breakdown slow motion` | 54 |
| Harmonyc Movement | 13 | | `modern dance practice routine alone` | 52 |

`ml_score` 分位：keep 池 p01=0.272 / p50=0.516；drop 集 p99=0.259（与 τ=0.26 同界）。
→ 模型实际抓的是**打歌台直拍 / 面对面练习室（camera facing）** 这一块，方向正确但覆盖面窄；
`rescue` 在 τ=0.26 下仅保住 2 行。

### τ 曲线（OOF，keep 池分层加权口径）

| τ | 估计砍掉 | 小时 | 其中 F | 其中 T | drop 精度 | 池内 F 召回 |
|---|---------|------|--------|--------|-----------|-------------|
| **0.26（默认）** | **834** | **101** | **719** | **115** | **0.86** | **3.4%** |
| 0.30 | 1,931 | 223 | 1,412 | 518 | 0.73 | 6.6% |
| 0.35 | 4,781 | 642 | 3,472 | 1,309 | 0.73 | 16.2% |
| 0.40 | 9,804 | 1,306 | 7,025 | 2,778 | 0.72 | 32.8% |
| 0.45 | 13,237 | 1,820 | 9,604 | 3,633 | 0.73 | 44.9% |
| 0.542（通用 `t_like`） | 24,918 | 3,620 | 15,150 | 9,769 | 0.61 | 70.8% |

> **通用 `t_like` 在本品类不可用**：τ=0.542 会砍掉 **30% 金标 T**；故 `score_exo_dance_text.py`
> 默认读 `apply`，`t_like` 降级为 `--legacy-t-like`。其它品类 τ=0.26 相当于「几乎不砍」，
> 这正是本品类的真实处境（见下节）。

### 救援规则

`should_rescue_dance` 只认**明确单人信号**（`solo` / `improvis` / `即兴` / `单人舞蹈`），
且被团体·双人·舞cover·直拍·练习室·健身·非舞蹈主题阻断。金标上 `n_hit=11`、
**T 救回 11、F 误救 0**；τ=0.26 下实际救回 0 行（阈值本就很低），价值是防新批次回归。

## CLIP solo 闸门 v0.1（缩略图零样本）

标题层到极限后，按建议切到视觉层：封面是否「**单人在跳舞**」。
架构对齐 `exo_service` L3：`ClipEncoder`（open_clip ViT-B-32）+ 双问题合取。

| 项 | 值 |
|----|-----|
| 输入 | `text_gov_v01/…_ml_keep.csv`（40,125 / 5,986.8 h） |
| 配置 | `categories/exo_dance/rules/cascade_clip.toml` |
| 问题 | `q1_solo_dancer`（单人 vs 团体/双人）∧ `q2_is_dance`（在跳舞 vs 非舞画面） |
| 阈值 | 初值均为 **0.0**（与 service v1.2 同款零阈值起步） |
| 跑法 | `tools/run_exo_dance_cascade_clip.py … -o …/clip_solo_v01/ -n N`（`0`=全量） |
| 缩略图缓存 | `qc_thumb_cache/exemplar_sim`（与 service 共用） |

### 试跑（`-n 2000`，seed 42）

| | 条数 | 小时 |
|--|------|------|
| pass | 1,477 | 250.8 |
| fail | 523 | 70.4 |
| no_thumb | 0 | 0 |
| **pass_rate（有图）** | **73.9%** | |

金标重叠仅 7 条（T4 / F3）：**T_hurt=0/4**，F 抓到 1/3。样本过小，不作阈值依据。

### 全量（`-n 0`）

| | 条数 | 小时 |
|--|------|------|
| **clip_pass** | **28,448** | **4,489.6** |
| clip_fail | 11,621 | 1,485.5 |
| no_thumb（单列，不入 pass） | 56 | 11.7 |
| 有图 | 40,069 | — |
| **pass_rate（有图）** | **71.0%** | — |
| 相对 ml_keep | 丢掉 28.9% 行 / 25.0% 小时 | |

产物：`06_tools/clip_solo_v01/clip_solo_full_{scored,pass,fail}_0924.csv` + `clip_solo_full_summary.json`。

### 金标重叠（全量 × `03_qc/labeled`，ml_keep 内 197 条）

| | n | clip_fail | 率 |
|--|---|-----------|-----|
| T（pass） | 117 | **16** | **T_hurt = 13.7%** |
| F（fail） | 80 | 37 | F 召回 = 46.3% |
| no_thumb | — | 1（F） | — |

> ⚠️ 零阈值起步 **T_hurt 偏高**（16/117）。下一步只动视觉层：下调 `q1_solo_dancer` 阈值
> （负向放宽）或收紧 q1 neg 提示，**不改**黑名单 / MiniLM。本版 `clip_pass` 作视觉候选池，
> 交付仍须等人标；勿把 71% pass_rate 当合格率。

## 关键结论：本品类标题层已到极限

金标 T/F 的边界**不是标题层的语义边界**，证据三条：

### 1. 「教学向」在 T / F 两侧都大量出现

| 标题语义组 | T 命中（占 T） | F 命中（占 F） |
|-----------|---------------|---------------|
| 教学词 `tutorial/lesson/how to/step by step/breakdown/drills/learn` | 46（39.3%） | 23（15.5%） |
| 观看向 `practice/mirrored/fancam/직캠/cover/BE ORIGINAL/MV` | 7（6.0%） | 45（30.4%） |
| 双人 `couple/duet/partner/oversway/underarm` | 2（1.7%） | 10（6.8%） |
| 明确单人 `solo/improvis/即兴` | 11（9.4%） | **0（0.0%）** |

只有「solo/即兴」是干净的 T 信号 —— 这也正是 `should_rescue_dance` 只认它的原因。

### 2. 同格式、同频道、标签相反（可直接核对的矛盾）

| 同格式对 | 判 T | 判 F |
|---------|------|------|
| M2 的 `[MPD직캠] … FanCam` | `전소미 직캠`（SOMI） | `방탄소년단 정국 직캠`（Jungkook）×2 |
| 街舞教学 | `Locking Tutorial Compilation with Skeeter Rabbit` | `Popping Tutorial` / `Krump Tutorial` / `Locking Tutorial 7` |
| Salsa 教学 | `Salsa Footwork Tutorial ★Beginners★`、`…Tutorial 16: 47-Combo` | `Salsa Beginners 01 - Salsa Basic Techniques`（**同频道 Marius & Elena Official**） |

### 3. OOF 分数在低分区完全交错

keep 池内金标按 OOF 分升序，最低的 11 条是：
`KPOP IN PUBLIC`(F) → `KPOP IN PUBLIC`(F) → `Mirrored`(F) → **`SHINee 화보 촬영 Behind`(T)** →
`Folkeswing`(F) → `Mirror`(F) → **`DINO'S DANCEOLOGY`(T)** → **`BTS Jungkook fancam`(T)** →
`NCT TAEYONG FANCAM`(F) → **`SW2 Chiftitelli`(T)** → `BTS JUNGKOOK FanCam`(F)。
**任何切点都会同时砍到 T 和 F**，这就是 keep 池 OOF AUC 只有 0.691、drop 精度天花板 ~0.73 的原因。

### 4. keyword 硬闸：审计后**全部拒绝**

金标里 **5 个抽样 keyword 是 T=0（`n_f≥2`）**：

| keyword | 金标 F | 池内行 | 池内小时 | τ=0.26 留存 |
|---------|--------|--------|----------|-------------|
| `latin dance solo steps tutorial` | 6 | 1,011 | 123.5 | **99.6%** |
| `vogueing tutorial solo practice` | 5 | 917 | 130.0 | 95.1% |
| `popping techniques solo tutorial` | 4 | 703 | 95.5 | 97.3% |
| `home dance practice guide` | 3 | 430 | 56.4 | 99.3% |
| `street dance move by move breakdown` | 2 | 334 | 36.2 | 99.4% |

**池基准留存率 = 97.0%**，5 个候选全部 ≈/高于基准 → 该 keyword 捞回来的内容**大多像 T**，
拉黑会砍掉 **3,395 行 / 441.6 h 的正样本**。逐条看这批金标 F 也都是**离题的孤例**
（吉他赛 / 美发教程 / line dance / 圣经 / 羽毛球的检索误召），不构成 keyword 级结论。
→ 与 construction v0.2.2 相反，**本品类 keyword 杠杆不成立**（这正是「独立旁证」机制的价值）。

## 建议下一步

1. **复标 / 仲裁（最高优先）**：现有 265 条里至少 20–30 条互相矛盾（见 §2）。
   建议按「单人 / 多人」「是否有真人跳」「是否教学」三问拆成结构化标注，
   或对现有 gold 做一次**同题仲裁**，然后再训 —— 否则任何模型的 AUC 上限都被噪声锁死。
2. **放弃在标题层继续加正则**：v0.2.1 已把标题能表达的「观看向 / 双人 / 非舞健身」吃干净；
   续加的规则只会变成「把金标背下来」。
3. **CLIP 阈值 / 提示词迭代（进行中）**：`clip_solo_v01` 已全量落地；金标 T_hurt=13.7% 偏高，
   下一步只下调 `q1` 或收紧 neg，不改文本层。可对 `clip_pass` 重抽 c90 送标验视觉合格率。
4. **交付口径（视觉候选）**：`06_tools/clip_solo_v01/…_pass.csv`（**28,448 / 4,489.6 h**）；
   文本出货池仍为 `text_gov_v01/…_ml_keep.csv`（40,125 / 5,986.8 h）。
   人工合格率**未知**；`02_sample/text_gov_v01_ml_c90_strat/`（269 条）送标后
   `tools/ingest_human_qc.py` 才给得出货 pass_rate。MiniLM τ=0.26 **不应作为质量提升手段上报**。

## 历史对照

| 版本 | 规则 | 说明 | keep |
|------|------|------|------|
| `run01_v013` | v0.1.3 | 机采摸底（无金标），频道闸 + 主题闸 + r2 游戏/beat | 48,399 / 6,619.3 h |
| `run02_v020` | v0.2.0 | 首轮金标校正：纯 F 频道 127 + 11 条观看向/双人/健身 `r2` | 39,687 / 5,947.5 h |
| **`run03_v021`** | **v0.2.1** | **薄支撑不立闸：纯 F 频道 127 → 11（F 召回 94.0% → 41.2%）** | **41,367 / 6,094.1 h** |
| **`text_gov_v01`** | **v0.2.1 + MiniLM** | **τ=0.26（`apply`，保 T 口径）** | **40,125 / 5,986.8 h** |
| **`clip_solo_v01`** | **+ CLIP solo∧dance** | **阈值 0/0；no_thumb 56 单列** | **28,448 / 4,489.6 h** |

## 产物清单

| 用途 | 路径 |
|------|------|
| 规则 | `02_脚本/categories/exo_dance/rules/blacklist.toml`（v0.2.1） |
| 规则构建脚本 | `work/exo_dance_0924/gold_0df26300/build_rules.py`、`gen_channels.py`、`channel_support.py` |
| 训练 | `experiments/exo_dance_text_classifier.py` |
| 权重 / 标定 | `models/exo_dance_text_clf_f.pkl`、`models/exo_dance_text_clf_f_calibration.json` |
| 打分 | `02_脚本/tools/score_exo_dance_text.py` |
| 池内打分产物 | `06_tools/text_gov_v01/`（`_ml_keep` / `_ml_drop` / `_ml_scored`） |
| CLIP 配置 / 模块 | `categories/exo_dance/rules/cascade_clip.toml`、`cascade_clip.py` |
| CLIP CLI | `02_脚本/tools/run_exo_dance_cascade_clip.py` |
| CLIP 产物 | `06_tools/clip_solo_v01/`（`clip_solo_full_{scored,pass,fail}_0924` + summary） |
| 端到端核验 | `work/exo_dance_0924/gold_end2end_eval.py`（逐层 T_hurt/F_caught + 分层加权）→ `end2end_metrics.json` |
| 送标样本 | `02_sample/text_gov_v01_ml_c90_strat/exo_dance_sample_0924_labeled_template.csv`（**盲标**：已剔除 `ml_*` 列） |
| 评估脚本 | `work/exo_dance_0924/minilm_oof_eval.py`（OOF 诚实口径）、`minilm_pool_impact.py`（分层加权影响） |
| 人标 | `03_qc/{labeled,pass,fail,train_export}.csv` |

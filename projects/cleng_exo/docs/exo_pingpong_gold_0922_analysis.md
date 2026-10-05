# exo_团队协作·乒乓 人标 72fede47 规则迭代分析（v0.4.x）

- 人标文件：`团队协作and竞争_72fede47_qc_result.csv`（259 条，T 88 / F 90 / 未标 81）
- 抽样来源：`data/runs/exo_团队协作/machine_0922_pingpong/06_tools/text_gov_v02/双人乒乓_merged_0922_ml_keep_0922.csv`（τ=0.521 的 ml_keep，259/259 命中）
- 结论：新增规则在**金标 T_hurt = 0** 前提下，把该 gold 的 F 从 keep 中清掉 **71/90（79%）**。

## 1. 全 F 频道（n_t=0）

43 个频道在本次人标里全部为 F。按能否整频道拉黑分两类：

| 组 | 频道 | 处理 |
|---|---|---|
| 非乒乓串台（21） | VGBootCamp、International Cheer Union、Fully Faltoo、SQUASHTV、touchtennis、WST、The Lallantop、Unbox Therapy、Это Осетинская!、Wealth Optimistic、lexyz888、Roland-Garros、Casey Tuttle、Cricaman2.0、Dude Perfect、Go Beyond Sports、IndiCoder、King of the Palace - Candlepin Bowling、非常識なテニス上達理論、Fitness Style、Cedric Le Mer | ✅ `channel_pass2/gold_pure_f_ch_0922`（31 F，T_hurt=0） |
| 乒乓教学专营（4） | GlobalTTStudio、TT SpinMaster、DynamicTableTennisTV、【卓球動画】WRM-TV [TableTennis] | ✅ `channel_pass2/tt_tutorial_ch`（7 F，T_hurt=0） |
| 乒乓综合频道（18） | ITTFWorld、MobiSportz、Table Tennis Blitz、Tischtennis Bundesliga、ITTF-Africa/Foundation/Oceania、dunia pingpong、TT TRIX、tt.points、janus770、ttlondon2012、LA Ping Pong、RDTTA Newsletter、RMCSport、Table Tennis Akhada… | ❌ **不拉黑**：F 来自其单打/教程/联赛内容，由内容规则处理（同频道有 T） |

> 未纳入的纯 F 频道（n=1 且无法归类）：Alice Zhu、RDTTA Newsletter、Cricaman2.0 类杂项，先观察下批人标。

## 2. 排除舞蹈

本次 gold 里舞蹈 0 条，但**批次内仍有 87 行 / 10.8 h** 残留（`run05_v03` 口径），形态：

- `International Cheer Union` 的 `Hip Hop Doubles / Pom Doubles`（啦啦操，doubles 句式，~140 行）
- `Skating ISU` 的 `Ice Dance Rhythm/Free Dance`（花样滑冰冰舞）
- 标题含 `Dance/dancers`、中文「跳舞/舞蹈」的杂项

→ 新增 `pass2/dance_performance`：`dance* | choreograph | ballet | ballroom | zumba | salsa | bachata | tango | hip hop | k-pop | ice dance | pom doubles | 舞蹈/跳舞/芭蕾/街舞/拉丁舞`（T_hurt=0）。
注意：`International Cheer Union` 同时被频道规则命中（142 行），两条互补。

## 3. 全 F 关键词分析（是否值得做关键词闸门）

### 3.1 本次 gold 中「全 F」关键词（9 个）

| 关键词 | n | 实际内容 |
|---|---|---|
| WTT Grand Smash Men's Doubles | 4 | VGBootCamp 的 Smash 电竞、法国击剑 |
| men's doubles table tennis | 4 | MobiSportz 男子单打（MS 组别代码） |
| ITTF Doubles World Cup | 3 | ITTF 公益/非洲俱乐部联赛/马龙热身 |
| WTT Feeder Women's Doubles | 3 | Rubik's Cube、Unbox Therapy、The Lallantop（爬取污染） |
| men's doubles ping pong | 3 | 男单决赛、RDTTA、德甲联赛 |
| Mixed Doubles | 2 | 斯诺克混双、保龄球 |
| ITTF mixed doubles final / WTT Star Contender Women's Doubles / ping pong doubles match | 1 each | 非洲锦标赛、WTT 男单回顾、接发球教程 |

### 3.2 关键词分层通过率（≤0.3 为低质层，n≥5 才具备统计意义）

| 关键词 | n | T | T 率 | 建议 |
|---|---|---|---|---|
| WTT Star Contender Men's Doubles | 5 | 1 | 20% | 观察 |
| Olympic Table Tennis Mixed Doubles | 8 | 1 | 12% | 观察（多为单打/其他项目混入） |
| professional table tennis doubles | 6 | 1 | 17% | 观察（教程频道污染） |
| WTT Star Contender Mixed Doubles | 13 | 3 | 23% | 观察 |
| table tennis mixed doubles quarterfinal | 8 | 2 | 25% | 保留（结构干净，F 已由规则兜住） |
| Doubles Final | 10 | 5 | 50% | 保留 |
| WTT Finals Mixed Doubles | 13 | 9 | 69% | 保留 |
| Men's Doubles / Women's Doubles / 混双-团队类 | 9–13 | 8–13 | 89–100% | 核心保留层 |

### 3.3 结论：精确匹配 0 T，但闸门有代价（**v0.5.0 已按需求启用**）

1. 这 9 个关键词**精确匹配确实没有 T**（此前误写成「同关键词有 T」，实际情况是**名字相近的兄弟关键词**有 T）。
2. **但它们不是「主题干净」的词**：`keyword` 是 **playlist 标题**，池子里一个 keyword 横跨 78 个频道 230 行，且大量串台/爬取污染落在这个标签下——命中样本可见 `mixed doubles ping pong` 下躺着 `Kuih Tepung Pelita`（马来糕点教程）、`6 types of Forehand Flicks`（教程）；`WTT Champions Men's Doubles` 下躺着 `2018 BNP - Herbert/Mahut v Cuevas/Zaballos`（网球双打）。
3. 人标支持度：8 个关键词中，`table tennis doubles final` 11F 最强，`WTT Champions Men's/Women's Doubles` 9F/4F（另有 7/5 条未标），其余 2–3F。
4. 因此 v0.5.0 **启用了 `keyword_pass2` 闸门**（只匹配 keyword 列、整串锚定），并保留 `rule-of-three` 上界提示：`Mixed Doubles`(2F) 与 `WTT Star Contender Women's Doubles`(2F) 是支持度最弱的两条，如需回退优先摘这两条。

## 4. 规则落地（v0.4.1）

| 规则 | 类别 | 依据 |
|---|---|---|
| `channel_pass2/gold_pure_f_ch_0922` | 21 个非乒乓频道 | 全 F 频道 |
| `channel_pass2/tt_tutorial_ch` | 4 个乒乓教学频道 | 全 F 频道 |
| `pass2/dance_performance` | 舞蹈/啦啦操/冰舞 | 用户要求 |
| `pass2/esports_smash` | Smash Ultimate/SSBU | 全 F |
| `pass2/entertainment_tv` | Roadies/Shark Tank/Rubik/保龄球 | 全 F |
| `pass2/other_sport_misc` | 拳击/跳水/花滑/保龄/啦啦操 | 全 F |
| `pass2/para_multi_event` | boccia/waterpolo/aquatics/pencak/springboard/400m/para-* | 残奥/东运会非乒乓项目 |
| `pass2/singles_ms_ws`（扩） | Men's/Women's Singles、`| MS `、单打/男单/女单/단식 | 全 F |
| `pass2/singles_category_code` | MS/WS/MT/MR 组别代码 + 单人 v/s 单人（用 `[^|/]` 保护双打名字段） | MobiSportz 单打 |
| `pass2/tutorial_footwork`（扩） | how to/tips/secrets/mistakes/drill/lesson | 全 F |
| `pass2/nondoubles_tt` | 热身/公益/非洲俱乐部/直播流 | 全 F |

### 4.1 验证

| 口径 | 结果 |
|---|---|
| gold 259 条：T 误杀 | **0 / 88** |
| gold 259 条：F 清除 | **71 / 90（79%）** |
| gold 未标 81 条中被规则清掉 | 15（多为频道级） |

### 4.2 批次影响

| 阶段 | 行数 | 时长 |
|---|---|---|
| v0.3.0 clean keep | 20,142 | 16,729.7 h |
| **v0.4.1 clean keep** | **19,843** | **16,583.0 h** |
| v0.3.0 口径 ml_keep（text_gov_v02） | 11,859 | 11,336.3 h |
| **v0.4.1 ml_keep（text_gov_v041）** | **9,746** | **9,574.4 h** |

产物：
- 清单 `data/runs/exo_团队协作/machine_0922_pingpong/05_clean/run07_v041/`
- 打分 `data/runs/exo_团队协作/machine_0922_pingpong/06_tools/text_gov_v041/`

## 5. 残余 F 与后续可做项

- `World Table Tennis` 一个频道就占残余 F 的 43/67（官方频道，混排单打/团体/双打）→ 后续按内容规则细分。
- ITTFWorld 混团世界杯里的**单场对局**（`A vs B`，混团赛事下多为单打）→ 需「vs 且无 `/`/`&`」判别，但同类写法在双打里也出现（`Fun Guys vs ...` 被判 T），**暂不加**。
- MobiSportz 双打场次与单打场次同频道，后续可考虑「频道+组别代码」组合键。

## 6. v0.5.0：两批人标合并后的全 F 清理

第二批人标 `853ce1b0`（T78/F154，采自 `text_gov_v01_tlike_c90` 池）与第一批 `72fede47` **视频零重叠**，两批合并 518 条（T166/F244/未标 107）。

### 6.1 合并后的全 F 清单

- **全 F 频道 66 个**（n_t=0 且 n_f≥1），其中 v0.4.1 已覆盖 32 个，新增 34 个。
- **全 F 关键词 8 个**（n_t=0 且 n_f≥2）：`table tennis doubles final`(11F)、`WTT Champions Men's Doubles`(9F+7未标)、`WTT Champions Women's Doubles`(4F+5未标)、`mixed doubles ping pong`(3F+7未标)、`ping pong doubles match`(3F)、`WTT Feeder Women's Doubles`(3F)、`Mixed Doubles`(2F)、`WTT Star Contender Women's Doubles`(2F)。

### 6.2 新增机制：`keyword_pass2`

`pass2` 只匹配 `title_channel`（title+channel），不含 keyword，故新增 **`keyword_pass2`** section（只匹配 `keyword` 列、整串锚定），接入 `core/rules_loader.py`（section 列表）+ `core/certain_noise_clean.py`（新 step `step0b_keyword`，位于 channel 之后、pass2 之前）。其他品类未定义该 section，行为不变。

### 6.3 验证（合并 518 条金标）

| 项 | 结果 |
|---|---|
| **T 误杀** | **0 / 166** |
| F 清除 | 178 / 244（73%） |
| 未标被清 | 38 / 107 |
| 金标 keep | 302（T166 / F67 / 未标69） |

### 6.4 边际删除量（相对 v0.4.1）

| 闸门 | 行数 | 时长 |
|---|---|---|
| `keyword_pass2`（2434 行 / 1413.9 h） | | |
| ├ `table tennis doubles final` | 777 | 641.9 h |
| ├ `WTT Champions Women's Doubles` | 717 | 436.3 h |
| ├ `WTT Champions Men's Doubles` | 820 | 399.9 h |
| ├ `mixed doubles ping pong` | 762 | 210.2 h |
| ├ `Mixed Doubles` | 113 | 76.1 h |
| ├ `WTT Star Contender Women's Doubles` | 123 | 35.6 h |
| ├ `WTT Feeder Women's Doubles` | 149 | 25.2 h |
| └ `ping pong doubles match` | 117 | 22.9 h |
| `channel_pass2` 新增 34 频道 | 1,149 | 610.4 h |
| **合计新增删除** | **3,583** | **2,024.3 h** |

新增频道中体量最大：`table tennis "PingSunday EmRatThich"` 293 行/105.8 h、`janus770` 218/175.4 h、`Table Tennis Blitz` 173/23.1 h、`dunia pingpong` 149/30.8 h、`ttlondon2012` 112/30.4 h、`RMCSport` 32/42.0 h。

### 6.5 阶段影响

| 阶段 | v0.4.1 | v0.5.0 |
|---|---|---|
| clean keep | 19,843 行 / 16,583.0 h | **16,260 行 / 14,558.7 h** |
| ml_keep | 9,746 行 / 9,574.4 h | **8,303 行 / 8,606.6 h** |

产物：`05_clean/run08_v05/`、`06_tools/text_gov_v05/`。

### 6.6 残余风险

- `Mixed Doubles`(2F)、`WTT Star Contender Women's Doubles`(2F)、`WTT Feeder Women's Doubles`(3F)、`ping pong doubles match`(3F) 人标支持度薄，rule-of-three 95% 上界仍高；若要收紧，优先回退这 4 条。
- `janus770`(218 行/175.4 h) 只由 1 条 F 支撑，是新增频道里单位支持度最薄的。
- 剩余 67 条 F 中 43 条来自 `World Table Tennis`（内容混排），属下一轮规则目标。

# Rule Change Log

## 2026-06-03

### Rule: scoring_keyword_tag_strip

- 操作: 修改 `core/scoring.py` 的 `sport_score` UDF，剥离 keyword 中 `-` 开头的标签 token 再参与打
分，不修改 `blacklist_pass2`/`blacklist_r2`（黑名单继续检查完整 keyword）
- 原因: keyword 标签（-virtual -cartoon -game -song 等）是搜索过滤词，不是视频内容，却被
sport_score 当作负信号扣分，导致 15 条真实体育（其他运动如篮球、橄榄球）被误杀
- 来源: equestrian 008_drop_qc FN 分析, 15 条其他体育被 scoring 误扣分丢弃
- 预期: 减少 ~15 FN，Recall 提升，所有数据集受益

### Rule: equestrian_entities

- 操作: 在 `entities.toml` sports 列表新增 FEI, equestrian, dressage, show jumping, eventing,
Hickstead, Kentucky Three-Day, Badminton Horse Trials, Spruce Meadows, CHIO, CSIO,
Global Champions Tour, Longines, Aachen, Calgary Stampede；synonyms 新增 equestrian 同义词组
- 原因: 马术专用术语缺失导致 8 条真实马术赛事被 `no_signal` 丢弃
- 来源: equestrian 008_drop_qc FN 分析, FN 中 keyword 含 FEI/Hickstead/dressage 但不被识别
- 预期: 减少 ~8 FN，所有数据集受益

### Rule: channel_whitelist_exemption

- 操作: 在 `entities.toml` 新增 `[[channel_whitelist]]` 白名单豁免（FEI, HITS Ocala, FIE Fencing, Northern Ireland Open, Dressage），修改 `core/scoring.py` 的 `blacklist_pass2/blacklist_r2` UDF，在黑名单检查前先检查白名单，命中则跳过
- 原因: run02 中 11 条真实马术/击剑/斯诺克赛事被 `step1_blacklist` 误杀（vlog_entertainment/gaming 规则命中），黑名单本身规则合理但覆盖过宽
- 来源: equestrian 010_drop_qc FN 分析, step1_blacklist 误杀样本
- 预期: 救回 ~11 条马术/击剑/斯诺克 FN, 0 Precision 损失, 所有数据集受益

## 2026-06-05

### Rule: athletics_entities

- 操作: entities.toml sports 列表新增 sprint/hurdles/long jump/high jump/triple jump/pole vault/steeplechase/race walk/relay race/world athletics/european championships/ncaa track/usatf/iaaf；synonyms 新增 athletics 同义词组
- 原因: 田径 DROP 中 67% 被 blacklist 误杀 + 33% no_signal，强信号召回 69K 候选 Precision 67.3% 但 blacklist 挡掉 47K
- 来源: athletics_one Phase 8 召回分析, recovered_candidate 69,767 条的 drop_reason 分布
- 预期: 后续田径批次 no_signal 减少，recall 提升

## 2026-06-06 v003

### Rule: gaming_channels_pingpong_two_fp

- 操作: 新增 41 个游戏频道名正则（pass2）
- 来源: pingpong_two run01-run04 FP 分析，跨 4 轮 QC 累计命中 60+ FP
- 预期: 减少游戏频道污染
- 影响范围: 所有运动数据集

## 2026-06-08

### 修复：白名单检查前剥离 keyword 负向标签
- 操作: scoring.py blacklist_pass2/blacklist_r2 UDF 白名单检查前
  用 _strip_keyword_tags 剥离 keyword 里的 -tag 标签
- 原因: keyword 里的否定标签（如 -fifa、-movie）被白名单 regex 误命中，
  导致整条记录豁免黑名单。Johnny Mac's Stadium Ambience 是典型案例：
  kw 含 -fifa，被白名单 FIFA 命中豁免
- 来源: pingpong_two run_r2_test QC 分析，channel whitelist 误豁免根因排查
- 预期: 修复所有 keyword 负向标签导致的误豁免，所有数据集受益
- 实际: 待验证

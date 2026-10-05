# exo 烹饪教学 / 餐饮 · 文本阶段治理（v1，现行）

> 文本-only 阶段依据本文档；视觉 Stage 暂不上。

## 分层

| 层 | 职责 | 约束 |
|----|------|------|
| **黑名单** | 只丢 **与餐饮、烹饪无关** 的确定串台（游戏/音乐 MV/卡通儿歌/硬新闻/真人秀/财经考试/美妆健身等） | 不做「像不像合格 T」判断；家常菜谱等农内式噪声 **不进黑名单**，交 MiniLM |
| **MiniLM** | 在仍偏烹饪的池子里，按 **像不像人标 T** 打分 | 监督=人工 T/F；分数越高越像 T |
| **阈值 τ** | 松紧旋钮：`score < τ` → drop | **τ 越高 → 留下越像 T** |
| **人工** | 唯一可信 pass_rate | keep% / ml_score 不当交付 KPI |

## 操作要点

1. 规则：`02_脚本/categories/exo_cook/rules/blacklist.toml`（**v0.5.7**）
2. 频道清单：`rules/channel_blacklist_pure_f.csv`；keyword 闸：`keyword_blacklist_pure_f_v16.csv`
3. 打分：`02_脚本/tools/score_exo_cook_text.py`（默认 `t_like`；**仅开池/改 τ 时全量跑**）
4. 校准/模型：`models/exo_cook_text_clf_f{_calibration.json,.pkl}`（监督=v0.1–v1.6）
5. 训练：`experiments/exo_cook_text_classifier.py --train`
6. **增量丢金标 F（不全量打分）**：`02_脚本/tools/drop_gold_fail.py keep.csv --qc-snapshot … -o … --replace-keep`
7. 视觉：本阶段跳过

## machine_0916 现行操作点（2026-09-20 · text_gov_v06）

| 项 | 值 |
|----|-----|
| 人标 | v0.1–v1.6 并集金标；交付 KPI 只认独立新人标 |
| MiniLM | **2026-09-20 重训** n=1462；**已全量重打分** τ=**0.50** → ml_keep 243,956 / 48,092 h |
| 后置闸 | 频道→title→keyword→金标F → **86,585 / 18,988 h** |
| **现行 keep** | `text_gov_v06`：**86,585 / 18,988 h**（全量重打分 τ0.50 + 后置闸） |
| 产物 | `06_tools/text_gov_v05/` |

> 主路径 = BL + MiniLM(τ0.50 开池) + 频道/title/keyword 闸 + **新人标 F 增量丢 id**；改模型权重不自动等于重切全量 keep。

### 历史对照

| 版本 | τ | keep | 人标/估 |
|------|---|------|---------|
| BL v0.1 `run01` | — | 1,775,468 / 429,486 h | v0.1 pass **13.8%** |
| `text_gov_v01`（BL v0.2） | 0.32 | 886,363 / 160,876 h | v0.5 验收 **24.7%** |
| `text_gov_v02`（BL v0.3） | 0.36 | 751,722 / 135,207 h | 同池估≈31% |
| `text_gov_v03` | 0.50 | 480,529 / 85,516 h（+chbl→69,228 h） | 同池估≈41% / chbl 估≈87%（偏乐观） |
| v03 cluster_filt（chbl 后） | — | 84,804 / 17,123 h | v1.0 验收 **49.6%** |
| `text_gov_v04`（簇滤+τ） | 0.50 | 96,501 / 18,295 h | 待 c90 |
| `text_gov_v05`（金标 chbl） | 0.50 + n_f>n_t | 225,918 / 44,973 h | — |
| `text_gov_v05`（+n1000 纯F） | 0.50 + chbl | 220,484 / 43,767 h | — |
| `text_gov_v05`（+n500 纯F） | 0.50 + chbl | 163,525 / 34,346 h | — |
| `text_gov_v05`（+v1.6 规则） | 0.50 + chbl/title/kw | ~100,469 / 21,372 h | v1.6 pass **39.6%** |
| `text_gov_v05`（未全量重打分） | 旧 score + 闸 | 100,419 / 21,366 h | — |
| **`text_gov_v06`（现行）** | **重训全量 τ0.50** + 频道/title/kw/金标F | **86,585 / 18,988 h** | 待 c90 |

## 明确不做

- 裸词 `recipe` / `how to make` / `tutorial` / `cooking show` 进黑名单（金标 t_hurt 高）
- 甜点/蛋糕整类进黑名单（T 有 Peekaboo cake）
- 以 keep% / ml_score 冒充交付合格率
- 本阶段视觉 Stage；不合并「餐饮服务场景」原始 Downloads（仍只吃 cook 池）

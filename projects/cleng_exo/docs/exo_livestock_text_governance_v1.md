# exo 渔业牧业 · 文本阶段治理（v1，现行）

> 文本-only 阶段依据本文档；视觉 Stage 暂不上。对齐农业 `docs/exo_agriculture_text_governance_v2.md`。

## 分层

| 层 | 职责 | 约束 |
|----|------|------|
| **黑名单** | 只丢 **与渔牧劳作无关**（影视/音乐/BBQ 料理/故事综艺/儿童科普/渔具测评/综艺吃播等） | 不做「像不像合格渔牧 T」判断；渔牧内休闲钓鱼/howto 等 **不进黑名单**，交 MiniLM |
| **MiniLM** | 在仍偏渔牧的池子里，按 **像不像人标 T** 打分 | 监督=人工 T/F；分数越高越像 T（徒手/撒网/喂养等劳作实拍） |
| **阈值 τ** | 松紧旋钮：`score < τ` → drop | **τ 越高 → 留下越像 T**；无视觉时靠加 τ 抬合格率 |
| **人工** | 唯一可信 pass_rate | keep% / ml_score 不当交付 KPI |

## 操作要点

1. 规则：`02_脚本/categories/exo_livestock/rules/blacklist.toml`（**v0.5+**；双金标 T_hurt=0）
2. 打分：`02_脚本/tools/score_exo_livestock_text.py`（默认读 calibration.`t_like`）
3. 校准：`models/exo_livestock_text_clf_f_calibration.json`；新人标 SRS 上扫 τ → 估 pass_rate / 量
4. 训练：`experiments/exo_livestock_text_classifier.py`（title+channel；监督=v0.1+v1.1）
5. 视觉：本阶段跳过

## machine_0818 现行操作点（2026-09-17 **已归档**）

| 项 | 值 |
|----|-----|
| 人标基线 | `03_qc/` v0.1（v0.3 keep SRS；pass≈16%） |
| 人标验收 | `03_qc/v11_accept_c90/` v1.1（v04 keep；**pass 59.0%**） |
| 黑名单 | **v0.5**；quality→keep **608,560**；双金标 **T_hurt=0** |
| MiniLM | 重训 n=538（OOF AUC≈0.86）；交付 τ=**0.50**（估 pass≈**72%**） |
| **交付** | **139,418 / 33,922 h** → `/Users/muse/data/deliver/渔业牧业_merged_0813-0818_v05_ml_keep_tau050_0916.csv` |
| 归档 | `07_deliver/text_gov_v05_tau050_0917.md` |
| 下轮验收 | `02_sample/text_gov_v05_c90/`（c90 **270**，待标） |

交付 KPI 只认**独立**人标 pass_rate（v04 池已验收 = v1.1 **59%**；τ050 估 72% 偏乐观，须新人标）。

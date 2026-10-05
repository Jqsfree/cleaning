# exo 农业 · 文本阶段治理（v2，现行）

> 文本-only 阶段依据本文档；视觉 Stage 暂不上。有效样本定义仍见 `exo_agriculture_agent_spec_v1.md` §1（真人 + 作物可见交互）。

## 分层

| 层 | 职责 | 约束 |
|----|------|------|
| **黑名单** | 只丢 **与农业无关**（游戏/音乐/卡通/纯娱乐串台/明显非农烹饪综艺等） | 不做「像不像合格采收 T」判断；农内噪声（howto/农机展播/田间口播）**不进黑名单** |
| **MiniLM** | 在仍偏农的池子里，按 **像不像人标 T** 打分 | 监督=人工 T/F；分数越高越像 T |
| **阈值 τ** | 松紧旋钮：`score < τ` → drop | **τ 越高 → 留下越像 T**（越符合要的农业输出）；无视觉时靠加 τ 抬合格率 |
| **人工** | 唯一可信 pass_rate | keep% / ml_score 不当交付 KPI |

## 与旧口径差异

- 旧：黑名单 = certain-noise（含教程/课程等农内硬噪声）+ MiniLM 在 t_hurt≤10% 宁杀勿漏。
- 新：黑名单收窄为 **非农主题闸门**；农内 F 交给 MiniLM；τ 直接表达「有多像 T」，不再绑死 t_hurt≤10% 预算（可用人标曲线选操作点）。

## 操作要点

1. 规则：`02_脚本/categories/exo_agriculture/rules/blacklist.toml`（v0.6+）
2. 打分：`02_脚本/tools/score_exo_agriculture_text.py`（默认读 calibration.`t_like`）
3. 校准：`models/exo_agriculture_text_clf_f_calibration.json`；新人标 SRS 上扫 τ → 估 pass_rate / 量
4. 视觉：本阶段跳过；文本天花板外的歧义交人工或日后 CLIP

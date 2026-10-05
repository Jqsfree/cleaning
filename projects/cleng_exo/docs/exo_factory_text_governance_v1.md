# 工坊制造（exo_factory）文本治理 v1

口径对齐 `docs/exo_agriculture_text_governance_v2.md`，仅字段 **`title`**。

## 有效样本定义（T）

第三人称或第一人称的**工坊/车间劳动与制作过程实拍或工序演示**：木工木旋与家具制作、榫卯拼板、
锻造机加、铸造首饰、缝纫机实操、陶瓷釉、木料修复，以及明确的「How to make/build …」工序教学。

## 排除（F）

- 成品/工艺展示但无劳动主体（`Hand Turned Whiskey Smoker`、`Make a Brass Hammer`）
- 纯设备/工具广告与评测（`STYLECNC Popular wood turning lathe machine`）
- 维修、安装、家装 DIY 串台（补轮箍、装推拉门、砌砖窑）
- 材料/塑料制品产线（`How Plastic Chairs Are Made`）
- 黑匠/锻造厂合集剪辑、企业宣传、AI 教程、ERP/软件口播、剧集（`Production Line S1 E14`）

## 四层（2026-09-22 起）

1. **纯 F 频道闸门**（`rules/blacklist.toml` v0.2.0 的 `[[channel_pass2]]`）
   - 依据：三批人标 803 条 / 724 频道中，**该频道全部人标均为 F（n_t=0）** 的 427 个
   - 清单 `rules/channel_blacklist_pure_f.csv`（带 `n_f` 分档：全 F=427 / 保守 n_f≥2=7）
   - 金标 **T_hurt = 0**（零误杀硬闸门）；全量 clean 剔 41,190 条 / 15,284.7 h
2. **黑名单**：非工坊主题闸门（教程/播客/官方 MV 等 certain-noise）
3. **MiniLM**（`models/exo_factory_text_clf_f.pkl`）：像不像人标 T 的排序
   - 训练 gold：**union3 803**（v1.1 268 + v1.3 269 + v02 268），OOF AUC 0.728；v1.3 池 AUC 0.642
   - `calibration.t_like` = **τ=0.50**（排除 F：`ml_score < τ` 判 F）
4. **关键词救援：已停用**（旧 `production line|cnc|factory floor` 过宽，会把剧集/教程捞回）

## 编码复用

title 向量按 `sha1(title)` 落 `data/assets/embeddings/text/*.duckdb`（fp16，100 万 title ≈ 0.9 GB）。
换 LR 头 / 换 τ / 重打分 / 增量批次**都不再重跑 MiniLM**；fp16 往返对 `ml_score` 影响 < 1e-4（实测 20k 行最大 7.1e-5、判定零翻转）。

## 交付口径（2026-09-22 最新）

链路：quality 1,297,325 / 365,506 h → 纯F频道 −15,285 h + 关键词 −17,737 h → clean **1,201,795 / 332,484 h**
→ v04 排除 F（τ=0.50）→ **keep 106,953 / 30,042 h**（τ=0.40 → 200,129 / 52,953 h；τ=0.60 → 55,636 / 17,144 h）

人标验收（c90 分层 + FPC，独立于训练集）：

| 批次 | 母池 | 加权合格率 | 95%CI |
|------|------|-----------|-------|
| v1.1 | v02 τ=0.60 keep | **54.5%** | 50.0–58.9% |
| v1.3 | v03 τ=0.50 keep | **50.0%** | 45.5–54.4% |

带内 OOF keep 内 T 精：56.6%（τ0.50）/ 60.4%（τ0.60）。**交付 KPI 以独立人标为准。**

## 迭代建议

两批 c90 人标（54.5% → 50.0%）说明**带内区分度是瓶颈**（AUC 0.64–0.70）：τ=0.50 时 keep 内合格率仅
比池基准高 6–7 pp。继续靠 title MiniLM 加 τ 收益递减，建议：

1. 对 `text_gov_v04` keep 再采一轮 c90 人标（分层 + FPC），确认 30,042 h 的真实合格率；
2. 若需 ≥70%，补特征（channel 历史通过率、时长、description/关键词）或换更强编码器（Ettin 类），
   单靠 title MiniLM 很难突破。

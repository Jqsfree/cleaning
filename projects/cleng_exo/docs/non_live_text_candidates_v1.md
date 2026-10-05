# 跨品类非真人形式：通用标题候选层 v0.1.0

本层读取 `title`，提取跨品类的非真人或非现场活动形式线索。A1–A11 是共享候选，B1–B5 是品类条件候选。所有命中目前只写 `review`，不改变输入 Keep/Drop，也不代替画面判定。

## 入口与规则

- 项目统一入口：`02_脚本/pipeline/metadata.py non-live-audit|non-live-scan`
- 独立兼容入口：`02_脚本/pipeline/non_live_text.py audit|scan`
- 规则文件：`02_脚本/core/non_live_candidates.toml`
- 规则处理：`02_脚本/core/non_live_text.py`

先在所有能找到的人标源上回放：

```bash
.venv/bin/python3 02_脚本/pipeline/metadata.py non-live-audit \
  --root /Users/muse/Downloads --root data/runs --root work --root models \
  --root /Users/muse/.cursor/projects/Users-muse-code-datacleng-cleng-exo-cleng-exo/attachments \
  --out work/non_live_text_0928/audit_v06
```

再对任一质量或规则 Keep 做影子扫描：

```bash
.venv/bin/python3 02_脚本/pipeline/metadata.py non-live-scan \
  --input data/runs/exo_construction/machine_0923/05_clean/run09_v024/建筑施工_merged_0923_clean_0928.csv \
  --out work/non_live_text_0928/construction_v024_shadow_v02
```

`audit.json` 记录规则版本、输入文件、重复副本、标签冲突以及每条规则命中的 T/F/U；`rule_hits.csv` 可逐条看人标 T 反例与来源。`scan` 只输出 `candidates.csv` 和 `summary.json`，其中 `decision=review`。输出目录须是新路径，避免覆盖先前报告。

## 首轮回放，2026-09-28

- 发现 218 个候选标注文件路径；跳过 65 个内容完全相同的副本和 7 个文件名含 `weak` 的弱标注文件。搜索范围含下载区、项目工作目录、模型金标目录与附件；候选源仍需人工确认。
- 去重后有 18,794 条 T/F 标注记录；其中 106 个 ID+标题键同时出现 T 与 F，报告保留两侧证据。跨品类定义差异或重复标注都可能造成冲突，不能据此训练单一真值。
- A1（游戏操作画面）命中 F 22、T 1；T 是“开箱 + gameplay”混合标题，不能直接硬丢。
- A2（纯音轨）命中 F 9、T 0；A3（数字画面）F 1、T 0；A4（静态图片）F 1、T 0；A8（无人作业）F 2、T 0。样本量都不足以证明全量精度。
- A5、A7、A9–A11 本轮没有命中；这说明当前词面形式缺少样本支持，不代表这些视频不存在。
- B2（第一视角）命中 F 19、T 6，B3（集锦）F 19、T 13，B4（玩具模型）F 37、T 11，B5（教程）F 516、T 406；不能作全品类硬过滤。
- 在建筑施工 v0.2.4 的 134,914 条规则 Keep 上，影子扫描标记 15,850 条 Review，其中仅 B5 教程就命中 15,133 条；A 组任一命中共 124 条。原 Keep 未改。

## 晋升规则之前

1. 核实每份 `qc_result` 是否真由人工标注。工具对文件名为 `qc_result` 且无 `label_source` 的文件只做“假定人标”，在报告中保留来源；显式 `label_source=human` 的记录单独标记。
2. 给每份人标补上品类、标注定义和批次。不同品类对教学、POV、机械作业的 T 定义不同；同一视频可以出现合法的跨品类标签差异。
3. 对每条规则检查跨品类 T 反例、独立批次支持、标题语境和全量命中分布，再确定是否可自动 Drop。`F=若干且 T=0` 只是观察，不是误删率保证。
4. 标题无法识别未写明的静帧、无人画面或真实可播放性。这些仍需视频/图片证据。

当前层是可复用的候选分析基础，不登记为交付完成，也不自动改写类别黑名单。

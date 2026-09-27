# 体育数据集构建 SOP（V1）

## 核心原则

禁止在项目初期大量编写黑名单、白名单和评分规则。

第一轮目标不是过滤，而是认识数据。

所有业务规则必须来源于质检证据，而不是主观猜测。

规则建立流程必须满足：

样本 → 质检 → 污染分析 → 规则生成 → 重新验证

而不是：

猜测问题 → 编写规则 → 希望有效

---

# Phase 0：数据规范化

目标：

建立统一格式的数据源。

允许执行：

* 字段标准化
* 空值处理
* 编码统一
* 去重
* 时长过滤
* 明显损坏数据删除

禁止执行：

* 体育关键词过滤
* 黑名单过滤
* 白名单过滤
* 打分过滤
* Playlist 判断

输出：

data/runs/001_baseline/baseline.parquet

data/runs/001_baseline/baseline_stats.md

---

# Phase 1：基线数据集（Baseline）

目标：

获得尽可能接近原始数据的数据集。

仅保留通用数据质量处理。

输出：

data/runs/001_baseline/baseline.parquet

data/runs/001_baseline/baseline_stats.md

记录：

* 原始样本数
* 去重数量
* 时长过滤数量
* 空值过滤数量
* 最终保留数量

---

# Phase 2：随机抽样 + LLM 质检

目标：

识别真实污染来源。

Step 1: 随机抽样 (phase2_sample.py)

从 Baseline 中抽取 1000 条样本。

Step 2: LLM 质检 (phase2_qc.py)

调用 Qwen 文生模型 (qwen3.5-flash) 逐条判断 T/F，
自动标注 audit_label 列。

T = 真实体育比赛内容
F = 非体育（游戏/音乐/教程/播客…）

可选：抽样后人工抽查 LLM 标注质量。

输出：

data/runs/002_audit/audit_sample_v1.parquet（含 audit_label）
data/runs/002_audit/audit_stats.csv

命令：

```bash
python3 phase2_sample.py baseline.parquet -o data/runs/002_audit/ --sample-size 1000
python3 phase2_qc.py data/runs/002_audit/audit_sample_v1.parquet -o data/runs/002_audit/ -w 20
```

---

# Phase 3：污染分析

目标：

找到污染来源。

统计：

* 高频关键词
* 高频频道
* 高频 Playlist
* 高频实体
* 高频标题模式

输出：

data/runs/003_analysis/pollution_analysis_v1.md

统计维度:
- 高频污染 keyword（Top 20）
- 高频污染 channel（Top 20）
- keyword 标签污染（后缀如 -shuttlecock -short 等）
- 污染类别分布（若有 audit_category）

必须回答:

1. 最大污染来源是什么？
2. 占比是多少？
3. 是否具有稳定特征？
4. 能否通过规则识别？

禁止直接编写规则。

先完成分析。

---

# Phase 4：规则生成

目标：

依据证据生成规则。

每条规则必须记录来源。

示例：

Rule:
reaction

Source:
audit_v1

Sample Count:
178

Pollution Rate:
91%

规则来源必须能够追溯到具体样本。

输出：

data/runs/004_rules/rules_v1.toml

包含：

* blacklist
* whitelist
* entity_aliases
* score_signals

---

# Phase 5：规则清洗

目标：

使用规则重新生成数据集。

流程：

Pass1：
全局统计

生成：

* playlist_stats
* channel_stats
* entity_stats

Pass2：
规则过滤

依据：

* blacklist
* whitelist
* playlist_stats
* scoring

输出：

data/runs/005_clean/clean_high.parquet

data/runs/005_clean/clean_all.parquet

---

# Phase 6：效果验证

目标：

验证 Phase 5 规则决策 vs Phase 2 Ground Truth。

从 Phase 5 产物中按 keep/drop 各抽 500 条，
与 Phase 2 标注结果比对。

检查：

* 误杀率（False Positive: Phase5=KEEP, GT=F）
* 漏检率（False Negative: Phase5=DROP, GT=T）
* Precision / Recall / F1
* 保留率 / 污染率

记录到 `rule_experiment_log.md`。

输出：

data/runs/006_eval/evaluation_report_v1.md

---

# Phase 7：迭代 + 实验记录

如果污染率仍然较高且未超过 3 轮:

返回 Phase 2。

每轮结束更新 `rule_experiment_log.md`:

| Round | 规则变更 | pass2 | Precision | Recall | 通过率 |
|-------|---------|-------|-----------|--------|--------|

约束:
- ≤5 条/轮
- ≤3 轮
- A 类优先
- 退步则回滚

抽样 → 分析 → 规则生成 → 验证 → 记录

形成迭代闭环。

---

# 项目原则

优先级：

数据质量

>

可解释性

>

性能优化

>

工程优化

在完成有效质检和规则验证之前：

不要优先考虑：

* 多进程
* 分布式
* Spark
* Ray
* 复杂工程化

先证明规则有效，再优化速度。

---

# 规则管理规范

## 单一主规则

所有 Phase 5 使用同一套主规则: `02_脚本/rules/`

## 修改流程

1. `python3 backup_rules.py` — 备份当前主规则
2. 编辑 `02_脚本/rules/blacklist.toml`（等 TOML 文件）
3. 跑 Phase 5 → Phase 6 验证效果
4. 效果退步 → `python3 backup_rules.py --restore 0` 回滚
5. 记录到 `rule_experiment_log.md`

## 迭代约束

- 每轮最多新增 **5 条**规则
- 最多迭代 **3 轮**
- **A 类问题优先**: Precision<60% 或 Recall<50%
- 退步则回滚上一版本

```bash
# 备份
python3 backup_rules.py

# 查看备份列表
python3 backup_rules.py --list

# 回滚
python3 backup_rules.py --restore 0
```

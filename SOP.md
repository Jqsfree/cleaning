# 体育赛事数据集构建 SOP v3

## 核心原则

目标：

QC 抽样通过率 ≥ 70%

在满足上述条件的前提下，最大化保留数据量。

注意：

这里的 70% 指的是最终 Keep 数据集随机抽样后的体育赛事占比（QC Pass Rate），不是 Keep Rate，不是 Retention Rate。

---

# 规则变更记录（强制）

任何规则新增、删除、修改，必须记录：

1. 修改内容
2. 修改原因
3. 来源样本
4. 预期影响

禁止出现无法追溯来源的规则。

记录在与规则同级的 `rule_log.md`：

```
## 2026-06-03

### Rule: r4_esports_channels

- 操作: 新增 5 个 esports 频道正则
- 来源: curling 009_keep_qc FP 分析, 7 FP 0 TP
- 预期: 减少 ~20 FP, 0 TP 损失
```

---

# Phase 1：基础过滤与去重

脚本：

`phase0_normalize.py`

输出：

`001_baseline/baseline.parquet`

处理：

* video_id 去重
* 空标题过滤
* 损坏数据过滤
* 超短视频过滤
* 无效 URL 过滤

---

# Phase 2：共享规则过滤

脚本：

`clean_sports_v3.py`

规则来源：

`02_脚本/rules/`

输出：

`005_clean/run01/`

* clean_all.parquet
* clean_dropped.parquet

---

# Phase 3：KEEP + DROP 随机抽样 QC

脚本：

`chunk_text_qc_v2.py`

要求：

从 clean_all.parquet 中随机抽样 300~500 条，跑 QC。

从 clean_dropped.parquet 中随机抽样 300~500 条，跑 QC。

统计：

* KEEP 中：体育赛事数量 / 非体育数量 → QC Pass Rate (Precision)
* DROP 中：体育赛事数量 / 非体育数量 → FN Rate (误杀率)

---

# Phase 4：决策

如果 **Precision ≥ 70%** 且 **FN Rate 可接受**：

冻结当前规则。

进入 Done。

完成项目记录，产出交付文件。

如果 **Precision < 70%**：

→ Phase 5 (加数据集黑名单，减少 FP)

如果 **FN Rate 偏高**（DROP 中真实体育占比显著）：

→ 补充共享 `02_脚本/rules/entities.toml`（加缺失的体育实体/频道/同义词），降低 `no_signal` 误杀

→ 共享规则修改后回到 Phase 2 重新过滤

---

# Phase 5：构建数据集专属规则 (FP 黑名单)

仅处理 FP（非体育被放行）。

规则来源仅允许：

* false_positive
* false_negative
* audit_sample
* QC 结果

禁止凭主观感觉新增规则。

输出：

`data/runs/{sport}/rules/`

---

# Phase 6：重新过滤

脚本：

`clean_sports_v3.py`

规则：

shared rules + sport rules

输出：

`005_clean/run02/`

---

# Phase 7：再次随机抽样 QC

样本：

500 条

重新计算：

QC Pass Rate

---

# 停止条件

满足任意一个：

1. QC Pass Rate ≥ 70%
2. 已完成 3 轮规则迭代

立即停止。

禁止无限优化规则。

---

# 项目记录（强制）

每完成一个 Phase，必须立即写项目记录。

禁止积压记录。禁止事后补写。

记录位置：

`data/runs/{sport}/project_log.md`

模板：

```
## 2026-06-03 15:30

### Phase 3 - KEEP 抽样 QC

输入：
005_clean/run01/clean_all.parquet

样本：
500

结果：
QC Pass Rate: 57.0%

结论：
未达到 70%

下一步：
进入 Phase 5 分析 false_positive
```

---

## 运动专属规则管理

运动专属规则 (`data/runs/{sport}/rules/`) 仅作为**临时实验层**。

约束:

1. 必须经过 QC 验证（Precision ≥ 70%）后方可推广
2. 验证通过后，**必须合并至共享规则库** (`02_脚本/rules/`)
3. 合并后**删除数据集专属规则目录**
4. **禁止长期保留独立运动规则体系**

推广流程:

```
Phase 5 专属规则 → Phase 6 过滤 → Phase 7 QC ≥ 70%
  → promote_rules.py (备份 + 合并)
  → 删除 data/runs/{sport}/rules/
  → 所有数据集受益
```

## FN 处理原则

当 DROP 中出现较高比例的真实体育（FN，误杀）时:

1. **禁止修改黑名单** — 黑名单控制 Precision，松了会引入污染
2. **优先补充共享信号到 02_脚本/rules/entities.toml**:
   - 缺失的体育实体词 → 加入 lexicon
   - 缺失的同义词 → 加入 synonyms
   - 缺失的频道 → 加入 whitelist channel 列表
3. **共享规则修改 → 所有数据集受益**
4. 目标: 降低 `no_signal` 与 `score_threshold` 导致的误杀
5. 验证: 重跑 Phase 5 + Phase 6，对比 Recall 变化

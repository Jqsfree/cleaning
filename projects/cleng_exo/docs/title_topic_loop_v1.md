# exo 标题主题闭环 v1

> 历史设计。当前运行和验收入口已升级，参见 [metadata 管线 v2](exo_metadata_pipeline_v2.md)。新 run 必须预先声明验收目标；旧 needs_iteration/零缺陷规则及本页旧命令的默认验收说明已被 v2 替代。

本轮依据：用户明确现有 Downloads 人标针对 metadata；直接复用 qc_result T/F，不解释成视频画面标注。农业已于 2026-09-16 结束；本模块只对齐历史资产，不改动农业交付。

## 目标与口径

保留标题符合当前品类人标边界的数据，排除串台主题。只以标题为模型输入；channel 用于划分独立训练/校准/测试组，keyword 仅保留为元数据。文本层不调用图像模型。

topic_label 是导入后统一的 T/F/U 字段，qc_result 原值和源文件均保留。它直接承接用户确认的 metadata 人标，不要求重复标注，也不声称所有 F 都是跨品类串台；人标可能包含更细的主题边界。

严格区分：
- 人标 T/F：验收与训练依据。
- MiniLM 分数、规则命中、LLM 判断：机器候选判断。
- keep：待独立人标验收的候选保留池。
- review：标题信息不足、证据冲突或模型未达校准门槛；不会混入 keep。
- drop：机器候选拒绝池；同时抽检人标 T 损失。

## 新入口

`02_脚本/tools/batch_ops/topic_loop.py`，独立于已完成农业流水线。
作为现有 `02_clean` 或 `text_gov` 后的可选文本步骤，输出独立新目录；任何旧目录都不覆盖。

### 1. 导入现有人标

```bash
.venv/bin/python3 02_脚本/tools/batch_ops/topic_loop.py import-gold \
  --source-dir "$HOME/Downloads/humen-烹饪餐饮" --category exo_cook \
  --out work/topic_cook_round1/gold
```

仅读 *_qc_result.csv。空标、“无法播放”等非纯 T/F/U 标签排除；同 video_id 或同标准化标题标签冲突进入 conflicts.csv。按频道隔离 60/20/20 开发、校准、测试集。已有闭环使用过的人标只能作为历史离线测试，不能冒充当前池全新验收。

### 2. 训练并独立测试（人工标签 → 标题 MiniLM + LR）

```bash
.venv/bin/python3 02_脚本/tools/batch_ops/topic_loop.py train \
 --train work/topic_cook_round1/gold/train.csv \
 --calibration work/topic_cook_round1/gold/calibration.csv \
 --policy 02_脚本/categories/exo_cook/rules/title_topic_v1.json \
 --output work/topic_cook_round1/candidate.joblib

.venv/bin/python3 02_脚本/tools/batch_ops/topic_loop.py benchmark \
 --model work/topic_cook_round1/candidate.joblib \
 --test work/topic_cook_round1/gold/test.csv --out work/topic_cook_round1/holdout
```

校准门槛：T/F 各至少20条；候选阈值至少覆盖20个校准样本，组内经验精度至少95%。找不到时关闭相应自动通道（keep阈值1.01 / drop阈值-0.01），不擅自回退到宽松阈值。这个门槛是保守默认工程设置，不等于用户保证或总体置信结论。

### 3. 跑新一轮 + 三池随机抽检

```bash
.venv/bin/python3 02_脚本/tools/batch_ops/topic_loop.py run \
 --input INPUT.csv --policy 02_脚本/categories/exo_cook/rules/title_topic_v1.json \
 --model work/topic_cook_round1/candidate.joblib --out work/topic_cook_round1/round
```

输出 keep.csv / drop.csv / review.csv、三池 audit_*.csv 和 manifest.json。
每池默认简单随机抽取最多271条；标题人标列为空，不偷带模型标签。
小池全部抽取。每池单独估计，不把三池不同比例样本混算总体精度。
重复 video_id 保留首条并计数；坏行直接报错，不静默丢弃。
不指定模型时只有“初始候选规则”分流，不能视为已经上线的主题过滤器。

### 4. 对难例启用标题语义复核

```bash
.venv/bin/python3 02_脚本/tools/batch_ops/topic_loop.py verify \
 --input INPUT.csv --gold work/topic_cook_round1/gold/train.csv \
 --policy 02_脚本/categories/exo_cook/rules/title_topic_v1.json \
 --out work/topic_cook_round1/semantic

.venv/bin/python3 02_脚本/tools/batch_ops/topic_loop.py run \
 --input work/topic_cook_round1/semantic/verified.csv \
 --verifier-report work/topic_cook_round1/semantic/verifier.json \
 --policy 02_脚本/categories/exo_cook/rules/title_topic_v1.json \
 --out work/topic_cook_round1/semantic_audit
```

复用已有 DASHSCOPE 配置；只发送标题、品类定义与检索到的人工开发样本。每条 T/F 必须附标题原文依据；无依据、接口失败、解析失败均进 review。
缓存键包括模型、提示版本、人标示例库、品类策略和标题，不重复付费请求相同结果。
禁止把示例库同标题样本作为独立复核输入；正式验收必须使用未参与开发的新样本。
该入口可产生费用，批量前先小规模比对人标。
语义复核仍是机器判断，不能成为人工标签。

### 5. 回收原人标或新一轮人标并评估

人工表需 topic_label=T/F/U、label_source=human、reviewer（可记录原文件审核来源）。已存在且未参与开发的人标可按 video_id 和标题一致性连接回抽样表，无需重标。

```bash
.venv/bin/python3 02_脚本/tools/batch_ops/topic_loop.py evaluate \
 --round work/topic_cook_round1/round --labels HUMAN_AUDIT.csv --target 0.95
```

输出 human_topic_evaluation.json 与 human_topic_errors.csv。
默认验收：抽检完整，keep 未观察到 F/U，且 T 比例90% Wilson置信下界达到95%。同时报告 drop 中人工T，不将其掩盖为“清得干净”。未标完为 pending_labels；不达标为 needs_iteration。
keep率不是人工合格率。任何抽样都不能证明全池绝对零串台。

### 6. 错例反馈与规则回归

```bash
.venv/bin/python3 02_脚本/tools/batch_ops/topic_loop.py propose \
 --labels HUMAN_DEVELOPMENT.csv \
 --policy 02_脚本/categories/exo_cook/rules/title_topic_v1.json --output proposals.json

.venv/bin/python3 02_脚本/tools/batch_ops/topic_loop.py revise \
 --labels HUMAN_DEVELOPMENT.csv --proposals proposals.json \
 --policy 02_脚本/categories/exo_cook/rules/title_topic_v1.json \
 --select RULE_NAME --output title_topic_v2.json
```

复用 core.text_boundary 提案挖掘，但只输入人标标题，channel清空；提案不会自动激活。
选中的新规则必须在所提供的人标全集上命中至少3个F且不伤T/U，生成新版本；使用过的人标标题纳入开发指纹，不可用于新版本验收。
模型模式以人标校准分类器为准，规则不偷偷覆盖模型；想更新模型必须显式 train。规则模式的冲突仍进review。

### 7. 两轮验收后封闭本版本

```bash
.venv/bin/python3 02_脚本/tools/batch_ops/topic_loop.py close \
 --rounds ROUND_A ROUND_B --output text_acceptance.json
```

两轮必须同一策略/模型/语义复核配置、相同目标、均通过人标验收，keep审计标题不得重复。改规则、模型或示例库后重新验收。不能因“挖不到新规则”宣布清洗合格。
标注完成前可完成软件闭环，不能提前完成质量验收。

## 本轮已有资产对齐

Downloads 人标去重后的标题数：
- 农业2144：历史对齐，未重训/重跑交付。
- 烹饪1191、亲子1557、渔牧530、商业服务1365：候选开发与独立频道测试。
- 娱乐当前未找到同目录已有人标，不用LLM标签代替金标训练。

初始实验位于 work/title_topic_loop_0918；正式摘要归档 docs/title_topic_loop_initial_results.json。
当前四个候选MiniLM模型在95%经验精度+最少20条的校准门槛下均未开放keep，不能当作达标生产模型。
本轮不自动替换现有生产模型；后续以新轮人工结果推动迭代。

## 已执行验证

11项回归测试通过。四类候选模型及金标已归档至 models/title_topic_loop_v1_0918/。
烹饪独立测试前50条的语义试验：keep=14（人标T7/F7）、drop=32（T5/F27）、review=4（T3/F1）；0接口错误。
完整闭环已运行并返回 needs_iteration，未发布。该50条是诊断子集，不是当前全量池随机验收。
错例：data/runs/exo_cook/machine_0916/06_tools/title_topic_loop_v1_0918/human_topic_errors.csv。
农业交付和现有生产模型保持原状。

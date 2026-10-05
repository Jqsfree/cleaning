# exo metadata 管线 v2：入库画像、已知人标回放与发布门禁

2026-09-21。实现入口仍为 `02_脚本/tools/batch_ops/topic_loop.py`，共用当前 topic_loop，不再复制每品类脚本。本版只处理 metadata；历史生产模型、原始数据和历史交付未被重写。

## 已实现的工作流

输入 → 严格解析/入库画像 → 品类规则或现有本地模型 → keep/review/drop → 冻结候选和验收目标 → 盲审 → 判决 → 发布或专项。

- 每次 run 必须使用新的输出目录；自动保存 intake/profile.json、stage_eval.json、manifest.json 与三池。可用 `--reference-gold` 自动对相同分类器做历史人标回放。
- 初筛信息缺失不会变成凭空的语义结论。空输入、空 ID、重复 ID 对应冲突 metadata、坏 CSV 阻塞本轮；完全一致的重复 ID 保留首条并计数。CSV/TSV/Parquet 严格读取，不静默跳行。
- 画像包含字段缺失、字符长度抽样分位数、提供的语言字段、频道集中度、有效时长和重复情况。还没有自动语言识别、token 截断测量、embedding 漂移，报告明确标注未测。
- 新 audit.csv 打乱池顺序，清空所有人标字段，不包含模型决定/分数；原有 audit_keep/drop/review.csv 仅为兼容导出。正式盲审使用合并 audit.csv。
- `--target` 在 run 前冻结；未配置目标可以诊断，不能发布；evaluate 不允许临时降低目标。
- 每池预先固定一次随机抽样；全部标完才判断。keep 采用 90% Wilson 区间：下界达标为 accepted，上界低于目标为 failed_acceptance，中间为 inconclusive。全池被标注时按实际总体比例判定；U 不计为 T。空 keep 禁止通过。
- 证据不足没有自动反复抽样；需要事先制定新的抽样协议。本版也没有推断总体 T 损失上限，只单列 drop 中 T/U；如项目要求 T 损失门槛，应补齐对应的分层抽样与验收配置再上线该要求。
- 专项 run 必须传 `--parent-round`：父轮真实人验失败、品类一致、输入内容哈希一致。父轮用于改规则的标注标题被记为开发数据，不再冒充新验收。已暴露条目若再次入选新审计会阻塞，需要新独立证据或明确的人工裁决分区；本版不偷偷排除它们来提高通过率。
- run/propose/revise 不自动生成并激活新业务规则。next 返回所需动作；用户提交已审阅规则/模型后才执行专项。
- `release` 重新核实人标、冻结目标、候选与各池内容哈希；仅 candidate 模式的 accepted 版本可原子导出。历史回放不能发布。修改池、标签、计数/样本配置或候选身份都会阻止复用旧验收。
- exo 的旧 orchestrate deliver 不再 glob 选择 keep 或回退 quality，要求 `--accepted-round`；旧 lot_accept.prepare 不允许把 pending exo 写入交付区。其他非 exo 品类行为保留。旧交付目录非空不再被认为具有新门禁的发布证据，不改写历史交付事实。

## 日常命令

以下命令从仓库根运行；路径参数替换为实际批次。TARGET 使用业务预先确定的验收目标，不能用模型分数替代。

```bash
.venv/bin/python 02_脚本/tools/batch_ops/topic_loop.py profile \
  --input INPUT.csv --category exo_cook --out work/cook_intake

.venv/bin/python 02_脚本/tools/batch_ops/topic_loop.py run \
  --input INPUT.csv --policy 02_脚本/categories/exo_cook/rules/title_topic_v1.json \
  --reference-gold models/title_topic_loop_v1_0918/exo_cook/gold/train.csv \
  --target TARGET --out data/runs/exo_cook/machine_NEW/06_tools/metadata_round01

.venv/bin/python 02_脚本/tools/batch_ops/topic_loop.py evaluate \
  --round data/runs/exo_cook/machine_NEW/06_tools/metadata_round01 \
  --labels HUMAN_AUDIT.csv

.venv/bin/python 02_脚本/tools/batch_ops/topic_loop.py next \
  --round data/runs/exo_cook/machine_NEW/06_tools/metadata_round01

.venv/bin/python 02_脚本/tools/batch_ops/topic_loop.py release \
  --round data/runs/exo_cook/machine_NEW/06_tools/metadata_round01 \
  --out data/runs/exo_cook/machine_NEW/07_deliver/metadata_release01
```

run 不传 --model 时只是当前 JSON 标题规则候选，不代表它已经达到质量要求；本轮回放证明烹饪初始规则明显不够。传本地模型使用现有 topic bundle。统一入库不会自动替换其他脚本的旧 quality 步骤。

退出码：1 为配置/执行错误；profile 返回 2 表示输入被阻塞；evaluate 返回 2 表示候选尚未通过验收。run 返回 0 只代表执行完毕，仍需人验。regression 的 evaluate 返回 0 表示回放完成，仍禁止 release。

若用旧 orchestrate 发布，还须注册与候选一致的输入快照；不匹配时会拒绝，不能将旧批次原始输入与另一个候选混用。建议新 metadata 批次先直接使用上述 release 入口。

## 现有人标怎样用于实验

现有标准化人标位于 `models/title_topic_loop_v1_0918/{category}/gold/`，包含来源和原始标签，已有按频道划分的 train/calibration/test。本版核实 split 间 ID、标准化标题和非空频道不重叠；U 不成为训练正例。历史暴露情况不能完全排除，所有报告标为 retrospective，不授予新批次交付资格。

```bash
.venv/bin/python 02_脚本/tools/batch_ops/topic_loop.py replay \
  --gold models/title_topic_loop_v1_0918/exo_cook/gold/test.csv \
  --policy 02_脚本/categories/exo_cook/rules/title_topic_v1.json \
  --out work/cook_rules_replay

.venv/bin/python 02_脚本/tools/batch_ops/topic_loop.py compare \
  --gold-dir models/title_topic_loop_v1_0918/exo_cook/gold \
  --category exo_cook --methods tfidf embedding --encoder LOCAL_ENCODER_SNAPSHOT \
  --fields title --target 0.95 --out work/cook_model_comparison
```

compare 的 0.95 是本次实验约束示例，不是用户交付 SLA。TF-IDF 词表只拟合 train；同样的 LR 配置，calibration 选阈值，test 仅计算结果；无合格阈值时 keep/drop 为 null、进入 review。0.5 二分类结果仅诊断，不是可交付策略。模型本地加载，未调用付费 API、不下载 E5。

可比较 title/channel/description，但存档 gold 当前不含 description，因此不能据此声称已完成简介增益实验。E5 路径须可识别为 e5 以启用 query: 前缀；其他命名方式需要先扩展显式编码器配置。更换模型后不能复用旧分类头或阈值。

## 阶段评估口径

`core/stage_eval.py` 支持连续阶段 trace：每层 keep 进入下一层，drop/review 终止；缺决定、重复 ID、标题指纹不符、前层已终止记录重新出现都会失败。每层报告阶段入口分母与全链累计结果：F 清除率、T 误删率、drop 中 T 占比、keep 纯度（U 单列）、T 保留覆盖、review 比例。零分母为 null。

难例/便利样本的比例不直接外推全批。参考回放不是当前批次验收。只标 keep 无法估前层误杀；需要时从不同首次 drop 层随机抽样并保存抽样概率。新语言、长文本、字段缺失等切片以及按抽样设计加权的误杀门槛仍是后续扩展项。

## 本轮实际实验与边界

四品类共 4,643 条存档人标，测试集共 926 条；标题作为相同输入，比较 TF-IDF 与本地 MiniLM。逐类数值见同目录 exo_metadata_regression_20260921.json。

- 烹饪 AUC：TF-IDF 0.6028，MiniLM 0.6713。
- 亲子 AUC：TF-IDF 0.6098，MiniLM 0.6063。
- 渔牧 AUC：TF-IDF 0.8272，MiniLM 0.8460。
- 商业服务 AUC：TF-IDF 0.7011，MiniLM 0.7559。

在此次实验的 95% 目标、90% Wilson 下界、至少 20 条校准支持条件下，两方法四品类均未开启自动 keep/drop；小校准集和有限区分能力都可能导致这一结果，不能简单归因于模型差。它更不能证明降低阈值后就可交付。

烹饪 JSON 初始规则在 255 条历史测试人标上的实际回放：keep 121，其中 T=48/F=73，纯度 39.67%；review 134。该比例仅针对这一历史集合。回放全链正确返回 retrospective_replay_not_acceptance，其诊断质量判决为 failed_acceptance，禁止发布。

本轮不将实验胜者自动替换生产模型，不修改 gold 标签、不重写历史原始/交付数据。先获得可复现的诊断能力与发布约束，再依据错误类型补标签或优化特征。


### 字段消融补充

同一历史 split 加入 channel 后，TF-IDF/MiniLM 的 AUC 分别为：烹饪 0.5787/0.6737、亲子 0.6209/0.6170、渔牧 0.8714/0.8559、服务 0.7047/0.7430。频道对不同品类和方法影响不同，不应全局固定加权。这是单组历史切分的点估计，不能据微小差距宣称统计显著或直接上线。

渔牧校准集只有 50 条 T：即使这 50 条全被正确挑出，90% Wilson 下界也只有约 94.87%，低于实验目标 95%。因此关闭 keep 不全是模型问题，也包含证据数量不足；不要为得到非零自动量而偷偷降低要求。

### 软件验证

改动前全量测试：102 通过、1 失败。改动后：131 通过、1 个相同的既有失败；本次相关测试 40 项通过。既有失败为 test_exo_medical_text_classifier.py 的描述字段预期与当前 title+channel 实现不一致，本轮没有为凑全绿更改医疗模型输入口径。

候选绑定也覆盖计数、抽样配置、开发数据指纹和输入哈希；批次 manifest 改变身份/输入路径时不再沿用旧阶段状态，调用方会停止。文件与数据完整性是发布条件，不只是日志告警。

### 第二轮诊断与规则提案修正

详见 [metadata 错误诊断与简介消融](exo_metadata_diagnostics_20260921.md)。新增 `enrich-gold`、`diagnose`，以及 `propose/revise --validation-gold`。历史 gold 可以从已记录的人标源恢复简介，但实际覆盖仅 166/4,643；cook/service 简介消融未解决自动阈值关闭问题。对比报告现已测量真实 token 长度；本次输入未截断，这不代表所有未来批次都无截断。

revise 现在要求独立频道回归及对应输入哈希，并将提案开发、回归的所有标题记为开发证据。过去仅靠同样本“3 F、零 T/U”生成规则的用法不再受支持。没有通过验证的候选继续停留在诊断阶段，不改动已使用的品类规则。

### v3 统一入口与批次管理

统一入口升级为 `02_脚本/pipeline/metadata.py`，原 topic_loop 工具仍兼容。新增 start/status/retry/specialist/publish/catalog/doctor；抽样可以固定各池数量，并可预设保守误删门槛。旧文档中“未实现全链损失估计”的描述是 v2 历史限制，当前实现及统计口径见 [基础设施 v3](exo_metadata_infrastructure_v3.md)。批次级常用命令以仓库 [README](../README.md) 为准。

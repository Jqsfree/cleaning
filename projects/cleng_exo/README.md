# exo metadata 数据清洗

本项目处理标题、频道等 metadata。统一入口是 `02_脚本/pipeline/metadata.py`；品类策略通过 `--policy` 选择，同一套命令适用于不同品类。

## 先查看项目

```bash
.venv/bin/python 02_脚本/pipeline/metadata.py doctor
.venv/bin/python 02_脚本/pipeline/metadata.py catalog
.venv/bin/python 02_脚本/pipeline/metadata.py --help
```

`doctor` 检查品类配置；`catalog` 汇总批次、已登记路径、候选和下一步。默认索引不读取全部大文件，状态会明确标为未核验。需要校验内容时运行 `status --batch-root PATH` 或 `catalog --verify`。

## 新批次

以下 0.95 合格率、0.05 损失上限和抽样量都是演示参数；运行前按实际业务约定确定。损失门槛未配置时，只执行合格率验收，不能声称已验收误删率。

```bash
.venv/bin/python 02_脚本/pipeline/metadata.py start \
  --input INPUT.csv \
  --policy 02_脚本/categories/exo_cook/rules/title_topic_v1.json \
  --batch 20260921_batch01 --target 0.95 --max-t-loss 0.05 \
  --audit-sizes 271 271 271
```

命令返回批次路径、候选路径和 `audit.csv`。不传 `--model` 时使用规则候选；现有 cook 初始规则尚未达到交付要求。`--reference-gold` 可添加历史回归，但参考样本会标记为开发数据。

```bash
.venv/bin/python 02_脚本/pipeline/metadata.py evaluate \
  --round ROUND_PATH --labels HUMAN_AUDIT.csv

.venv/bin/python 02_脚本/pipeline/metadata.py status --batch-root BATCH_ROOT

# 仅在人工验收失败后，用审阅过的新策略执行专项；沿用原批次验收目标
.venv/bin/python 02_脚本/pipeline/metadata.py specialist \
  --batch-root BATCH_ROOT --policy REVISED_POLICY.json

# 仅在执行中断/失败时，按原配置重试；不重新抽取已完成候选的验收样本
.venv/bin/python 02_脚本/pipeline/metadata.py retry --batch-root BATCH_ROOT

# 验收通过后，校验原始证据、导出精确候选并登记交付位置
.venv/bin/python 02_脚本/pipeline/metadata.py publish --batch-root BATCH_ROOT
```

`start` 为同品类已登记的相同输入快照阻止重复创建 candidate；请使用原批次的 status/retry/specialist。实验可显式使用 `--purpose regression`，但禁止发布。

## 目录与状态

- `data/runs/{category}/{source}_{batch}/manifest.json`：批次身份、输入哈希、阶段和交付路径。
- `06_tools/metadata/attempt_0001/operation.json`：执行配置、开始/结束或失败记录。
- 同目录 `candidate/`：冻结三池、抽样计划、盲审模板、人验记录。
- `07_deliver/metadata_{candidate_id}/`：通过验收的精确输出及发布证明。
- `work/`：可重建的诊断、实验和索引快照；不是原始数据的唯一存储位置。

历史批次不会自动升级成验收通过。索引不会移动、删除或复制生产大表。当前候选三池仍保留完整 metadata；大表改为单份底表加窄决策记录，属于后续存储迁移。

原 `tools/batch_ops/topic_loop.py` 保留为兼容入口，与新入口调用同一个 CLI 实现。`profile / diagnose / enrich-gold / replay / compare / propose / revise` 仍可调用。

详细实现、统计口径与兼容变化见 [基础设施说明](docs/exo_metadata_infrastructure_v3.md)，历史实验见 [错误诊断](docs/exo_metadata_diagnostics_20260921.md)。

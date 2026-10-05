# metadata 第二轮：错误诊断、简介消融和规则提案验证

2026-09-21。这里的结果来自现有历史人标，是开发与回归证据，不是当前生产批次合格率。原始标签、原始数据、生产模型和已存在的规则未被改写。入口仍共用 `02_脚本/tools/batch_ops/topic_loop.py`。

## 本轮解决的问题

新增 `enrich-gold`：只从每条 gold 记录的 `label_file` 找回 metadata，核对 ID、标题、频道、原始 T/F/U 标签，保留 split 顺序、标签和审阅来源，另存副本。源文件或原 gold 字段冲突会中止；源文件及输出均记录完整 SHA256。缺失简介如实保留为空，不生成补写内容。

新增 `diagnose`：记录每条规则/诊断特征的 T/F/U 命中、当前 keep 上的反事实删除影响、频道与标签文件切片、误保留/误删明细。`cases.csv` 包含全部对照，`errors.csv` 包含错例；`cause_hypothesis` 和 `diagnosis_reviewer` 保持空白，自动信号不冒充人工拒绝原因。probe 只允许 title/channel/description，不能使用标签或标签文件名作为特征，也不会写入实际规则。

修正 `propose → revise`：同一批标注上“至少 3 条 F、零 T/U”仅是发现候选的条件。提案明确区分 `development_addable` 与 `eligible_for_revision`。revise 必须提供与提案绑定的 `--validation-gold`；开发/验证的 ID、标准化标题、非空频道不得重叠。验证命中 T/U 或 F 支持不足会阻止生成修订候选；验证结果按实际规则优先级统计 keep/drop/review 变化。3 条支持只是回归最低要求，不代表统计上达到任何交付纯度。

提案与验证用过的标题全部进入修订规则的开发指纹；旧提案缺少来源哈希时必须重新生成。通过回归生成的仍是 candidate，必须走冻结候选和新人工批次验收。规则候选不会自动部署。

本地 embedding 对比报告增加真实 tokenizer 长度统计，记录各 split 在模型截断前的 token 分位数、最大值和超限条数。

## cook 的 73 条误保留说明了什么

基线 JSON 规则在 255 条历史测试样本上 keep=121（48 T、73 F）。同一集合有 11 个频道同时出现 T/F；这不等于标注冲突，只说明频道不能直接代表单条是否合格。

在当前 keep 中直接加黑名单的代价：

- `recipe`：去掉 53 条 F，同时误删 22 条 T。
- `street food`：去掉 2 条 F，同时误删 13 条 T。
- 营养/健康词：去掉 12 条 F，同时误删 1 条 T；但在 train 上对应 7 F / 4 T，不能据测试集的好看比例倒推为稳定规则。
- `promo`/预告词：去掉 3 条 F，同时误删 1 条 T；train/calibration 当前 keep 上均无命中，缺少独立支持。

73 条 F 中只有 2 条有简介。17 条出现营养健康、预告、节目分集、户外等诊断信号，其余 56 条没有这些信号；这些分组只是排查入口，不是确认的 F 原因。不能将“是菜谱”推断成 F，也不能将“有 chef、cooking、recipe”直接推断成符合人标的 T。

所有测试错例现已用于开发排查，后续不得将该 test 集重新称为未接触过的独立验收集。没有根据这 73 条错例激活新关键词或频道黑名单。

## 简介是否有帮助

从记录中的原始人工标注文件恢复后，四品类 4,643 条样本仅 166 条有简介（约 3.58%）：cook 31/1,191，parent 3/1,557，livestock 10/530，service 122/1,365。parent 的训练集简介全部为空。

对 cook 和 service 做同 split、同算法参数的消融，比较 title+channel 与 title+channel+description。简介只去 URL、规范空白、限制前 1,500 字符，未加入采集词、标签、来源文件名或人工结果：

- cook：TF-IDF AUC 0.5787 → 0.5817；MiniLM 0.6737 → 0.6724。
- service：TF-IDF AUC 0.7047 → 0.7138；MiniLM 0.7430 → 0.7398。

这些是单组历史切分的点估计，不能声称显著改善。四个实验组合的自动 keep/drop 阈值仍关闭（实验约束：95% 目标、90% Wilson 下界、至少 20 条校准支持），未替换生产模型。parent/livestock 没有追加简介模型实验，原因是可用简介极少。

本次 cook/service 的 train、calibration、test 都没有超过本地 MiniLM 的 128-token 上限；service 最大 120 token，cook 最大 81 token。这里暂时没有“模型截断导致效果差”的证据。低简介覆盖限制了实验结论，不能外推为完整简介永远无用。

## 规则生成方式的实测缺陷

旧挖掘方法在 cook train 找到 14 条候选；到独立频道的 calibration，1 条命中人工 T（`bread recipe`），13 条未达到最低 F 支持，因此新验证门禁全部阻止修订。若回看已暴露 test，有 3 条候选命中 T，10 条完全没有命中。

问题在于：从 F 中发现词后，再用同一批样本证明它“纯 F”，只证明局部拟合；少量同频道重复样本也不等于跨频道稳定性。解决路径是独立分组验证、T/U 反例保护、实际决策变化回放、开发数据暴露记录，以及最终批次验收。规则不足时保留 review，不为了增加自动处理量强行上线。

## 通用调用

从仓库根执行。CAT、GOLD、OUT 等是需要替换的路径，不是项目新增的固定目录层级。enrich-gold/diagnose/compare 每次使用新输出目录；所有产物集中在一个实验根下。

```bash
.venv/bin/python 02_脚本/tools/batch_ops/topic_loop.py enrich-gold \
  --gold-dir models/title_topic_loop_v1_0918/exo_cook/gold \
  --category exo_cook --out work/EXPERIMENT/enriched_gold

.venv/bin/python 02_脚本/tools/batch_ops/topic_loop.py diagnose \
  --gold work/EXPERIMENT/enriched_gold/test.csv \
  --policy 02_脚本/categories/exo_cook/rules/title_topic_v1.json \
  --probes work/exo_metadata_diagnostics_20260921/cook_diagnostic_probes.json \
  --out work/EXPERIMENT/diagnosis

.venv/bin/python 02_脚本/tools/batch_ops/topic_loop.py propose \
  --labels work/EXPERIMENT/enriched_gold/train.csv \
  --validation-gold work/EXPERIMENT/enriched_gold/calibration.csv \
  --policy 02_脚本/categories/exo_cook/rules/title_topic_v1.json \
  --output work/EXPERIMENT/proposals.json
```

只有 `eligible_for_revision=true` 的规则才可进入 revise，且仍需语义审阅；其 `--validation-gold` 必须与 propose 相同。不要从同一批 F 中反复挑选词并反复更换验证集直到通过。

本轮归档：`work/exo_metadata_diagnostics_20260921/summary.json` 是总入口；每品类 `enriched_gold/enrichment.json` 记录数据来源，cook 的 `diagnosis_test/errors.csv` 是 73 条待归因错误，`validated_rule_proposals.json` 是提案阻塞证据。仅保存小规模人标实验副本，没有增加生产全量大表副本。

## 后续标注怎样补

保留现有 T/F 不变，在独立诊断层补 `metadata_topic=T/F/U`、可见依据字段、拒绝原因代码、标准版本和审阅人。先确认“主题不符”“制作/呈现要求不符”“metadata 无法判断”“标准边界待裁决”各自的口径，再做少量对照复核。同频道不同 T/F 本身不能用来宣判谁标错。

日常运行可自动执行格式守恒检查、历史回归、按来源/频道切片和特征覆盖诊断，不需要每层重新人工标注；但这些检查只能说明未发生已知退化，不能证明新批次达到合格率。交付仍由冻结候选的独立人工验收决定。

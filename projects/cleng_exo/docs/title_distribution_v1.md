# CSV 标题主题分布分析

这一步只发现标题主题并整理证据，不修改输入，也不按主题删除视频。适用于项目中含 `video_id,title` 的 CSV、TSV 或 Parquet；`duration_seconds`、`channel` 可选。主题归属只使用 `title`，时长和频道仅用于汇总说明。

## 运行

```bash
.venv/bin/python3 02_脚本/pipeline/metadata.py topic-distribution \
  --input /absolute/path/to/input.csv \
  --out /absolute/path/to/new-output-directory \
  --topics 20 --sample-size 10000
```

也可以独立运行 `02_脚本/pipeline/title_distribution.py`，将输入文件作为位置参数。默认使用多语言字符 TF-IDF 发现词面主题。如果本机已有 sentence-transformers 模型，可用 `--encoder /absolute/path/to/local/model` 改用语义向量；脚本不会下载模型。

脚本从全部非空标题中均匀抽样拟合主题，再为输入中的每条非空标题归簇。`--sample-size` 只控制拟合样本，并不限制统计范围。随机种子、参数、输入 SHA256 和方法会写入 `manifest.json`。输入文件在两遍扫描之间改变时，运行报错。

## 输出

- `topic_distribution.csv`：每个主题的条数、占比、时长、常见标题模式、近中心标题，以及归属边界较近的比例。
- `title_patterns_by_topic.csv`：标题片段在主题内的命中条数、比例，以及相对全表的提升倍数。英文提取词和相邻词组，中文提取 2～3 字片段；这些是模式线索，不是已审定的清洗规则。
- `topic_examples.csv`：每个主题的中心样例和随机样例，帮助判断主题是否混杂。
- `title_topic_assignments.csv`：逐条 `video_id,title,topic_id,distance,assignment_margin`，可回连原表做抽样质检。`assignment_margin` 是最接近与次接近中心的相对距离差，不是正确概率。
- `topic_review.csv`：在主题统计后附空白人工列，供填写主题名、是否属于目标品类、审核者和备注。
- `report.md`：可读的主题摘要；`manifest.json`：输入和运行记录。

## 下一步如何使用

先查看占比较大、时长较长的主题，再同时检查中心样例、随机样例和标题模式；为混杂主题检查逐条归属与边界较近的样本。人工确定目标品类和需要清除的主题后，再从清楚的标题模式整理可审计的规则，并对规则命中与未命中样本分别抽样。不能仅凭 `topic_id` 或关键词提升倍数直接整簇删除；主题编号只在同一次运行内有意义。

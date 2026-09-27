# sport-live — YouTube 体育赛事视频数据清洗管道

> 从 ~407 万条 YouTube playlist 抓取数据中，通过多阶段规则管道 + LLM 质检验证，筛选高质量体育赛事/直播/解说视频。

## 项目结构

```
sport-live/
├── 02_脚本/                          # 共享脚本（所有数据集通用）
├── datasets/
│   ├── sports/                       # ★ 体育赛事数据集（已处理完成）
│   │   ├── 原始数据/
│   │   ├── 03_分块清洗产物/
│   │   ├── 04_合并清洗产物/
│   │   ├── 05_精炼产物/
│   │   ├── 06_QC质检验证/
│   │   ├── 07_文档/
│   │   └── output/
│   └── new_dataset/                  # 新数据集（待处理）
│       ├── 原始数据/
│       └── output/
└── README.md
```

## 管道流程

```
原始数据 → 脚本清洗 → 分块产物 → 合并产物 → 精炼产物 → QC 质检验证 → 最终产物
  ①          ②          ③          ④          ⑤           ⑥
```

## Sports 数据集导航

| 目录 | 说明 | 关键产出 |
|------|------|---------|
| [`原始数据/`](datasets/sports/原始数据/) | ① YouTube playlist 抓取原始 CSV（~407 万条） | 3 个 chunk 的原始数据 |
| [`02_脚本/`](02_脚本/) | ② 全部 Python 清洗/精炼/质检脚本 | 7 个脚本（含抽样） |
| [`03_分块清洗产物/`](datasets/sports/03_分块清洗产物/) | ③ 各 chunk 独立清洗 + 精炼输出 | chunk_1-2, chunk_05 产物 |
| [`04_合并清洗产物/`](datasets/sports/04_合并清洗产物/) | ④ 合并后清洗中间产物和基础精炼 | ⭐ 基础版 refined CSV（350 MB） |
| [`05_精炼产物/`](datasets/sports/05_精炼产物/) | ⑤ 多轮 QC 反馈迭代增强版本 | ⭐ sig_wl_bl 完整版（206 MB） |
| [`06_QC质检验证/`](datasets/sports/06_QC质检验证/) | ⑥ 各阶段 LLM 文本质检结果（n=385） | 通过/未通过名单、冲突分析 |
| [`07_文档/`](datasets/sports/07_文档/) | ⑦ 全部项目文档 | 清洗报告、规则手册、进程文档 |
| [`output/`](datasets/sports/output/) | 脚本默认输出目录 | 脚本直出产物 |

## 快速开始 — Sports 数据集

```bash
cd datasets/sports
conda activate data_cleaning
export DASHSCOPE_API_KEY='your-key'

# 完整管道
python3 ../../02_脚本/clean_sports_chunk_v2.py 原始数据/sports_chunk_05.csv
python3 ../../02_脚本/clean_sports_chunk_refine_v2.py output/sports_chunk_05_clean_v2_all.csv
python3 ../../02_脚本/stratified_sample.py output/sports_chunk_05_clean_v2_refined.csv
python3 ../../02_脚本/chunk_text_qc_v2.py qc_sample.csv -w 20
```

## 快速开始 — 新数据集

```bash
# 1. 将原始 CSV 放入 datasets/new_dataset/原始数据/
cp /path/to/your/data.csv datasets/new_dataset/原始数据/

# 2. 进入数据集目录运行管道
cd datasets/new_dataset
python3 ../../02_脚本/clean_sports_chunk_v2.py 原始数据/your_data.csv
python3 ../../02_脚本/clean_sports_chunk_refine_v2.py output/your_data_clean_v2_all.csv
python3 ../../02_脚本/stratified_sample.py output/your_data_clean_v2-qc_refined.csv
python3 ../../02_脚本/chunk_text_qc_v2.py qc_sample.csv -w 20
```

> **工作原理**：所有脚本使用相对路径 `output/` 作为输出目录。只需在对应数据集目录下运行脚本，输入输出自动隔离。

## 核心指标（Sports v2 refined）

| 指标 | 值 |
|------|-----|
| 原始数据量 | ~407 万条 |
| 清洗后数据量 | ~55 万条 |
| 合格率 (Precision) | ~73% |
| 召回率 (Recall) | ~39% |
| 最优产物 | `datasets/sports/04_合并清洗产物/sports_chunk_merged_clean_v2-qc_refined.csv` |

详细说明见 [`datasets/sports/07_文档/项目进程文档.md`](datasets/sports/07_文档/项目进程文档.md)。

# 02 脚本

完整的清洗管道脚本集。按调用顺序排列。

## 主清洗管道

| 脚本 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `clean_sports_chunk_v2.py` | v2 主清洗（单 chunk） | 原始 CSV | high/medium/all/dropped |
| `clean_sports_chunk_v2_merged.py` | v2 主清洗（多 chunk 合并） | 合并 CSV | high/medium/all/dropped |
| `clean_chunk_fast.py` | 快速清洗（跳过 Step1） | post_bl CSV | high/medium/all/dropped |

## 精炼

| 脚本 | 用途 |
|------|------|
| `clean_sports_chunk_refine.py` | 共享精炼规则库（被 refine_v2 引用） |
| `clean_sports_chunk_refine_v2.py` | v2 精炼包装器 |

## 质检验证

| 脚本 | 用途 |
|------|------|
| `chunk_text_qc_v2.py` | LLM 文本质检（主力，原子写/备份/错误追溯） |
| `chunk_text_qc.py` | LLM 文本质检 v1（冲突检测已禁用，保留备用） |
| `stratified_sample.py` | 分层抽样（95% 置信度, ±5% 误差） |

## 依赖

```bash
conda activate data_cleaning
pip install pandas duckdb tqdm openai
export DASHSCOPE_API_KEY='your-key'
```

## 典型调用顺序

```bash
# 1. 清洗
python3 02_脚本/clean_sports_chunk_v2.py 原始数据/sports_chunk_05.csv

# 2. 精炼
python3 02_脚本/clean_sports_chunk_refine_v2.py output/sports_chunk_05_clean_v2_all.csv

# 3. 抽样
python3 02_脚本/stratified_sample.py output/sports_chunk_05_clean_v2_refined.csv

# 4. 质检
python3 02_脚本/chunk_text_qc_v2.py qc_sample.csv -w 20
```

## v3 DuckDB 管道（NEW）

| 脚本 | 用途 |
|------|------|
| `clean_sports_v3.py` | v3 DuckDB + Parquet 清洗管道 |
| `backup_rules.py` | 主规则备份 & 回滚 |
| `core/rules_manager.py` | 规则加载、备份、保护 |

| `core/scoring.py` | DuckDB UDF 注册（黑名单/计分/实体） |
| `core/playlist.py` | Pass1 DuckDB SQL 聚合 |
| `core/cleaner.py` | Pass2 DuckDB 多步过滤 |
| `core/reports.py` | 报告 + 审计样本生成 |
| `rules/*.toml` | 统一规则仓库 |

```bash
# 默认输出 Parquet
python3 clean_sports_v3.py input.csv -o output/

# 大文件分块
python3 clean_sports_v3.py input.csv --chunksize 500000

# 同时输出 CSV + Parquet
python3 clean_sports_v3.py input.csv --fmt both
```

v2 脚本冻结保留，规则向后兼容。

## SOP 管道 (SOP.md)

所有脚本统一 `-o runs/XXX_name/` 输出约定：

```bash
# Phase 0/1 — 数据规范化
python3 phase0_normalize.py data.csv -o data/runs/001_baseline/
#   → data/runs/001_baseline/baseline.parquet
#   → data/runs/001_baseline/baseline_stats.md

# Phase 2 — 抽样
python3 phase2_sample.py data/runs/001_baseline/baseline.parquet -o data/runs/002_audit/
#   → data/runs/002_audit/audit_sample_v1.parquet
#   → data/runs/002_audit/audit_stats.csv

# Phase 3 — 污染分析 (需先标注 audit_sample)
python3 phase3_analyze.py data/runs/002_audit/audit_sample_v1.parquet -o data/runs/003_analysis/
#   → data/runs/003_analysis/pollution_analysis_v1.md

# Phase 5 — 规则清洗
python3 clean_sports_v3.py data/runs/001_baseline/baseline.parquet -o data/runs/005_clean/
#   → data/runs/005_clean/clean_high.parquet
#   → data/runs/005_clean/clean_all.parquet
#   → data/runs/005_clean/clean_dropped.parquet
#   → data/runs/005_clean/pipeline_report.md
```

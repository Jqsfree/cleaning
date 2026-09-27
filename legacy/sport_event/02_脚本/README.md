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

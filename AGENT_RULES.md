# Agent 执行规范

## 环境

```bash
conda activate data_cleaning
export DASHSCOPE_API_KEY='sk-xxx'  # QC 需要
cd /home/jqs/sport-live
```

## 新数据集 SOP 流程

```
1. Phase 0  数据规范化 → baseline.parquet + baseline_stats.md
   python3 02_脚本/phase0_normalize.py data/raw/xxx.csv -o data/runs/{sport}/001_baseline/

2. 初始化数据集规则
   python3 -c "
   import sys; sys.path.insert(0,'02_脚本')
   from core.rules_manager import init_dataset_rules
   init_dataset_rules('data/runs/{sport}/rules/', force=True)
   "

3. Phase 2  抽样 500 条
   python3 02_脚本/phase2_sample.py data/runs/{sport}/001_baseline/baseline.parquet \
     -o data/runs/{sport}/002_audit/ --sample-size 500 --seed 42

4. Phase 2  LLM 质检
   python3 02_脚本/phase2_qc.py data/runs/{sport}/002_audit/audit_sample_v1.parquet \
     -o data/runs/{sport}/002_audit/ -w 20 --force

5. Phase 3  污染分析
   python3 02_脚本/phase3_analyze.py data/runs/{sport}/002_audit/audit_sample_v1.parquet \
     -o data/runs/{sport}/003_analysis/

6. Phase 5  规则过滤
   python3 02_脚本/clean_sports_v3.py data/runs/{sport}/001_baseline/baseline.parquet \
     -o data/runs/{sport}/005_clean/ --run run01

7. Phase 7  keep QC（抽样 + 验证）
   python3 02_脚本/phase2_sample.py data/runs/{sport}/005_clean/run01/clean_all.parquet \
     -o data/runs/{sport}/008_keep_qc/ --sample-size 300 --seed 42
   python3 02_脚本/chunk_text_qc_v2.py data/runs/{sport}/008_keep_qc/audit_sample_v1.parquet \
     -o data/runs/{sport}/008_keep_qc/ -w 20
```

## 迭代循环

```
分析 keep QC 中的 F 样本 → 加黑名单 → 重跑 Phase 5 → 重抽 keep → 重QC
```

### 加规则命令

```bash
python3 -c "
import sys; sys.path.insert(0,'02_脚本')
import shutil, re, tomllib
from datetime import datetime

patterns = [
    r'\b(关键词1|关键词2)\b',
    r'\b(频道名1|频道名2)\b',
]

path = 'data/runs/{sport}/rules/blacklist.toml'
shutil.copy2(path, f'{path}.bak_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}')
lines = open(path).read().split('\n')
insert_at = next((i for i,l in enumerate(lines) if l.strip().startswith('[[r2]]')), len(lines))
existing = {m.group(1) for l in lines if (m:=re.search(r'pattern\s*=\s*\"(.*)\"',l))}
added = 0
for p in patterns:
    if p not in existing:
        lines.insert(insert_at,''); lines.insert(insert_at+1,'[[pass2]]')
        lines.insert(insert_at+2,'category = \"keep_qc_fp\"')
        lines.insert(insert_at+3,f'pattern = \"{p}\"'); insert_at+=4; added+=1
fixed = []
for line in lines:
    m = re.match(r'(\s*pattern\s*=\s*\")(.*)(\"\s*)$', line)
    if m and '\\\\\\\\' not in m.group(2): fixed.append(f'{m.group(1)}{m.group(2).replace(chr(92),chr(92)*2)}{m.group(3)}')
    else: fixed.append(line)
open(path,'w').write('\n'.join(fixed))
r = tomllib.load(open(path,'rb'))
print(f'pass2={len(r[\"pass2\"])} added={added}')
"
```

## 分析 keep QC 失败样本

```bash
python3 -c "
import pandas as pd
path = 'data/runs/{sport}/008_keep_qc/audit_sample_v1_textqc_*.csv'
import glob; path = glob.glob(path)[-1]  # 取最新
df = pd.read_csv(path)
fails = df[df['qc_text_result'] == 'F']
print(f'T={len(df[df.qc_text_result==\"T\"])} F={len(fails)}')
print('\nF keyword top:')
for kw,cnt in fails['keyword'].value_counts().head(10).items():
    print(f'  {cnt:>3} {str(kw)[:60]}')
print('\nF samples:')
for _,row in fails.head(10).iterrows():
    print(f'  [{str(row.get(\"keyword\",\"\"))[:30]}] {str(row.get(\"title\",\"\"))[:55]}')
"
```

## 关键规则

1. **QC 用 parquet 文件时，脚本是 `chunk_text_qc_v2.py`**（不是 `phase2_qc.py`）
2. **Phase 5 每次重跑会覆盖同 run 的 clean_all.parquet**——重新抽样要重做
3. **黑名单检查 title+channel+keyword**（修过的），不是只查 title
4. **改规则后必须重跑 Phase 5 + 重抽 keep + 重QC**，之前的数据过时
5. **规则文件路径:** 通用 `02_脚本/rules/`，数据集专属 `data/runs/{sport}/rules/`
6. **Phase 5 会自动使用数据集专属规则**（如果存在）

## 监控面板

```bash
conda activate data_cleaning
streamlit run monitor.py
# → http://localhost:8501
```

## 最终产出

```bash
# 导出 done CSV
python3 -c "
import duckdb
con = duckdb.connect()
con.execute(\"COPY (SELECT * FROM read_parquet('data/runs/{sport}/005_clean/run01/clean_all.parquet')) TO 'data/runs/{sport}/{原始文件名}_done.csv' (HEADER, DELIMITER ',')\")
n = con.execute(\"SELECT COUNT(*) FROM 'data/runs/{sport}/{原始文件名}_done.csv'\").fetchone()[0]
print(f'{n:,} 行')
con.close()
"
```

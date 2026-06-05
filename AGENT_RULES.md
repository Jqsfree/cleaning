# Agent 执行规范

> 阶段编号对齐 SOP v4.2

## 环境

```bash
conda activate data_cleaning
export DASHSCOPE_API_KEY='sk-xxx'  # QC 需要
cd /home/jqs/sport-live
```

## 新数据集 SOP 流程（SOP v4.2 阶段编号）

```
Phase 1  数据规范化 → {raw}_raw.parquet + baseline_stats.md
   python3 02_脚本/phase0_normalize.py data/raw/xxx.csv -o data/runs/{sport}_{batch}/001_baseline/

Phase 2  规则过滤（共享 + 运动专属）
   python3 02_脚本/clean_sports_v3.py data/runs/{sport}_{batch}/001_baseline/{raw}_raw.parquet \
     -o data/runs/{sport}_{batch}/005_clean/ --run run01
   # project_log 必须记录: Shared Rules Version: v{N}

Phase 3  KEEP + DROP 随机抽样 QC
   # 抽样量根据 keep 规模: <50K→300, 50K~500K→500, >500K→1000
   python3 02_脚本/phase2_sample.py data/runs/{sport}_{batch}/005_clean/run01/clean_all.parquet \
     -o data/runs/{sport}_{batch}/007_keep_qc/ --sample-size 300 --seed 42
   # LLM 质检 (sample files in 005_clean/run{N}/)
   python3 02_脚本/chunk_text_qc_v2.py data/runs/{sport}_{batch}/005_clean/run01/{raw}_{run}_keep_sample.parquet \
     -o data/runs/{sport}_{batch}/005_clean/run01/ -w 20
   # 必须记录: Precision, FN Rate, Retention Rate

Phase 4  决策
   Precision ≥ 70% 且 FN < 8% → 冻结交付
   Precision < 70% → Phase 5（加专属黑名单）
   FN ≥ 8% → 补充 entities.toml + 升版本号 + Phase 2 重跑
   Retention Rate 下降 >30% → 人工复核

Phase 5  构建数据集专属规则 (FP 黑名单)
   # 改动前先备份: cp rules/blacklist.toml rules/blacklist.toml.bak.$(date +%Y%m%d)
   # 分析 Phase 3 QC 中的 F 样本
   python3 02_脚本/phase3_analyze.py data/runs/{sport}_{batch}/007_keep_qc/audit_sample_v1.parquet \
     -o data/runs/{sport}_{batch}/003_analysis/
   # 单轮新增规则 ≤ 20 条，超过必须人工审批

Phase 6  重新过滤（共享 + 运动专属规则）
   python3 02_脚本/clean_sports_v3.py data/runs/{sport}_{batch}/001_baseline/{raw}_raw.parquet \
     -o data/runs/{sport}_{batch}/005_clean/ --run run{N+1}

Phase 7  再次随机抽样 QC
   # 同 Phase 3，run 递增 → 回 Phase 4 决策

Phase 8（可选） DROP 临时召回
   # 触发条件: FN ≥ 8% 且 估算绝对量 ≥ 10K 且 人工确认
```

## 迭代循环 (Phase 5→6→7)

```
分析 Phase 7 keep QC 中的 F 样本 → 加黑名单 → 重跑 Phase 6 → 重抽 keep → Phase 7 重QC → Phase 4 决策
```

最多 3 轮黑名单迭代，entities 修改最多 2 次。

**加规则原则:** 频道名规则一律放专属 `rules/blacklist.toml`，不进入共享库。
共享 `02_脚本/rules/current/blacklist.toml` 只放内容类型规则（kpop/gaming/宗教/动漫等）。

### 加规则命令（写入运动专属 rules/）

```bash
python3 -c "
import sys; sys.path.insert(0,'02_脚本')
import shutil, re, tomllib
from datetime import datetime

patterns = [
    r'\b(关键词1|关键词2)\b',
    r'\b(频道名1|频道名2)\b',
]

path = 'data/runs/{sport}_one/rules/blacklist.toml'
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
path = 'data/runs/{sport}_{batch}/007_keep_qc/audit_sample_v1_textqc_*.csv'
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
2. **Phase 6 每次重跑会覆盖同 run 的 clean_all.parquet**——重新抽样要重做
3. **黑名单检查 title+channel+keyword**（修过的），不是只查 title
4. **改规则后必须重跑 Phase 6 + 重抽 keep + Phase 7 重QC**，之前的数据过时
5. **规则文件路径（三层结构）:**
   - **共享规则** `02_脚本/rules/current/` — 内容类型黑名单 + 跨运动实体词
   - **共享规则版本** `02_脚本/rules/releases/v{N}/` — 历史版本，每次修改前升版
   - **运动专属规则** `data/runs/{sport}_one/rules/` — 频道名黑名单，同运动新批次 cp -r 复用
6. **Phase 2 (`clean_sports_v3.py`) 会加载数据集专属规则**（如果 `data/runs/{sport}_one/rules/` 存在）并合并共享规则一起使用
7. **停止条件:** Precision ≥ 70% 且 FN < 8%，或 3 轮黑名单迭代完成
8. **改动规则前必须备份**（专属: `.bak.日期`，共享: 升版本号）
9. **单轮新增规则 ≤ 20 条**，超过必须人工审批

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
con.execute(\"COPY (SELECT * FROM read_parquet('data/runs/{sport}_{batch}/005_clean/run01/clean_all.parquet')) TO 'data/runs/{sport}_{batch}/{原始文件名}_done.csv' (HEADER, DELIMITER ',')\")
n = con.execute(\"SELECT COUNT(*) FROM 'data/runs/{sport}_{batch}/{原始文件名}_done.csv'\").fetchone()[0]
print(f'{n:,} 行')
con.close()
"
```

# sport-live — YouTube 体育赛事视频数据清洗管道

> 从 ~240 万条 YouTube playlist 抓取数据中，通过多阶段规则管道 + LLM 质检验证，筛选高质量体育赛事/直播/解说视频。覆盖 15 个运动项目。

## 项目结构

```
sport-live/
├── data/
│   ├── raw/                          # 原始 CSV（15 个文件，~240 万条）
│   ├── runs/                         # 管道产物（{sport}_{batch}/ 格式，SOP v4）
│   │   └── {sport}_{batch}/
│   │       ├── 001_baseline/         # Phase 1 产物
│   │       ├── 005_clean/            # Phase 2/6 产物
│   │       ├── 00{7,8,9,10}_*_qc/   # Phase 3/7 QC 结果
│   │       ├── rules/                # 数据集专属规则
│   │       ├── deliver/              # 最终交付文件
│   │       └── project_log.md        # 项目日志
│   ├── archive/                      # 历史数据
│   └── 交付/                         # 跨数据集汇总交付
├── 02_脚本/                          # 全部 Python 脚本 + 共享规则
│   ├── rules/                        # 共享规则 (blacklist/entities/whitelist)
│   ├── core/                         # 核心模块
│   └── *.py                          # 管道脚本
└── README.md
```

## 管道流程（SOP v4）

```
原始 CSV → Phase 1 基础过滤 → Phase 2 共享规则过滤 → Phase 3 QC 抽样
  → Phase 4 决策 → Phase 5 专属规则 → Phase 6 重过滤 → Phase 7 再QC
  → Phase 8 召回(可选) → 交付
```

## 数据集总览

见 [项目记录.md](项目记录.md) — 15 个数据集，8 个已完成，7 个进行中/待处理。

## 快速开始 — SOP v4 管道

```bash
cd sport-live
conda activate data_cleaning

# 1. Phase 1: 数据规范化
python3 02_脚本/phase0_normalize.py data/raw/xxx.csv -o data/runs/{sport}_{batch}/001_baseline/

# 2. Phase 2: 共享规则过滤
python3 02_脚本/clean_sports_v3.py data/runs/{sport}_{batch}/001_baseline/baseline.parquet \
  -o data/runs/{sport}_{batch}/005_clean/ --run run01

# 3. Phase 3: 随机抽样 QC
python3 02_脚本/phase2_sample.py data/runs/{sport}_{batch}/005_clean/run01/clean_all.parquet \
  -o data/runs/{sport}_{batch}/007_keep_qc/ --sample-size 300 --seed 42
python3 02_脚本/chunk_text_qc_v2.py data/runs/{sport}_{batch}/007_keep_qc/audit_sample_v1.parquet \
  -o data/runs/{sport}_{batch}/007_keep_qc/ -w 20

# 4. Phase 4: 决策 → Precision ≥ 70%? 交付 : 迭代
```

详见 [SOP_v4.2.md](SOP_v4.2.md) 和 [AGENT_RULES.md](AGENT_RULES.md)。

## 核心指标（当前 15 个数据集）

| 指标 | 值 |
|------|-----|
| 运动项目 | 15 个 |
| 已完成 | 8 个 (射箭/排球/手球/乒乓球/攀岩/冰壶/短道速滑/马术) |
| 进行中 | 7 个 (冰球/跳水/链球/马拉松/铅球×2/美式橄榄球) |
| 平均 Precision (done) | ~73% |
| 总原始数据量 | ~240 万条 |

详细进度见 [项目记录.md](项目记录.md)。

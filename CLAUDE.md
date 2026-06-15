# sport-live — YouTube 体育赛事视频数据清洗管道

## 环境

```bash
conda activate data_cleaning
cd /home/jqs/sport-live/02_脚本
export DASHSCOPE_API_KEY="sk-..."  # 文本 QC 需要
```

## 项目结构

```
sport-live/
├── data/raw/          # 原始 YouTube CSV 抓取数据
├── data/runs/         # 管道输出 ({sport}_{batch}/ → 001_baseline/, 005_clean/run{N}/, deliver/)
├── 02_脚本/           # 所有管道脚本
│   ├── core/          # 共享库 (scoring.py, cleaner.py, rules_manager.py, playlist.py, progress.py, ...)
│   └── rules/         # 共享规则 (blacklist.toml, entities.toml, whitelist.toml)
│       ├── current/   # 版本管理的真理源 — 总是在这里修改
│       └── releases/  # 历史版本 (v001-v010)
├── tests/             # Pytest (test_scoring.py + test_qc.py)
├── monitor.py         # Streamlit 管道监控面板
└── AGENT_RULES.md     # 完整阶段命令和规则参考 (读它以获取详细信息)
```

## 管道阶段

```
Phase 0 → 规范化 (phase0_normalize.py → {raw}_raw.parquet)
Phase 2 → 抽样 (phase2_sample.py) + QC (chunk_text_qc_v2.py → T/F 标注)
Phase 3 → 污染分析 (phase3_analyze.py)
Phase 4 → 规则生成 (phase4_rules.py, 仅分析)
Phase 5 → 规则过滤 (clean_sports_v3.py → keep/drop + report)
Phase 6 → 评估 (phase6_evaluate.py)
→ 如果 Precision < 70%: 添加黑名单规则, 重跑 Phase 5
→ 如果 Precision >= 70%: 冻结 → 交付
```

## 关键命令

```bash
# Phase 0: 规范化原始 CSV
python3 phase0_normalize.py ../data/raw/xxx.csv -o ../data/runs/{sport}_batch/001_baseline --min-duration 90 --max-duration 21600

# Phase 2: 抽样 (95% 置信度)
python3 phase2_sample_v2.py ../data/runs/{sport}_batch/001_baseline/{raw}_raw.parquet -o ../data/runs/{sport}_batch/005_clean/run01/keep_qc --margin 0.05

# Phase 2 QC: 文本 LLM
python3 phase2_qc.py ../data/runs/{sport}_batch/005_clean/run01/keep_qc/xxx_qc.parquet --model qwen3.5-flash

# Phase 5: 规则清洗
python3 clean_sports_v3.py ../data/runs/{sport}_batch/001_baseline/{raw}_raw.parquet -o ../data/runs/{sport}_batch/005_clean --run run01

# Phase 6: 评估
python3 phase6_evaluate.py --audit xxx_qc.csv --clean-dir ../data/runs/{sport}_batch/005_clean/run01
```

## 规则管理约束

1. **总是在 `current/` 中修改规则** — 不要直接编辑 `rules/`
2. **修改后同步**: `python3 -c "from core.rules_manager import sync_current_to_main; sync_current_to_main()"`
   - 或者手动: `\cp 02_脚本/rules/current/*.toml 02_脚本/rules/`
3. **修改前验证 TOML**: `python3 -c "import tomllib; tomllib.load(open('02_脚本/rules/current/blacklist.toml','rb'))"`
4. **pass2 = 硬过滤** (直接移除), **r2 = 软减分** (评分惩罚)。非体育频道必须使用 pass2
5. **停止条件**: QC T rate ≥ 70% 且 FN < 8%, 或 3 轮 pass2 迭代
6. **保留率监控**: 保留率下降 > 30% 时发出警告
7. **交付**: keep + recall 独立交付, 不自动合并

## 合并规则 (merge_rules)

用于合并运动专属规则与共享规则:
```python
from core.scoring import set_rules_dir, merge_rules
set_rules_dir('02_脚本/rules')
merge_rules('data/runs/{sport}_batch/rules')
```

注意: `merge_rules()` 中的 v4.2 `split("|")` bug 已修复 — 代码现在使用 pattern 列表。

## 监控

```bash
streamlit run monitor.py
```

## 测试

```bash
python3 -m pytest tests/test_scoring.py -v
```

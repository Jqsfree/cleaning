# 02 脚本

完整的清洗管道脚本集。

## SOP 管道

| Phase | 脚本 | 用途 |
|-------|------|------|
| 1 | `phase0_normalize.py` | 基础过滤与去重 |
| 2 | `clean_sports_v3.py` | 共享规则过滤 |
| 3 | `phase2_sample.py` + `chunk_text_qc_v2.py` | KEEP + DROP 随机抽样 QC |
| 5 | 编辑 `data/runs/{sport}/rules/` | 构建数据集专属规则 |
| 6 | `clean_sports_v3.py --run run02` | 重新过滤 |

## 核心模块

| 文件 | 用途 |
|------|------|
| `core/sop.py` | SOP 加载 + 运行日志 |
| `core/scoring.py` | DuckDB UDF（黑名单/计分/实体对齐） |
| `core/cleaner.py` | Pass2 多步过滤 |
| `core/playlist.py` | Pass1 playlist 分析 |
| `core/reports.py` | 报告 + 审计样本 |
| `core/progress.py` | 进度追踪 |
| `core/rules_manager.py` | 规则加载、备份 |

## 规则管理

| 文件 | 用途 |
|------|------|
| `rules/blacklist.toml` | 共享黑名单 |
| `rules/whitelist.toml` | 共享正负信号 |
| `rules/entities.toml` | 共享体育词典 |
| `backup_rules.py` | 主规则备份 & 回滚 |
| `promote_rules.py` | 数据集规则推广到主规则 |

## 质检

| 文件 | 用途 |
|------|------|
| `chunk_text_qc_v2.py` | LLM 文本质检（Qwen） |
| `phase2_qc.py` | Phase 2 QC 入口 |
| `phase2_sample.py` | 随机抽样工具 |
| `phase3_analyze.py` | 污染分析 |

## 其他

| 文件 | 用途 |
|------|------|
| `phase4_rules.py` | 规则生成 |
| `phase6_evaluate.py` | 效果验证 |

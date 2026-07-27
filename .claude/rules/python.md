---
paths:
  - "**/*.py"
  - "**/02_脚本/**"
---

# Python 工作流

- 环境：`conda activate data_cleaning`（给用户命令，不代为执行）
- CLI 用 argparse，不加框架；改 phase 脚本前先读 `02_脚本/core/`
- 修改 `categories/*/rules/` 或 `rules/` 下 TOML 需确认
- 管道脚本（`phase*.py`、`clean_*.py`）会改动数据，执行前需确认
- 数据查看优先 Read/Grep；批量统计可用 `python3 -c` 或 heredoc
- 用户说「检查/看下/对比/对齐」→ 直接探索统计出表

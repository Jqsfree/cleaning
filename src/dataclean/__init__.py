"""
dataclean — YouTube 视频元数据清洗管道（WIP / 冻结）

**非生产。** 日常入口：``02_脚本/``（见 AGENTS.md、本目录 README.md）。
本包仅保留实验用 Phase 包装；品类注册表委托 ``core.category_registry``。
"""

import sys
from pathlib import Path

# 过渡期：确保生产 02_脚本/ 模块可通过 sys.path 访问
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "02_脚本"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

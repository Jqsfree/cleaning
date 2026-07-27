"""
dataclean — YouTube 视频元数据清洗管道

通用管道框架 + 类别插件系统。
"""

import sys
from pathlib import Path

# 过渡期：确保旧 02_脚本/ 模块可通过 sys.path 访问
# Stage 3+ 会把 core 模块移入本包，届时移除此段
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "02_脚本"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

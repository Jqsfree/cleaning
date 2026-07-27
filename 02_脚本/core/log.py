#!/usr/bin/env python3
"""
core/log.py -- 轻量级结构化日志（供 core 模块和旧脚本使用）

与 src/dataclean/core/logging.py 接口一致，不依赖 dataclean 包。
"""

import sys
import time
from typing import TextIO


class Logger:
    """轻量级结构化日志。"""

    def __init__(self, stream: TextIO = sys.stdout):
        self._stream = stream

    def __call__(self, msg: str, level: str = "INFO") -> None:
        ts = time.strftime("%H:%M:%S")
        tag = f"[{ts}]" if not level or level == "INFO" else f"[{ts}][{level}]"
        print(f"{tag} {msg}", file=self._stream, flush=True)

    def info(self, msg: str) -> None:
        self(msg, "INFO")

    def ok(self, msg: str) -> None:
        self(msg, "OK")

    def warn(self, msg: str) -> None:
        self(msg, "WARN")

    def error(self, msg: str) -> None:
        self(msg, "ERROR")


log = Logger()


def banner(title: str, width: int = 62, char: str = "=") -> None:
    """打印分隔横幅。"""
    bar = char * width
    print()
    print(bar)
    print(f"  {title}")
    print(bar)

#!/usr/bin/env python3
"""
core/logging.py — 统一日志

替代各脚本中分散的 print() + log() 函数，
提供带时间戳、级别、自动 flush 的结构化日志。
"""

import sys
import time
from typing import TextIO


class Logger:
    """轻量级结构化日志，不引入 logging 模块的复杂度。

    用法:
        from dataclean.core.logging import log
        log("加载数据...")
        log("完成", level="OK")
        log("警告: 发现空值", level="WARN")
    """

    def __init__(self, stream: TextIO = sys.stdout, prefix: str = ""):
        self._stream = stream
        self._prefix = prefix

    def __call__(self, msg: str, level: str = "INFO") -> None:
        ts = time.strftime("%H:%M:%S")
        tag = f"[{ts}]" if not level or level == "INFO" else f"[{ts}][{level}]"
        line = f"{self._prefix}{tag} {msg}"
        print(line, file=self._stream, flush=True)

    def info(self, msg: str) -> None:
        self(msg, level="INFO")

    def ok(self, msg: str) -> None:
        self(msg, level="OK")

    def warn(self, msg: str) -> None:
        self(msg, level="WARN")

    def error(self, msg: str) -> None:
        self(msg, level="ERROR")

    def step(self, msg: str) -> None:
        """打印阶段分隔标题。"""
        self(f"── {msg} ──", level="")


# 全局默认 logger 实例
log = Logger()


def banner(title: str, width: int = 62, char: str = "=") -> None:
    """打印统一的分隔横幅。

    替代各 phase 脚本中分散的 print("="*62) 模式。
    """
    bar = char * width
    print()
    print(bar)
    print(f"  {title}")
    print(bar)


def ensure_input_exists(path: str) -> None:
    """校验输入文件存在，不存在则打印错误并退出。"""
    import os
    if not os.path.exists(path):
        log(f"文件不存在: {path}", level="ERROR")
        sys.exit(1)

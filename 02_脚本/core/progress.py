#!/usr/bin/env python3
"""
core/progress.py — 管道进度追踪

脚本调用 update() / mark_done() 写入 output_dir/progress.json。
无强制监控面板；可读文件即可查看进度。
"""

from __future__ import annotations

import json
import os
import threading
import time

_PROGRESS_FILE = "progress.json"


def update(output_dir: str, phase: int | str, **kwargs):
    """写入进度文件。"""
    path = os.path.join(output_dir.rstrip("/"), _PROGRESS_FILE)
    os.makedirs(output_dir, exist_ok=True)
    data = {
        "phase": phase,
        "timestamp": time.strftime("%H:%M:%S"),
        "updated_at": time.time(),
        **kwargs,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def mark_done(output_dir: str, phase: int | str, **kwargs):
    """标记阶段完成。"""
    update(output_dir, phase, status="done", **kwargs)


def read(output_dir: str) -> dict | None:
    """读取进度文件。"""
    path = os.path.join(output_dir.rstrip("/"), _PROGRESS_FILE)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class ThrottledProgress:
    """长任务节流写 progress.json（默认约 5s 或每 N 条）。"""

    def __init__(
        self,
        output_dir: str,
        stage: str | int,
        *,
        interval_sec: float = 5.0,
        every_n: int = 50,
        **base,
    ):
        self.output_dir = output_dir
        self.stage = stage
        self.interval_sec = interval_sec
        self.every_n = every_n
        self.base = dict(base)
        self._lock = threading.Lock()
        self._last = 0.0
        self._since = 0

    def tick(self, force: bool = False, **kwargs) -> None:
        with self._lock:
            self._since += 1
            now = time.time()
            due = (
                force
                or self._since >= self.every_n
                or (now - self._last) >= self.interval_sec
            )
            if not due and self._last > 0:
                return
            payload = {**self.base, **kwargs, "status": kwargs.get("status", "running")}
            update(self.output_dir, self.stage, **payload)
            self._last = now
            self._since = 0

    def done(self, **kwargs) -> None:
        with self._lock:
            payload = {**self.base, **kwargs}
            mark_done(self.output_dir, self.stage, **payload)
            self._last = time.time()
            self._since = 0

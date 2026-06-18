#!/usr/bin/env python3
"""
core/progress.py -- 管道进度追踪

每个 phase 脚本调用 update() 写入 progress.json，
monitor.py 读取渲染 Streamlit 面板。
"""

import json, os, time
from pathlib import Path

_PROGRESS_FILE = "progress.json"


def update(output_dir: str, phase: int, **kwargs):
    """写入进度文件。"""
    path = os.path.join(output_dir.rstrip("/"), _PROGRESS_FILE)
    data = {
        "phase": phase,
        "timestamp": time.strftime("%H:%M:%S"),
        "updated_at": time.time(),
        **kwargs,
    }
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def mark_done(output_dir: str, phase: int, **kwargs):
    """标记阶段完成。"""
    update(output_dir, phase, status="done", **kwargs)


def read(output_dir: str) -> dict | None:
    """读取进度文件。"""
    path = os.path.join(output_dir.rstrip("/"), _PROGRESS_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def read_all(base_dir: str) -> dict[str, dict]:
    """读取 base_dir 下所有子目录的进度。"""
    result = {}
    if not os.path.isdir(base_dir):
        return result
    for name in sorted(os.listdir(base_dir)):
        subdir = os.path.join(base_dir, name)
        if os.path.isdir(subdir):
            data = read(subdir)
            if data:
                result[name] = data
    return result

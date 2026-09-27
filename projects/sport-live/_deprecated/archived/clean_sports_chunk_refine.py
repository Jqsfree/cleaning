#!/usr/bin/env python3
"""
共享精炼规则 — 规则已迁移至 blacklist.toml + entities.toml + cleaning_core.py。

本文件保留向后兼容：所有规则和函数从 cleaning_core 代理。
"""

import sys, tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

def _resolve_path(name):
    for d in [Path(__file__).resolve().parent / 'rules', Path(__file__).resolve().parent]:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(name)
import cleaning_core as _c

# 向后兼容导出：从 TOML 文件重新加载原始规则列表
_bl = tomllib.load(open(_resolve_path('blacklist.toml'), 'rb'))
R2_BLACKLIST_PATTERNS = [item['pattern'] for item in _bl['r2']]
R2_BLACKLIST_RE = _c.BLACKLIST_R2_RE

WEAK_TITLE_PATTERNS = [
    item['pattern']
    for item in _c._wl.get('weak_titles', [])
]
WEAK_TITLE_RE = _c.WEAK_TITLE_RE
WEAK_CHANNEL_PATTERNS = [
    item['pattern']
    for item in _c._wl.get('weak_channels', [])
]
WEAK_CHANNEL_RE = _c.WEAK_CHANNEL_RE
WEAK_ENTITY_WORDS = _c.WEAK_ENTITY_WORDS

def refine_row(row):
    return _c.refine_row(row)

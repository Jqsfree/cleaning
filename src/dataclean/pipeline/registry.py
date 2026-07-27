#!/usr/bin/env python3
"""
pipeline/registry.py — 类别和阶段注册表

集中管理所有 category cleaner 和 pipeline phase 的注册与发现。
取代 phase5_clean.py 中分散的 _CLEANERS 字典。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Any

from dataclean.core.logging import log

# ── Category 注册表 ──
# {name: module_path} — 与 phase5_clean.py 的 _CLEANERS 等价

CATEGORY_REGISTRY: dict[str, str] = {
    "language_teaching": "categories.language_teaching.cleaner",
    "beauty": "categories.beauty.cleaner",
    "welding": "categories.welding.cleaner",
    "film_tv": "categories.film_tv.cleaner",
}

# 未来可扩展：
# "ego_repair": "categories.ego_repair.cleaner",
# "lila_outdoor": "categories.lila_outdoor.cleaner",


def register_category(name: str, module_path: str) -> None:
    """注册一个新的数据集类别。

    用法:
        register_category("my_category", "categories.my_category.cleaner")
    """
    CATEGORY_REGISTRY[name] = module_path
    log(f"注册类别: {name} → {module_path}")


def list_categories() -> list[str]:
    """列出所有已注册的类别名。"""
    return sorted(CATEGORY_REGISTRY.keys())


def load_cleaner(category: str) -> Callable[..., dict]:
    """动态加载类别 cleaner 模块，返回其 clean() 函数。

    等价于 phase5_clean.py 的 _load_cleaner()。
    """
    module_path = CATEGORY_REGISTRY.get(category)
    if module_path is None:
        available = ", ".join(list_categories())
        print(f"[ERROR] 未知类别: {category}")
        print(f"  可用: {available}")
        sys.exit(1)

    import importlib
    try:
        mod = importlib.import_module(module_path)
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[ERROR] 无法加载类别模块: {module_path}")
        print(f"  {e}")
        print(f"  检查 categories/{category}/cleaner.py 是否存在")
        sys.exit(1)
    if not hasattr(mod, "clean"):
        print(f"[ERROR] 类别模块缺少 clean() 函数: {module_path}")
        sys.exit(1)
    return mod.clean


# ── Phase 注册表 ──
# {phase_id: {name, description, module}}

PHASE_REGISTRY: dict[int, dict[str, Any]] = {
    0: {"name": "normalize", "desc": "数据规范化：字段标准化、去重、损坏/时长过滤 → baseline.parquet"},
    2: {"name": "sample",   "desc": "统计学抽样 → audit_sample.parquet"},
    3: {"name": "analyze",  "desc": "污染分析 → pollution_analysis_v1.md"},
    5: {"name": "clean",    "desc": "规则清洗：黑名单 + 打分 UDF → *_keep.parquet"},
    6: {"name": "evaluate", "desc": "效果验证：误杀率/漏检率评估"},
}


def list_phases() -> dict[int, dict[str, Any]]:
    """列出所有已注册的管道阶段。"""
    return dict(PHASE_REGISTRY)

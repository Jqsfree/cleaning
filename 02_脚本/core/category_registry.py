#!/usr/bin/env python3
"""
core/category_registry.py — 品类 cleaner / QC-only 单源注册表

生产入口（02_clean / run.py）与 WIP src/dataclean 均应读此处，禁止再维护第二份 dict。
"""

from __future__ import annotations

from typing import Callable

# {name: import path under 02_脚本/ on PYTHONPATH}
CLEANER_MODULES: dict[str, str] = {
    "language_teaching": "categories.language_teaching.cleaner",
    "beauty": "categories.beauty.cleaner",
    "welding": "categories.welding.cleaner",
    "film_tv": "categories.film_tv.cleaner",
    "live_sell": "categories.live_sell.cleaner",
    "human_live": "categories.human_live.cleaner",
    "exo": "categories.exo.cleaner",
}

# 无 cleaner；勿对 02_clean --category
QC_ONLY_CATEGORIES: frozenset[str] = frozenset({
    "ego_repair",
    "lila_outdoor",
})


def list_cleaner_categories() -> list[str]:
    return sorted(CLEANER_MODULES.keys())


def has_cleaner(category: str) -> bool:
    return category in CLEANER_MODULES


def is_qc_only(category: str) -> bool:
    return category in QC_ONLY_CATEGORIES


def get_cleaner_module(category: str) -> str | None:
    return CLEANER_MODULES.get(category)


def load_cleaner(category: str) -> Callable[..., dict]:
    """动态加载类别 cleaner，返回 clean()。未知类别 SystemExit。"""
    import importlib
    import sys

    module_path = CLEANER_MODULES.get(category)
    if module_path is None:
        available = ", ".join(list_cleaner_categories())
        print(f"[ERROR] 未知类别: {category}", flush=True)
        print(f"  可用 cleaner: {available}", flush=True)
        if category in QC_ONLY_CATEGORIES:
            print(f"  （{category} 为 QC-only，勿 02_clean --category）", flush=True)
        sys.exit(1)

    try:
        mod = importlib.import_module(module_path)
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[ERROR] 无法加载类别模块: {module_path}", flush=True)
        print(f"  {e}", flush=True)
        print(f"  检查 categories/{category}/cleaner.py 是否存在", flush=True)
        sys.exit(1)
    if not hasattr(mod, "clean"):
        print(f"[ERROR] 类别模块缺少 clean() 函数: {module_path}", flush=True)
        sys.exit(1)
    return mod.clean

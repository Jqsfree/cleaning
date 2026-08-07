#!/usr/bin/env python3
"""
pipeline/registry.py — WIP 阶段注册表（品类 cleaner 已迁至生产单源）

**禁止在本文件维护 CLEANER 列表。** 权威源：
  ``02_脚本/core/category_registry.py``

本包 Phase CLI 已冻结（须 ``--allow-wip``）；日常入口：``02_脚本/pipeline/``。
"""

from __future__ import annotations

from typing import Any, Callable

from core.category_registry import (  # noqa: E402
    CLEANER_MODULES,
    QC_ONLY_CATEGORIES,
    get_cleaner_module,
    has_cleaner,
    is_qc_only,
    list_cleaner_categories,
    load_cleaner as _load_cleaner_core,
)

# 兼容旧名：只读视图，勿原地改 dict 当生产真相
CATEGORY_REGISTRY: dict[str, str] = CLEANER_MODULES


def register_category(name: str, module_path: str) -> None:
    """已废弃：请改 02_脚本/core/category_registry.CLEANER_MODULES。"""
    raise RuntimeError(
        "register_category 已禁用；请编辑 02_脚本/core/category_registry.py "
        f"（试图注册 {name!r} → {module_path!r}）"
    )


def list_categories() -> list[str]:
    return list_cleaner_categories()


def load_cleaner(category: str) -> Callable[..., dict]:
    """委托生产 core.category_registry.load_cleaner。"""
    return _load_cleaner_core(category)


# ── Phase 注册表（WIP 冻结） ──
# 生产 SOP 见 AGENTS.md 双路径；禁止把本表当成统一 Phase0–7 编排源。

_WIP_PHASE_FROZEN = True

PHASE_REGISTRY: dict[int, dict[str, Any]] = {
    0: {"name": "normalize", "desc": "[WIP-FROZEN] 旧 Phase0；生产用 01_quality", "frozen": True},
    2: {"name": "sample", "desc": "[WIP-FROZEN] 旧 Phase2；生产用 03_sample→02_sample/", "frozen": True},
    3: {"name": "analyze", "desc": "[WIP-FROZEN] 旧 Phase3；生产用 04_analyze", "frozen": True},
    5: {"name": "clean", "desc": "[WIP-FROZEN] 旧 Phase5；生产用 02_clean→05_clean/", "frozen": True},
    6: {"name": "evaluate", "desc": "[WIP-FROZEN] 已废弃；生产不做 05_evaluate", "frozen": True},
}


def list_phases() -> dict[int, dict[str, Any]]:
    """列出 WIP 冻结阶段（非生产编排）。"""
    return dict(PHASE_REGISTRY)


def assert_wip_allowed(*, allow_wip: bool = False) -> None:
    """生产路径误调 WIP CLI 时阻断。"""
    if _WIP_PHASE_FROZEN and not allow_wip:
        raise SystemExit(
            "[ERROR] src/dataclean Phase CLI 已冻结；请用 02_脚本/pipeline/ "
            "（见 AGENTS.md）。确需实验加 --allow-wip。"
        )


# re-export helpers for callers that imported from registry
__all__ = [
    "CATEGORY_REGISTRY",
    "CLEANER_MODULES",
    "QC_ONLY_CATEGORIES",
    "PHASE_REGISTRY",
    "assert_wip_allowed",
    "get_cleaner_module",
    "has_cleaner",
    "is_qc_only",
    "list_categories",
    "list_cleaner_categories",
    "list_phases",
    "load_cleaner",
    "register_category",
]

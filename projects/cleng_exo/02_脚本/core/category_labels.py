#!/usr/bin/env python3
"""
core/category_labels.py — 品类 id ↔ 中文名单源映射

用途：产物命名要求「品类中文名 + 操作 + 日期」（如 ``商业服务_merged_0813.csv``），
但批次路径里的品类目录是英文 id（``data/runs/exo_service/…``），故需要一处
统一的映射，禁止在脚本里再散落第二份 dict（对标 ``core/category_registry.py``）。

约定：
- 键为 exo 家族品类 id（= ``data/runs/{category}/`` 与 ``raw/{category}/`` 的目录名）
- 值为该品类在数据文件名中使用的中文名（历史文件名口径，如 ``农业采集_0813-0818_clean_0828.csv``）
- ``exo`` 为通用兜底品类，无专属中文名，回落为 id 本身
"""

from __future__ import annotations

# {category_id: 中文名}
CATEGORY_LABELS: dict[str, str] = {
    "exo": "exo",
    "exo_agriculture": "农业采集",
    "exo_cook": "烹饪教学",
    "exo_factory": "工厂生产",
    "exo_fitness": "健身训练",
    "exo_livestock": "渔业牧业",
    "exo_medical": "医疗场景",
    "exo_outdoor": "户外探险",
    "exo_parent": "亲子互动",
    "exo_service": "商业服务",
    "exo_entertainment": "娱乐表演",
    "exo_env": "公益环保",
    "exo_unbox": "商品开箱",
    "exo_construction": "建筑施工",
    "exo_dance": "单人舞蹈",
    "exo_团队协作": "团队协作",
    # 兼容旧 id（已迁至 exo_团队协作/{pingpong,tennis}）
    "exo_pingpong": "双人乒乓",
    "exo_tennis": "双人网球",
}

# {中文名: category_id}
LABEL_TO_CATEGORY: dict[str, str] = {v: k for k, v in CATEGORY_LABELS.items()}


def category_label(category: str | None) -> str:
    """品类 id → 中文名。未知/空 → 原样返回（去掉首尾空白），便于临时批次。"""
    if not category:
        return ""
    key = str(category).strip()
    return CATEGORY_LABELS.get(key, key)


def label_to_category(label: str | None) -> str | None:
    """中文名 → 品类 id；不是已知中文名时返回 None。"""
    if not label:
        return None
    return LABEL_TO_CATEGORY.get(str(label).strip())


def known_categories() -> list[str]:
    """全部已登记品类 id（排序）。"""
    return sorted(CATEGORY_LABELS)


def known_labels() -> list[str]:
    """全部已登记中文名（排序）。"""
    return sorted(LABEL_TO_CATEGORY)

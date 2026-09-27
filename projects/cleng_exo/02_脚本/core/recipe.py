#!/usr/bin/env python3
"""
core/recipe.py — 品类 × 来源声明式流程配方

编排器只解释 recipe，不内置全局 Phase0–7。
权威文件：categories/<name>/recipe.toml（可缺省继承 _shared/recipe_default.toml）。
"""

from __future__ import annotations

import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

_CATEGORIES = Path(__file__).resolve().parent.parent / "categories"
_DEFAULT = _CATEGORIES / "_shared" / "recipe_default.toml"

Stage = dict[str, Any]


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = deepcopy(base)
    for k, v in overlay.items():
        if k == "flow":
            # flow 整段覆盖（按 source），不逐 stage merge
            base_flow = dict(out.get("flow") or {})
            over_flow = dict(v or {})
            base_flow.update(over_flow)
            out["flow"] = base_flow
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _normalize_flow(raw: dict[str, Any]) -> dict[str, list[Stage]]:
    """tomllib [[flow.human]] → {'human': [...], 'machine': [...]}。"""
    flow = raw.get("flow") or {}
    out: dict[str, list[Stage]] = {"human": [], "machine": []}
    for source in ("human", "machine"):
        stages = flow.get(source) or []
        if isinstance(stages, dict):
            # 容错：误写成 [flow.human] 单表
            stages = [stages]
        norm: list[Stage] = []
        for st in stages:
            if not isinstance(st, dict) or not st.get("id"):
                continue
            item = dict(st)
            item.setdefault("kind", "auto")
            item.setdefault("max_runs", 1)
            item.setdefault("optional", False)
            deps = item.get("depends") or []
            if isinstance(deps, str):
                deps = [deps]
            item["depends"] = list(deps)
            norm.append(item)
        out[source] = norm
    return out


def load_recipe(category: str) -> dict[str, Any]:
    """加载品类 recipe；无文件则用 default 并填 category。"""
    default: dict[str, Any] = {}
    if _DEFAULT.is_file():
        default = tomllib.loads(_DEFAULT.read_text(encoding="utf-8"))

    path = _CATEGORIES / category / "recipe.toml"
    data = dict(default)
    if path.is_file():
        overlay = tomllib.loads(path.read_text(encoding="utf-8"))
        data = _deep_merge(data, overlay)

    meta = dict(data.get("meta") or {})
    meta.setdefault("category", category)
    data["meta"] = meta
    data["flow"] = _normalize_flow(data)
    data.setdefault(
        "paths",
        {
            "quality": "01_quality",
            "sample": "02_sample",
            "qc": "03_qc",
            "rules": "04_rules",
            "clean": "05_clean",
            "tools": "06_tools",
            "deliver": "07_deliver",
        },
    )
    data.setdefault(
        "layers",
        {
            "bronze": "raw/{category}/",
            "silver": "01_quality + 05_clean",
            "gold": "07_deliver",
        },
    )
    return data


def flow_for(recipe: dict[str, Any], source: str) -> list[Stage]:
    source = source.strip().lower()
    if source not in ("human", "machine"):
        raise ValueError("source 必须是 human 或 machine")
    return list((recipe.get("flow") or {}).get(source) or [])


def stage_by_id(recipe: dict[str, Any], source: str, stage_id: str) -> Stage | None:
    for st in flow_for(recipe, source):
        if st.get("id") == stage_id:
            return st
    return None


def has_cleaner(recipe: dict[str, Any]) -> bool:
    return bool((recipe.get("meta") or {}).get("has_cleaner", False))


def deliver_tool(recipe: dict[str, Any], source: str) -> str:
    for st in flow_for(recipe, source):
        if st.get("id") == "deliver":
            return str(st.get("tool") or "copy_keep")
    return "copy_keep"


def needs_ge720(recipe: dict[str, Any], source: str) -> bool:
    return deliver_tool(recipe, source) == "ge720"

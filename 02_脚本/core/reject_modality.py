#!/usr/bin/env python3
"""
core/reject_modality.py — 模态映射与置信带辅助
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from core.reject_taxonomy import get_registry, normalize_tag

_SCRIPT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODALITY_MAP = (
    _SCRIPT_ROOT / "categories" / "_shared" / "reject_modality_map.toml"
)
DEFAULT_CASCADE = (
    _SCRIPT_ROOT / "categories" / "_shared" / "reject_cascade.toml"
)


def load_modality_map(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_MODALITY_MAP
    if not p.exists():
        return {"vision_thumb": {"default_fail_tag": "provisional:thumb_fail"}}
    return tomllib.loads(p.read_text(encoding="utf-8"))


def load_cascade_cfg(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CASCADE
    if not p.exists():
        return {
            "sampling": {"n_validate": 100, "seed": 42},
            "text_rule": {"default_band": "high"},
            "text_ml": {"low_max": 0.20, "high_min": 0.50},
            "thumb": {"default_fail_band": "high"},
            "sources": {},
            "metrics": {"min_n_trusted": 30, "max_overturn_rate": 0.35},
        }
    return tomllib.loads(p.read_text(encoding="utf-8"))


def confidence_band_for_rule(cfg: dict[str, Any] | None = None) -> str:
    c = cfg or load_cascade_cfg()
    return str(c.get("text_rule", {}).get("default_band", "high"))


def confidence_band_for_ml_score(score: float, cfg: dict[str, Any] | None = None) -> str:
    """低分 = 高把握负例(drop)。"""
    c = cfg or load_cascade_cfg()
    ml = c.get("text_ml", {})
    low_max = float(ml.get("low_max", 0.20))
    high_min = float(ml.get("high_min", 0.50))
    if score <= low_max:
        return "high"
    if score >= high_min:
        return "low"
    return "mid"


def confidence_band_for_thumb(cfg: dict[str, Any] | None = None) -> str:
    c = cfg or load_cascade_cfg()
    return str(c.get("thumb", {}).get("default_fail_band", "high"))


def source_status(propose_source: str, cfg: dict[str, Any] | None = None) -> str:
    c = cfg or load_cascade_cfg()
    sources = c.get("sources", {}) or {}
    key = propose_source.split(":")[0].split("_")[0]
    # blacklist:drop_reason → blacklist；vision_thumb → vision_thumb
    for name in ("blacklist", "ml_action", "vision_thumb", "column"):
        if name in propose_source or propose_source.startswith(name):
            return str(sources.get(name, "active"))
    if propose_source.startswith("column:"):
        return str(sources.get("column", "active"))
    return str(sources.get(key, "active"))


def map_vision_thumb_row(
    row: dict[str, Any] | Any,
    *,
    modality_map: dict[str, Any] | None = None,
) -> str | None:
    """
    从 vision_thumb 结果行得到 reject_tag；非 fail 返回 None。
    识别列: qc_thumb_result / qc_result / label
    """
    mmap = modality_map or load_modality_map()
    vt = mmap.get("vision_thumb", {})
    fail_labels = {str(x).strip().lower() for x in vt.get("fail_labels", ["f", "fail"])}
    pass_labels = {str(x).strip().lower() for x in vt.get("pass_labels", ["t", "pass"])}

    def _get(key: str) -> str:
        if hasattr(row, "get"):
            v = row.get(key, "")
        else:
            v = getattr(row, key, "") if hasattr(row, key) else ""
        if v is None:
            return ""
        return str(v).strip()

    result = (
        _get("qc_thumb_result")
        or _get("qc_result")
        or _get("label")
        or _get("result")
    )
    key = result.lower()
    if key in pass_labels or key == "t":
        return None
    if key not in fail_labels and key != "f":
        # 未知结果不提案
        if not key:
            return None
        # 非 T/F 且不在 fail 列表 → 跳过
        if key not in fail_labels:
            return None

    # reason hints
    evidence = (
        _get("qc_thumb_evidence")
        or _get("reason")
        or _get("reject_hint")
        or _get("evidence")
    )
    ev_l = evidence.lower()
    reg = get_registry()
    for hint in vt.get("reason_hints", []) or []:
        contains = str(hint.get("contains", "")).lower()
        tag = str(hint.get("tag", "")).strip()
        if contains and contains in ev_l and tag:
            return normalize_tag(tag, reg) or tag

    default = str(vt.get("default_fail_tag", "provisional:thumb_fail"))
    return normalize_tag(default, reg) or default

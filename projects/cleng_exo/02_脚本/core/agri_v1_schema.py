#!/usr/bin/env python3
"""exo 农业 v1 标注 schema（spec §2）。

valid = human_present == 1 AND crop_interaction == 1
"""

from __future__ import annotations

from typing import Any

import pandas as pd

SCENE_TYPES = (
    "agri_field",
    "market_kitchen",
    "indoor_garden",
    "game_cg",
    "machinery_ad",
    "mixed_vlog",
    "other",
    "no_human",
    "human_no_interaction",
)

BINARY_FIELDS = (
    "human_present",
    "human_synthetic",
    "crop_interaction",
    "mechanized_only",
)

AGRI_V1_FIELDS = (
    "video_id",
    "keyword",
    "channel",
    "duration_seconds",
    *BINARY_FIELDS,
    "action_free_text",
    "scene_type",
    "qc_confidence",
)

AGRI_V1_OPTIONAL_CONTEXT = (
    "title",
    "description",
    "thumbnail_url",
    "thumb_path",
    "human_label",
    "qc_result",
)

AGRI_V1_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "human_present": ("human_present", "has_human", "真人出镜"),
    "human_synthetic": ("human_synthetic", "is_cg", "CG游戏动画"),
    "crop_interaction": ("crop_interaction", "has_crop_interaction", "作物交互"),
    "mechanized_only": ("mechanized_only", "仅机械"),
    "action_free_text": ("action_free_text", "action_type", "动作类型"),
    "scene_type": ("scene_type", "场景类型"),
    "qc_confidence": ("qc_confidence", "confidence", "标注把握"),
}


def _norm_binary(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "是", "有"}:
        return "1"
    if s in {"0", "false", "f", "no", "n", "否", "无"}:
        return "0"
    return ""


def _norm_scene_type(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().lower()
    if s in SCENE_TYPES:
        return s
    return ""


def detect_agri_v1_col(columns: list[str] | pd.Index, field: str) -> str | None:
    lower_map = {str(c).lower(): str(c) for c in columns}
    for name in AGRI_V1_FIELD_ALIASES.get(field, (field,)):
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def is_valid_v1(row: pd.Series) -> bool:
    hp = _norm_binary(row.get("human_present", ""))
    ci = _norm_binary(row.get("crop_interaction", ""))
    return hp == "1" and ci == "1"


def normalize_agri_v1_frame(
    df: pd.DataFrame,
    *,
    id_col: str | None = None,
) -> pd.DataFrame:
    """将输入表规范为 v1 §2 字段（字符串 dtype，空值留空）。"""
    if df.empty:
        return pd.DataFrame(columns=list(AGRI_V1_FIELDS))

    id_col = id_col or detect_agri_v1_col(df.columns, "video_id") or "video_id"
    if id_col not in df.columns:
        raise ValueError(f"缺少 video_id 列；实际: {list(df.columns)}")

    out = pd.DataFrame()
    out["video_id"] = df[id_col].astype(str).str.strip()

    for field in AGRI_V1_FIELDS:
        if field == "video_id":
            continue
        src = detect_agri_v1_col(df.columns, field)
        if src and src in df.columns:
            out[field] = df[src]
        elif field in df.columns:
            out[field] = df[field]
        else:
            out[field] = ""

    for field in BINARY_FIELDS:
        out[field] = out[field].map(_norm_binary)

    out["scene_type"] = out["scene_type"].map(_norm_scene_type)
    for col in ("keyword", "channel", "action_free_text", "qc_confidence", "duration_seconds"):
        out[col] = out[col].fillna("").astype(str).str.strip()

    out["valid_v1"] = out.apply(is_valid_v1, axis=1)
    return out.reset_index(drop=True)


def merge_agri_v1_context(
    df: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    id_col: str = "video_id",
) -> pd.DataFrame:
    """合并可选上下文列（title/thumbnail 等）供人工标注。"""
    out = df.copy()
    ctx_cols = [c for c in AGRI_V1_OPTIONAL_CONTEXT if c in raw.columns and c not in out.columns]
    if not ctx_cols:
        return out
    slim = raw[[id_col, *ctx_cols]].drop_duplicates(id_col)
    slim = slim.rename(columns={id_col: "video_id"})
    slim["video_id"] = slim["video_id"].astype(str).str.strip()
    return out.merge(slim, on="video_id", how="left")


def audit_sample_columns() -> list[str]:
    """Phase 0 / Stage 5 人工复核表推荐列顺序。"""
    return [
        *AGRI_V1_FIELDS,
        "valid_v1",
        "human_label",
        "qc_result",
        "title",
        "thumbnail_url",
        "review_notes",
    ]

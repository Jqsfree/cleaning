#!/usr/bin/env python3
"""
core/human_qc.py — 人工质检结果契约（schema + 归一化）

人工标注是唯一可信 pass rate 来源。本模块统一字段与标签映射，
供 ingest_human_qc / 训练导出 / drop 回流复用。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# ── 规范字段 ─────────────────────────────────────────────────
CANONICAL_FIELDS = (
    "video_id",
    "human_label",       # pass | fail
    "qc_dimension",      # text | thumb | storyboard | overall | …
    "category",
    "source",            # human | machine
    "batch",
    "labeled_at",
    # 排除类验证（可选；缺省合法——人工不做全量细类标注）
    "reject_tags",       # canonical ids，逗号分隔
    "reject_action",     # confirm | correct | unset
    "reject_raw",        # 原始未映射字符串（可选）
)

# 训练可选列（有则保留）
TRAIN_OPTIONAL_FIELDS = (
    "title",
    "channel",
    "keyword",
    "description",
    "thumbnail_url",
    "thumb_path",
)

REJECT_TAG_CANDIDATES = (
    "reject_tags",
    "reject_tag",
    "排除类",
    "fail_reason",
    "drop_category",
    "blacklist_category",
)

REJECT_ACTION_CANDIDATES = (
    "reject_action",
    "reject_verify",
    "排除验证",
)

_CONFIRM_TOKENS = frozenset({
    "confirm", "confirmed", "ok", "yes", "y", "true", "1",
    "确认", "对", "正确",
})
_CORRECT_TOKENS = frozenset({
    "correct", "corrected", "fix", "edit",
    "纠正", "修改", "更正",
})

QC_DIMENSIONS = (
    "text",
    "thumb",
    "storyboard",
    "overall",
    "two_person",
    "definition",
)

# 列名候选（按优先级）
VIDEO_ID_CANDIDATES = (
    "video_id",
    "videoid",
    "videoId",
    "id",
    "yt_id",
    "youtube_id",
)

LABEL_CANDIDATES = (
    "human_label",
    "audit_label",
    "qc_result",
    "qc_text_result",
    "label",
    "result",
    "判定",
    "质检结果",
    "人工标注",
    "人工结果",
)

# pass / fail 映射（小写 key）
_PASS_TOKENS = frozenset({
    "pass", "p", "t", "true", "yes", "y", "1", "keep",
    "合格", "通过", "是", "正确", "ok",
})
_FAIL_TOKENS = frozenset({
    "fail", "f", "false", "no", "n", "0", "drop",
    "不合格", "不通过", "否", "错误", "failed",
})


def normalize_label(value: Any) -> str | None:
    """将任意标签归一化为 'pass' | 'fail'；无法识别返回 None。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    key = s.lower()
    # 中文不走 lower 后再查一遍原串
    if key in _PASS_TOKENS or s in _PASS_TOKENS:
        return "pass"
    if key in _FAIL_TOKENS or s in _FAIL_TOKENS:
        return "fail"
    return None


def detect_column(columns: list[str] | pd.Index, candidates: tuple[str, ...]) -> str | None:
    """在列名中按候选优先级找第一个命中（大小写不敏感）。"""
    lower_map = {str(c).lower(): str(c) for c in columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def detect_video_id_col(columns: list[str] | pd.Index) -> str | None:
    return detect_column(columns, VIDEO_ID_CANDIDATES)


def detect_label_col(columns: list[str] | pd.Index) -> str | None:
    return detect_column(columns, LABEL_CANDIDATES)


def detect_reject_tag_col(columns: list[str] | pd.Index) -> str | None:
    return detect_column(columns, REJECT_TAG_CANDIDATES)


def detect_reject_action_col(columns: list[str] | pd.Index) -> str | None:
    return detect_column(columns, REJECT_ACTION_CANDIDATES)


def normalize_reject_action(value: Any) -> str:
    """归一化为 confirm | correct | unset。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unset"
    s = str(value).strip()
    if not s:
        return "unset"
    key = s.lower()
    if key in _CONFIRM_TOKENS or s in _CONFIRM_TOKENS:
        return "confirm"
    if key in _CORRECT_TOKENS or s in _CORRECT_TOKENS:
        return "correct"
    if key in ("unset", "none", "na", "n/a", "-"):
        return "unset"
    return "unset"


def normalize_frame(
    df: pd.DataFrame,
    *,
    category: str,
    source: str,
    batch: str,
    dimension: str = "overall",
    label_col: str | None = None,
    id_col: str | None = None,
    reject_col: str | None = None,
    reject_action_col: str | None = None,
    labeled_at: str = "",
) -> pd.DataFrame:
    """
    将人工标注表规范为 canonical schema。

    丢弃无法映射的标签行；保留训练可选列（若存在）。
    reject_tags / reject_action 可选——缺省不报错（人工不做全量细类标注）。
    """
    if df.empty:
        raise ValueError("输入表为空")

    from core.reject_taxonomy import normalize_tags

    id_col = id_col or detect_video_id_col(df.columns)
    label_col = label_col or detect_label_col(df.columns)
    if not id_col:
        raise ValueError(
            f"无法识别 video_id 列；候选: {VIDEO_ID_CANDIDATES}；"
            f"实际列: {list(df.columns)}"
        )
    if not label_col:
        raise ValueError(
            f"无法识别标签列；候选: {LABEL_CANDIDATES}；"
            f"实际列: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["video_id"] = df[id_col].astype(str).str.strip()
    labels = df[label_col].map(normalize_label)
    out["human_label"] = labels
    out["qc_dimension"] = dimension
    out["category"] = category
    out["source"] = source.strip().lower()
    out["batch"] = batch
    out["labeled_at"] = labeled_at

    reject_col = reject_col or detect_reject_tag_col(df.columns)
    reject_action_col = reject_action_col or detect_reject_action_col(df.columns)

    reject_tags_list: list[str] = []
    reject_raw_list: list[str] = []
    reject_action_list: list[str] = []
    for i in range(len(df)):
        raw_val = df.iloc[i][reject_col] if reject_col else ""
        raw_s = "" if raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val)) else str(raw_val).strip()
        tags = normalize_tags(raw_s) if raw_s else []
        reject_tags_list.append(",".join(tags))
        reject_raw_list.append(raw_s)
        act_raw = df.iloc[i][reject_action_col] if reject_action_col else ""
        # 有纠正后的 tags 但未写 action → 视为 correct；仅有 tags → confirm（抽样验证）
        act = normalize_reject_action(act_raw)
        if act == "unset" and tags:
            act = "confirm"
        reject_action_list.append(act)

    out["reject_tags"] = reject_tags_list
    out["reject_action"] = reject_action_list
    out["reject_raw"] = reject_raw_list

    for col in TRAIN_OPTIONAL_FIELDS:
        if col in df.columns:
            out[col] = df[col]
        else:
            # 常见别名
            aliases = {
                "channel": ("channel_title", "channelTitle", "uploader"),
                "thumbnail_url": ("thumbnail", "thumb_url", "thumb"),
                "description": ("desc", "video_description"),
            }
            for alt in aliases.get(col, ()):
                if alt in df.columns:
                    out[col] = df[alt]
                    break

    n_before = len(out)
    out = out[out["human_label"].notna()].copy()
    out = out[out["video_id"] != ""].copy()
    n_drop = n_before - len(out)
    if n_drop:
        # 调用方可从返回的 attrs 读取
        out.attrs["dropped_unmapped"] = n_drop
    return out.reset_index(drop=True)


def human_validated_rejects(df: pd.DataFrame) -> pd.DataFrame:
    """抽样验证金标：reject_action in confirm|correct 且 reject_tags 非空。"""
    if "reject_tags" not in df.columns or "reject_action" not in df.columns:
        return df.iloc[0:0].copy()
    mask = (
        df["reject_action"].isin(("confirm", "correct"))
        & df["reject_tags"].astype(str).str.strip().ne("")
    )
    return df.loc[mask].copy()


def split_pass_fail(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 human_label 拆分 pass / fail。"""
    pass_df = df[df["human_label"] == "pass"].copy()
    fail_df = df[df["human_label"] == "fail"].copy()
    return pass_df, fail_df


def train_export_columns(df: pd.DataFrame) -> pd.DataFrame:
    """导出训练友好子集：规范字段 + 可选文本/缩略图列。"""
    cols = [c for c in CANONICAL_FIELDS if c in df.columns]
    for c in TRAIN_OPTIONAL_FIELDS:
        if c in df.columns and c not in cols:
            cols.append(c)
    return df[cols].copy()


def pass_rate(df: pd.DataFrame) -> float:
    """人工合格率（唯一有意义的 rate）。空表返回 0.0。"""
    n = len(df)
    if n == 0:
        return 0.0
    return float((df["human_label"] == "pass").sum()) / n

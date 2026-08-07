"""真人直播多帧特征与可解释规则。"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from statistics import median

import numpy as np
import pandas as pd


def classify_thumbnail_person(
    person_boxes: Sequence[Sequence[float]] | None,
    *,
    frame_size: tuple[int, int] | None,
    min_person_area_ratio: float = 0.08,
    error: str = "",
) -> dict[str, float | int | str]:
    """按单张缩略图最大人体框面积分流；读取错误必须保留。"""
    if error or person_boxes is None or frame_size is None:
        return {
            "action": "keep_error",
            "reason": error or "invalid_thumbnail",
            "person_count": 0,
            "max_person_ratio": 0.0,
        }
    summary = summarize_person_frames(
        [person_boxes],
        frame_sizes=[frame_size],
        min_person_area_ratio=min_person_area_ratio,
    )
    count = len(person_boxes)
    ratio = float(summary["max_person_ratio"])
    if count == 0:
        action, reason = "highconf_drop", "no_person"
    elif ratio < min_person_area_ratio:
        action, reason = "highconf_drop", "small_person"
    else:
        action, reason = "keep_candidate", "person_visible"
    return {
        "action": action,
        "reason": reason,
        "person_count": count,
        "max_person_ratio": ratio,
    }


def make_blind_sample(
    pool: pd.DataFrame,
    exclude_video_ids: set[str],
    *,
    n: int = 385,
    seed: int = 42,
    duration_bins: int = 5,
) -> pd.DataFrame:
    """按时长分层、优先频道不重复生成独立盲样。"""
    work = pool.copy()
    work["video_id"] = work["video_id"].fillna("").astype(str)
    work = work[
        ~work["video_id"].isin({str(value) for value in exclude_video_ids})
    ].drop_duplicates("video_id").copy()
    if len(work) < n:
        raise ValueError(f"候选不足：需要 {n}，仅有 {len(work)}")
    duration = pd.to_numeric(work.get("duration_seconds"), errors="coerce")
    duration = duration.fillna(duration.median()).fillna(0)
    bins = max(1, min(duration_bins, len(work)))
    work["_duration_bin"] = pd.qcut(
        duration.rank(method="first"),
        q=bins,
        labels=False,
        duplicates="drop",
    ).astype(int)
    channel = work.get("channel", pd.Series("", index=work.index))
    work["_channel_key"] = channel.fillna("").astype(str).str.strip()
    work["_channel_key"] = work["_channel_key"].where(
        work["_channel_key"].ne(""),
        work["video_id"],
    )

    chosen: list[int] = []
    used_channels: set[str] = set()
    strata = sorted(work["_duration_bin"].unique().tolist())
    base, remainder = divmod(n, len(strata))
    for position, stratum in enumerate(strata):
        quota = base + int(position < remainder)
        part = work[work["_duration_bin"].eq(stratum)].sample(
            frac=1,
            random_state=seed + int(stratum),
        )
        for idx, row in part.iterrows():
            key = str(row["_channel_key"])
            if key in used_channels:
                continue
            chosen.append(idx)
            used_channels.add(key)
            if sum(work.loc[chosen, "_duration_bin"].eq(stratum)) >= quota:
                break
    if len(chosen) < n:
        remaining = work.loc[~work.index.isin(chosen)].sample(
            frac=1,
            random_state=seed + 10_000,
        )
        for idx, row in remaining.iterrows():
            key = str(row["_channel_key"])
            if key in used_channels:
                continue
            chosen.append(idx)
            used_channels.add(key)
            if len(chosen) >= n:
                break
    if len(chosen) < n:
        remaining_idx = work.index[~work.index.isin(chosen)].tolist()
        chosen.extend(remaining_idx[: n - len(chosen)])
    sample = work.loc[chosen[:n]].copy()
    sample["sample_stratum"] = sample["_duration_bin"].map(
        lambda value: f"duration_q{int(value) + 1}",
    )
    return sample.drop(columns=["_duration_bin", "_channel_key"]).reset_index(drop=True)


def parse_vlm_label(text: str) -> str:
    """从简短模型响应中提取独立 T/F/U，无法解析时返回 ERROR。"""
    matches = re.findall(r"(?<![A-Z])[TFU](?![A-Z])", str(text).upper())
    return matches[-1] if matches else "ERROR"


def resolve_label_groups(frame: pd.DataFrame) -> pd.Series:
    """按 channel -> source_ref -> video_id 生成防泄漏分组。"""
    groups = pd.Series("", index=frame.index, dtype=object)
    for column in ("channel", "source_ref", "video_id"):
        if column not in frame.columns:
            continue
        values = frame[column].fillna("").astype(str).str.strip()
        groups = groups.where(groups.astype(str).str.strip().ne(""), values)
    return groups


def summarize_person_frames(
    boxes_by_frame: Sequence[Sequence[Sequence[float]]],
    *,
    frame_sizes: Sequence[tuple[int, int]],
    min_person_area_ratio: float = 0.08,
) -> dict[str, float | int | list[float]]:
    """汇总每帧最大人体框占画面面积比例。frame_sizes 为 (width, height)。"""
    if len(boxes_by_frame) != len(frame_sizes):
        raise ValueError("boxes_by_frame 与 frame_sizes 长度不一致")
    largest_ratios: list[float] = []
    for boxes, (width, height) in zip(boxes_by_frame, frame_sizes, strict=True):
        frame_area = max(float(width * height), 1.0)
        ratios = []
        for box in boxes:
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            ratios.append(min(1.0, area / frame_area))
        largest_ratios.append(max(ratios, default=0.0))
    positive = [ratio for ratio in largest_ratios if ratio > 0]
    return {
        "frame_count": len(largest_ratios),
        "person_frames": int(sum(ratio > 0 for ratio in largest_ratios)),
        "large_person_frames": int(
            sum(ratio >= min_person_area_ratio for ratio in largest_ratios)
        ),
        "median_largest_person_ratio": float(median(largest_ratios)),
        "median_visible_person_ratio": float(median(positive)) if positive else 0.0,
        "max_person_ratio": float(max(largest_ratios, default=0.0)),
        "person_area_ratios": largest_ratios,
    }


def classify_multiframe_rule(
    person_summary: dict[str, float | int | list[float]] | None,
    game_scores: Iterable[float] | None,
    *,
    required_large_frames: int = 4,
    game_threshold: float = 0.60,
    required_game_frames: int = 4,
) -> str:
    """4/6 真人多数规则；游戏主体即使有小 facecam 也拒绝。"""
    if person_summary is None:
        return "ERROR"
    scores = (
        np.asarray(list(game_scores), dtype=float)
        if game_scores is not None
        else np.asarray([], dtype=float)
    )
    game_frames = int(np.sum(np.isfinite(scores) & (scores >= game_threshold)))
    if game_frames >= required_game_frames:
        return "F"
    large_frames = int(person_summary.get("large_person_frames", 0))
    if large_frames >= required_large_frames:
        return "T"
    person_frames = int(person_summary.get("person_frames", 0))
    if person_frames <= 1:
        return "F"
    return "U"

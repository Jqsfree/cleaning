"""真人直播视觉过滤：标签、特征存储、阈值、抽样与验收。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd


def normalize_qc_result(values: pd.Series) -> pd.Series:
    """将 bool/T/F/U 等标签归一成 True/False/None。"""
    true_values = {"true", "t", "1", "yes", "pass"}
    false_values = {"false", "f", "0", "no", "fail"}

    def _one(value: object) -> bool | None:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if value is None or pd.isna(value):
            return None
        text = str(value).strip().lower()
        if text in true_values:
            return True
        if text in false_values:
            return False
        return None

    return values.map(_one).astype(object)


def exclude_video_ids(
    frame: pd.DataFrame,
    video_ids: Iterable[str],
) -> pd.DataFrame:
    """排除已参与训练/校准的 video_id。"""
    excluded = {str(value).strip() for value in video_ids}
    ids = frame["video_id"].astype(str).str.strip()
    return frame.loc[~ids.isin(excluded)].copy()


def split_labeled_frame(
    frame: pd.DataFrame,
    *,
    conflict_ids: Iterable[str] = (),
    group_col: str = "channel",
    fallback_group_cols: tuple[str, ...] = ("source_ref",),
    seed: int = 42,
) -> pd.DataFrame:
    """按 group 拆 train/calibration/holdout；冲突样本单列 review。"""
    out = frame.copy()
    out["video_id"] = out["video_id"].astype(str).str.strip()
    out["qc_bool"] = normalize_qc_result(out["qc_result"])
    out["split"] = "unlabeled"
    conflicts = {str(v).strip() for v in conflict_ids}
    out.loc[out["video_id"].isin(conflicts), "split"] = "review"

    usable = out["qc_bool"].notna() & ~out["video_id"].isin(conflicts)
    if not usable.any():
        return out

    groups = pd.Series("", index=out.index, dtype=object)
    for column in (group_col, *fallback_group_cols):
        if column not in out.columns:
            continue
        values = out[column].fillna("").astype(str).str.strip()
        groups = groups.where(groups.astype(str).str.strip().ne(""), values)
    groups = groups.where(groups.astype(str).str.strip().ne(""), out["video_id"])
    out["label_group"] = groups
    groups = groups.loc[usable]
    unique_groups = groups.drop_duplicates().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    if n_groups < 3:
        # 少于三个 group 时退化到逐行稳定切分。
        row_ids = out.index[usable].to_numpy()
        rng.shuffle(row_ids)
        parts = np.array_split(row_ids, 3)
        for name, idx in zip(("train", "calibration", "holdout"), parts):
            out.loc[idx, "split"] = name
        return out

    n_train = max(1, round(n_groups * 0.6))
    n_cal = max(1, round(n_groups * 0.2))
    if n_train + n_cal >= n_groups:
        n_train = n_groups - 2
        n_cal = 1
    mapping: dict[str, str] = {}
    for i, group in enumerate(unique_groups):
        if i < n_train:
            mapping[group] = "train"
        elif i < n_train + n_cal:
            mapping[group] = "calibration"
        else:
            mapping[group] = "holdout"
    out.loc[usable, "split"] = groups.map(mapping).to_numpy()
    return out


def write_embedding_store(
    output_dir: str | Path,
    video_ids: list[str],
    vectors: np.ndarray,
) -> None:
    """写 float16 NPY + 行索引 CSV。"""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    array = np.asarray(vectors, dtype=np.float16)
    if array.ndim != 2 or array.shape[0] != len(video_ids):
        raise ValueError("embedding 行数必须与 video_ids 一致")
    np.save(root / "embeddings.npy", array)
    pd.DataFrame({
        "row": np.arange(len(video_ids), dtype=np.int64),
        "video_id": [str(v).strip() for v in video_ids],
    }).to_csv(root / "index.csv", index=False)
    (root / "meta.json").write_text(
        json.dumps(
            {"rows": len(video_ids), "dim": int(array.shape[1]), "dtype": "float16"},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def load_embedding_rows(
    store_dir: str | Path,
    video_ids: Iterable[str],
) -> tuple[np.ndarray, list[str]]:
    """按请求顺序读取存在的 embedding；跳过缺失 id。"""
    root = Path(store_dir)
    index = pd.read_csv(root / "index.csv", dtype={"video_id": str})
    row_by_id = dict(zip(index["video_id"], index["row"], strict=False))
    requested = [str(v).strip() for v in video_ids]
    found = [vid for vid in requested if vid in row_by_id]
    array = np.load(root / "embeddings.npy", mmap_mode="r")
    if not found:
        return np.empty((0, array.shape[1]), dtype=array.dtype), []
    rows = [int(row_by_id[vid]) for vid in found]
    return np.asarray(array[rows]), found


def choose_action_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    target_pass_rate: float = 0.85,
    max_overturn: float = 0.08,
    min_keep_labels: int = 20,
    min_drop_labels: int = 20,
    min_uncertain_gap: float = 0.20,
) -> dict[str, float | int | str]:
    """从校准标签选择覆盖最大的保守 keep/drop 阈值。"""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) == 0:
        raise ValueError("无可用校准标签")

    keep_options: list[tuple[float, int, float]] = []
    drop_options: list[tuple[float, int, float]] = []
    for threshold in np.unique(p):
        keep = p >= threshold
        if keep.sum() >= min_keep_labels:
            precision = float(y[keep].mean())
            if precision >= target_pass_rate:
                keep_options.append((float(threshold), int(keep.sum()), precision))
        drop = p <= threshold
        if drop.sum() >= min_drop_labels:
            overturn = float(y[drop].mean())
            if overturn <= max_overturn:
                drop_options.append((float(threshold), int(drop.sum()), overturn))

    if keep_options:
        keep_threshold, keep_n, keep_precision = max(
            keep_options, key=lambda item: (item[1], -item[0]),
        )
        keep_method = "target_met"
    else:
        # 无阈值达标时取经验精度最高者，明确标记未达标。
        candidates = []
        for threshold in np.unique(p):
            keep = p >= threshold
            if keep.sum() >= max(1, min(min_keep_labels, len(y))):
                candidates.append(
                    (float(threshold), int(keep.sum()), float(y[keep].mean())),
                )
        keep_threshold, keep_n, keep_precision = max(
            candidates, key=lambda item: (item[2], item[1]),
        )
        keep_method = "best_effort"

    if drop_options:
        drop_threshold, drop_n, drop_overturn = max(
            drop_options, key=lambda item: (item[1], item[0]),
        )
        drop_method = "target_met"
    else:
        candidates = []
        for threshold in np.unique(p):
            drop = p <= threshold
            if drop.sum() >= max(1, min(min_drop_labels, len(y))):
                candidates.append(
                    (float(threshold), int(drop.sum()), float(y[drop].mean())),
                )
        drop_threshold, drop_n, drop_overturn = min(
            candidates, key=lambda item: (item[2], -item[1]),
        )
        drop_method = "best_effort"

    if keep_threshold - drop_threshold < min_uncertain_gap:
        midpoint = float((drop_threshold + keep_threshold) / 2)
        half_gap = min_uncertain_gap / 2
        drop_threshold = max(0.0, midpoint - half_gap)
        keep_threshold = min(1.0, midpoint + half_gap)

    return {
        "keep_threshold": keep_threshold,
        "drop_threshold": drop_threshold,
        "keep_n": keep_n,
        "drop_n": drop_n,
        "keep_precision": keep_precision,
        "drop_overturn": drop_overturn,
        "keep_method": keep_method,
        "drop_method": drop_method,
    }


def build_feature_matrix(
    embeddings: np.ndarray,
    *,
    pos_sim: Iterable[float] | None = None,
    neg_sim: Iterable[float] | None = None,
    duration_seconds: Iterable[float] | None = None,
) -> np.ndarray:
    """CLIP embedding + 两个相似度 + log 时长；不含来源/频道文本。"""
    base = np.asarray(embeddings, dtype=np.float32)
    if base.ndim != 2:
        raise ValueError("embeddings 必须是二维数组")
    n = len(base)

    def _column(values: Iterable[float] | None) -> np.ndarray:
        if values is None:
            return np.full((n, 1), np.nan, dtype=np.float32)
        array = np.asarray(list(values), dtype=np.float32)
        if len(array) != n:
            raise ValueError("附加特征行数与 embeddings 不一致")
        return array.reshape(-1, 1)

    duration = _column(duration_seconds)
    duration = np.log1p(np.maximum(duration, 0))
    return np.hstack([
        base,
        _column(pos_sim),
        _column(neg_sim),
        duration,
    ])


def train_grouped_visual_model(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
    sample_weight: np.ndarray | None = None,
):
    """按 group 生成 OOF 概率，并在全部标签上拟合最终 LR。"""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=int)
    group_array = np.asarray(groups).astype(str)
    if len(x) != len(y) or len(y) != len(group_array):
        raise ValueError("features/labels/groups 行数不一致")
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights is not None and len(weights) != len(y):
        raise ValueError("sample_weight 行数不一致")
    if len(np.unique(y)) < 2:
        raise ValueError("训练标签必须同时包含 T/F")
    split_count = min(
        n_splits,
        len(np.unique(group_array)),
        int(np.bincount(y).min()),
    )
    if split_count < 2:
        raise ValueError("group 或少数类不足，无法交叉验证")

    def _pipeline():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=seed,
                ),
            ),
        ])

    oof = np.full(len(y), np.nan, dtype=np.float64)
    splitter = StratifiedGroupKFold(
        n_splits=split_count, shuffle=True, random_state=seed,
    )
    for train_idx, valid_idx in splitter.split(x, y, group_array):
        model = _pipeline()
        fit_kwargs = {}
        if weights is not None:
            fit_kwargs["model__sample_weight"] = weights[train_idx]
        model.fit(x[train_idx], y[train_idx], **fit_kwargs)
        oof[valid_idx] = model.predict_proba(x[valid_idx])[:, 1]
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF 概率不完整")
    final_model = _pipeline()
    fit_kwargs = {}
    if weights is not None:
        fit_kwargs["model__sample_weight"] = weights
    final_model.fit(x, y, **fit_kwargs)
    return final_model, oof


def assign_actions(
    probabilities: Iterable[float],
    *,
    keep_threshold: float,
    drop_threshold: float,
) -> np.ndarray:
    p = np.asarray(list(probabilities), dtype=float)
    action = np.full(len(p), "uncertain", dtype=object)
    action[p >= keep_threshold] = "keep_candidate"
    action[p <= drop_threshold] = "highconf_drop"
    action[~np.isfinite(p)] = "uncertain"
    return action


def _farthest_points(vectors: np.ndarray, count: int, seed: int) -> list[int]:
    if count <= 0 or len(vectors) == 0:
        return []
    count = min(count, len(vectors))
    x = np.asarray(vectors, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norm, 1e-8)
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(len(x)))]
    min_distance = 1 - x @ x[selected[0]]
    while len(selected) < count:
        next_idx = int(np.argmax(min_distance))
        selected.append(next_idx)
        min_distance = np.minimum(min_distance, 1 - x @ x[next_idx])
    return selected


def select_active_learning_sample(
    scored: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    n_boundary: int = 150,
    n_diverse_keep: int = 100,
    n_drop: int = 50,
    seed: int = 42,
    max_diverse_pool: int = 10_000,
) -> pd.DataFrame:
    """抽边界、多样 keep 与 drop overturn，且三路不重复。"""
    if len(scored) != len(embeddings):
        raise ValueError("scored 与 embeddings 行数不一致")
    work = scored.reset_index(drop=True).copy()
    chosen: list[pd.DataFrame] = []
    used: set[int] = set()

    uncertain = work.index[work["ml_action"].eq("uncertain")].tolist()
    uncertain.sort(key=lambda i: abs(float(work.at[i, "visual_prob"]) - 0.5))
    boundary_idx = uncertain[:n_boundary]
    if boundary_idx:
        part = work.loc[boundary_idx].copy()
        part["sample_route"] = "boundary"
        chosen.append(part)
        used.update(boundary_idx)

    keep_idx = [
        i for i in work.index[work["ml_action"].eq("keep_candidate")].tolist()
        if i not in used
    ]
    rng = np.random.default_rng(seed)
    if len(keep_idx) > max_diverse_pool:
        keep_idx = rng.choice(
            keep_idx, size=max_diverse_pool, replace=False,
        ).tolist()
    local = _farthest_points(embeddings[keep_idx], n_diverse_keep, seed)
    diverse_idx = [keep_idx[i] for i in local]
    if diverse_idx:
        part = work.loc[diverse_idx].copy()
        part["sample_route"] = "diverse_keep"
        chosen.append(part)
        used.update(diverse_idx)

    drop_idx = [
        i for i in work.index[work["ml_action"].eq("highconf_drop")].tolist()
        if i not in used
    ]
    if drop_idx:
        selected_drop = rng.choice(
            drop_idx, size=min(n_drop, len(drop_idx)), replace=False,
        ).tolist()
        part = work.loc[selected_drop].copy()
        part["sample_route"] = "drop_overturn"
        chosen.append(part)

    if not chosen:
        return work.head(0).assign(sample_route=pd.Series(dtype=str))
    return pd.concat(chosen, ignore_index=True)


def wilson_lower_bound(
    positives: int,
    total: int,
    *,
    confidence: float = 0.90,
) -> float:
    if total <= 0:
        return 0.0
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = positives / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (centre - radius) / denom)


def acceptance_decision(
    *,
    pass_count: int,
    labeled_count: int,
    kept_hours: float,
    overturn_count: int,
    drop_labeled_count: int,
    confidence: float = 0.90,
    min_pass_lower: float = 0.85,
    min_hours: float = 80_000,
    max_overturn: float = 0.08,
) -> dict[str, float | int | str | bool]:
    lower = wilson_lower_bound(pass_count, labeled_count, confidence=confidence)
    overturn = (
        overturn_count / drop_labeled_count if drop_labeled_count > 0 else 1.0
    )
    pass_ok = lower >= min_pass_lower
    hours_ok = kept_hours >= min_hours
    overturn_ok = overturn <= max_overturn
    return {
        "decision": "accept" if pass_ok and hours_ok and overturn_ok else "reject",
        "pass_rate": pass_count / labeled_count if labeled_count else 0.0,
        "pass_lower": lower,
        "kept_hours": float(kept_hours),
        "overturn_rate": overturn,
        "pass_ok": pass_ok,
        "hours_ok": hours_ok,
        "overturn_ok": overturn_ok,
    }

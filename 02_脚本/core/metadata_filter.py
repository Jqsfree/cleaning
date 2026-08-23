"""Stage -1 元数据文本过滤器：与 train_metadata_filter.py 特征构造完全一致。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "models/exo_agriculture_metadata_filter"
)
DEFAULT_THRESHOLD = 0.3
DEFAULT_EXPECTED_T_REJECT_RATE = 0.07
DEFAULT_MAX_T_REJECT_RATE = 0.10


def build_text_field(df: pd.DataFrame) -> pd.Series:
    """keyword 重复两遍加权 + title（与训练脚本一致；不含 channel）。"""
    keyword = df.get("keyword", pd.Series("", index=df.index)).fillna("").astype(str)
    title = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
    return (keyword + " " + keyword + " " + title).str.lower()


def build_numeric_features(df: pd.DataFrame) -> np.ndarray:
    dur = np.log1p(
        pd.to_numeric(df.get("duration_seconds"), errors="coerce").fillna(0).to_numpy(dtype=float),
    )
    view = np.log1p(
        pd.to_numeric(df.get("view_count"), errors="coerce").fillna(0).to_numpy(dtype=float),
    )
    return np.column_stack([dur, view])


def normalize_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "video_id" not in out.columns:
        raise ValueError("输入需含 video_id")
    out["video_id"] = out["video_id"].astype(str).str.strip()
    for col in ("title", "keyword", "channel", "description"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    out["duration_seconds"] = pd.to_numeric(out.get("duration_seconds"), errors="coerce").fillna(0)
    out["view_count"] = pd.to_numeric(out.get("view_count"), errors="coerce").fillna(0)
    return out


def load_blocklist_keywords(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.is_file():
        return set()
    df = pd.read_csv(p, dtype=str)
    col = "keyword" if "keyword" in df.columns else df.columns[0]
    return {
        str(v).strip().lower()
        for v in df[col].dropna()
        if str(v).strip()
    }


def load_metadata_models(model_dir: str | Path) -> tuple[Any, Any, Any]:
    root = Path(model_dir)
    clf = joblib.load(root / "clf.joblib")
    vectorizer = joblib.load(root / "vectorizer.joblib")
    scaler = joblib.load(root / "scaler.joblib")
    return clf, vectorizer, scaler


def score_metadata_frame(
    df: pd.DataFrame,
    *,
    model_dir: str | Path,
) -> np.ndarray:
    """返回每条候选的 P(T) 分数。"""
    frame = normalize_input_frame(df)
    text = build_text_field(frame)
    num = build_numeric_features(frame)
    _, vectorizer, scaler = load_metadata_models(model_dir)
    x_text = vectorizer.transform(text)
    x_num = csr_matrix(scaler.transform(num))
    x = hstack([x_text, x_num])
    clf, _, _ = load_metadata_models(model_dir)
    return clf.predict_proba(x)[:, 1].astype(np.float64)


def keyword_bucket(keyword: object) -> str:
    text = str(keyword or "").lower()
    if re.search(r"plant|sow|seed|grow|种植|育苗|播种", text):
        return "planting"
    if re.search(r"harvest|pick|采摘|收割|picking", text):
        return "harvesting"
    return "other"


def assign_metadata_actions(
    df: pd.DataFrame,
    scores: np.ndarray,
    *,
    blocklist: set[str],
    threshold: float,
) -> pd.DataFrame:
    frame = normalize_input_frame(df)
    out = frame.copy()
    out["metadata_score"] = scores
    kw_norm = out["keyword"].astype(str).str.strip().str.lower()
    block_mask = kw_norm.isin(blocklist)
    score_mask = out["metadata_score"].to_numpy(dtype=float) < threshold
    action = np.full(len(out), "metadata_pass", dtype=object)
    reason = np.full(len(out), "", dtype=object)
    action[block_mask] = "metadata_reject"
    reason[block_mask] = "blocklist_keyword"
    low_score = score_mask & ~block_mask
    action[low_score] = "metadata_reject"
    reason[low_score] = "score_below_threshold"
    out["metadata_action"] = action
    out["metadata_reject_reason"] = reason
    return out


def bucket_stats(frame: pd.DataFrame, *, action_col: str = "metadata_action") -> dict[str, dict]:
    work = frame.copy()
    work["keyword_bucket"] = work["keyword"].map(keyword_bucket)
    stats: dict[str, dict] = {}
    for bucket, part in work.groupby("keyword_bucket", dropna=False):
        reject = part[action_col].eq("metadata_reject")
        stats[str(bucket)] = {
            "n": int(len(part)),
            "n_reject": int(reject.sum()),
            "reject_rate": round(float(reject.mean()), 4) if len(part) else 0.0,
            "n_pass": int((~reject).sum()),
        }
    return stats


def load_calibration(model_dir: str | Path) -> dict:
    path = Path(model_dir) / "calibration.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "threshold": DEFAULT_THRESHOLD,
        "expected_positive_reject_rate": DEFAULT_EXPECTED_T_REJECT_RATE,
        "max_positive_reject_rate": DEFAULT_MAX_T_REJECT_RATE,
        "expected_positive_recall": 0.93,
        "notes": "阈值0.3：约保留93%正样本，滤约31%负样本；正样本误杀约7%",
    }


def eval_reject_overturn(
    labeled: pd.DataFrame,
    scored: pd.DataFrame,
    *,
    label_col: str = "y",
) -> dict:
    merged = labeled.merge(
        scored[["video_id", "metadata_action", "metadata_score", "metadata_reject_reason"]],
        on="video_id",
        how="inner",
    )
    y = merged[label_col].astype(int).to_numpy()
    reject = merged["metadata_action"].eq("metadata_reject").to_numpy()
    n_t = int((y == 1).sum())
    t_hurt = int((y[reject] == 1).sum()) if reject.any() else 0
    return {
        "n_labeled": int(len(merged)),
        "n_t": n_t,
        "n_f": int((y == 0).sum()),
        "n_reject": int(reject.sum()),
        "t_hurt": t_hurt,
        "t_hurt_rate": round(t_hurt / max(n_t, 1), 4),
        "f_caught": int((y[reject] == 0).sum()) if reject.any() else 0,
    }

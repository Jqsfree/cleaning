#!/usr/bin/env python3
"""真人直播多帧规则、VLM 与文本视觉融合离线对比。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "02_脚本"))
from core.visual_filter import assign_actions, choose_action_thresholds  # noqa: E402

NUMERIC_COLUMNS = [
    "person_frames",
    "large_person_frames",
    "median_largest_person_ratio",
    "median_visible_person_ratio",
    "max_person_ratio",
    "game_dominant_frames",
    "siglip_human_live_mean",
    "siglip_human_live_max",
    "siglip_talking_mean",
    "siglip_talking_max",
    "siglip_irl_mean",
    "siglip_irl_max",
    "siglip_game_mean",
    "siglip_game_max",
    "siglip_studio_mean",
    "siglip_studio_max",
    "siglip_other_media_mean",
    "siglip_other_media_max",
    "sim_score",
    "neg_sim",
    "duration_seconds",
]


def _text(row: pd.Series) -> str:
    parts = []
    for column in ("title", "keyword", "channel", "source_ref"):
        value = row.get(column, "")
        if pd.notna(value) and str(value).strip():
            parts.append(str(value).strip())
    return re.sub(r"\s+", " ", " ".join(parts))


def _feature_pipeline(seed: int) -> ColumnTransformer:
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=20_000,
        strip_accents="unicode",
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=20_000,
    )
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        [
            ("word", word, "model_text"),
            ("char", char, "model_text"),
            ("numeric", numeric, NUMERIC_COLUMNS),
        ],
        sparse_threshold=0.3,
    )


def _fit_one(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    seed: int,
) -> tuple[np.ndarray, object, object]:
    features = _feature_pipeline(seed)
    x_train = features.fit_transform(train)
    x_valid = features.transform(valid)
    if not sparse.issparse(x_train):
        x_train = sparse.csr_matrix(x_train)
        x_valid = sparse.csr_matrix(x_valid)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2_000,
        random_state=seed,
    )
    model.fit(x_train, train["y"].to_numpy(dtype=int))
    return model.predict_proba(x_valid)[:, 1], features, model


def _machine_oof(
    frame: pd.DataFrame,
    *,
    seed: int,
) -> np.ndarray:
    y = frame["y"].to_numpy(dtype=int)
    groups = frame["label_group"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )
    oof = np.full(len(frame), np.nan, dtype=float)
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(frame, y, groups),
        1,
    ):
        probability, _, _ = _fit_one(
            frame.iloc[train_idx],
            frame.iloc[valid_idx],
            seed=seed + fold,
        )
        oof[valid_idx] = probability
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF 概率不完整")
    return oof


def _action_metrics(
    frame: pd.DataFrame,
    action_column: str,
) -> dict[str, float | int]:
    keep = frame[action_column].eq("keep_candidate")
    drop = frame[action_column].eq("highconf_drop")
    duration = pd.to_numeric(frame["duration_seconds"], errors="coerce").fillna(0)
    total_hours = float(duration.sum() / 3600)
    keep_hours = float(duration[keep].sum() / 3600)
    return {
        "rows": len(frame),
        "keep_n": int(keep.sum()),
        "keep_precision": float(frame.loc[keep, "y"].mean()) if keep.any() else 0.0,
        "keep_hours": keep_hours,
        "keep_hour_share": keep_hours / total_hours if total_hours else 0.0,
        "drop_n": int(drop.sum()),
        "drop_overturn": float(frame.loc[drop, "y"].mean()) if drop.any() else 1.0,
        "total_hours": total_hours,
    }


def _direct_metrics(frame: pd.DataFrame, prediction: pd.Series) -> dict:
    valid = prediction.isin(["T", "F"])
    work = frame.loc[valid].copy()
    predicted = prediction.loc[valid].eq("T")
    return {
        "rows": len(frame),
        "valid": int(valid.sum()),
        "coverage": float(valid.mean()) if len(valid) else 0.0,
        "precision_t": float(work.loc[predicted, "y"].mean()) if predicted.any() else 0.0,
        "predicted_t": int(predicted.sum()),
        "overturn_f": float(work.loc[~predicted, "y"].mean()) if (~predicted).any() else 1.0,
        "predicted_f": int((~predicted).sum()),
        "accuracy": float(
            (predicted.to_numpy() == work["y"].to_numpy(dtype=bool)).mean()
        ) if len(work) else 0.0,
    }


def _cluster(row: pd.Series) -> str:
    title = str(row.get("title", "")).lower()
    game = float(row.get("siglip_game_mean", 0) or 0)
    people = int(row.get("person_frames", 0) or 0)
    if game >= 0.45 and people >= 2:
        return "small_facecam_game"
    if game >= 0.45:
        return "game_no_person"
    if re.search(r"\breaction\b|\breacts?\b", title):
        return "reaction_clip"
    if re.search(r"podcast|interview|talk show|late show", title):
        return "podcast_interview"
    if re.search(r"lesson|class|teaching|tutorial|lecture", title):
        return "teaching"
    if re.search(r"\birl\b|just chatting|webcam|livestream|live stream", title):
        return "irl_live"
    return "other"


def _read_scores(path: str | None, columns: list[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype={"video_id": str}, usecols=lambda c: c in columns)
    return frame.drop_duplicates("video_id", keep="last")


def main() -> None:
    parser = argparse.ArgumentParser(description="真人直播多帧离线 benchmark")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--vlm-scores")
    parser.add_argument("--pos-sim", action="append", default=[])
    parser.add_argument("--neg-sim")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.labels, dtype=str, low_memory=False)
    labels = labels[labels["split"].isin(["train", "calibration", "holdout"])].copy()
    labels["y"] = labels["qc_bool"].astype(str).str.lower().eq("true").astype(int)
    labels["model_text"] = labels.apply(_text, axis=1)
    features = pd.read_parquet(args.features)
    work = labels.merge(features, on="video_id", how="inner", validate="one_to_one")
    pos_parts = [
        _read_scores(path, ["video_id", "sim_score"])
        for path in args.pos_sim
    ]
    pos = (
        pd.concat(pos_parts, ignore_index=True)
        .drop_duplicates("video_id", keep="last")
        if pos_parts else pd.DataFrame(columns=["video_id", "sim_score"])
    )
    neg = _read_scores(args.neg_sim, ["video_id", "neg_sim"])
    work = work.merge(pos, on="video_id", how="left")
    work = work.merge(neg, on="video_id", how="left")
    for column in NUMERIC_COLUMNS:
        work[column] = pd.to_numeric(work.get(column), errors="coerce")

    machine_fit = work[
        work["source"].eq("machine")
        & work["split"].isin(["train", "calibration"])
    ].copy().reset_index(drop=True)
    machine_holdout = work[
        work["source"].eq("machine")
        & work["split"].eq("holdout")
    ].copy().reset_index(drop=True)
    human_holdout = work[
        work["source"].eq("human")
        & work["split"].eq("holdout")
    ].copy().reset_index(drop=True)

    oof = _machine_oof(machine_fit, seed=args.seed)
    thresholds = choose_action_thresholds(
        machine_fit["y"].to_numpy(dtype=int),
        oof,
        target_pass_rate=0.90,
        max_overturn=0.08,
        min_keep_labels=50,
        min_drop_labels=50,
    )
    holdout_prob, transformer, model = _fit_one(
        machine_fit,
        machine_holdout,
        seed=args.seed,
    )
    machine_holdout["fusion_prob"] = holdout_prob
    machine_holdout["fusion_action"] = assign_actions(
        holdout_prob,
        keep_threshold=float(thresholds["keep_threshold"]),
        drop_threshold=float(thresholds["drop_threshold"]),
    )
    fusion_metrics = {
        "machine_oof_auc": float(
            roc_auc_score(machine_fit["y"], oof),
        ),
        "machine_holdout_auc": float(
            roc_auc_score(machine_holdout["y"], holdout_prob),
        ),
        "thresholds": thresholds,
        "machine_holdout_actions": _action_metrics(
            machine_holdout,
            "fusion_action",
        ),
    }
    if len(human_holdout) and human_holdout["y"].nunique() == 2:
        human_x = transformer.transform(human_holdout)
        human_prob = model.predict_proba(human_x)[:, 1]
        fusion_metrics["human_holdout_auc"] = float(
            roc_auc_score(human_holdout["y"], human_prob),
        )

    rule_metrics = _direct_metrics(
        machine_holdout,
        machine_holdout["multiframe_rule"],
    )
    vlm_metrics: dict = {
        "status": "unavailable",
        "reason": (
            "DASHSCOPE_API_KEY 未设置，Qwen 未调用；本地 SmolVLM2 "
            "fallback 缺少 num2words，未用代理结果冒充 Qwen"
        ),
    }
    if args.vlm_scores and Path(args.vlm_scores).exists():
        vlm = pd.read_csv(args.vlm_scores, dtype=str)
        machine_holdout = machine_holdout.merge(
            vlm[["video_id", "vlm_result"]],
            on="video_id",
            how="left",
        )
        vlm_metrics = {
            "status": "local_proxy",
            "metrics": _direct_metrics(
                machine_holdout,
                machine_holdout["vlm_result"].fillna("ERROR"),
            ),
            "note": "DashScope key unavailable; local SmolVLM2 is the direct-VLM proxy",
        }

    machine_holdout["error_cluster"] = machine_holdout.apply(_cluster, axis=1)
    errors = machine_holdout[
        (
            machine_holdout["fusion_action"].eq("keep_candidate")
            & machine_holdout["y"].eq(0)
        )
        | (
            machine_holdout["fusion_action"].eq("highconf_drop")
            & machine_holdout["y"].eq(1)
        )
    ].copy()
    errors.to_csv(out / "fusion_holdout_errors.csv", index=False)
    machine_holdout.to_csv(out / "machine_holdout_scored.csv", index=False)
    cluster_metrics = (
        machine_holdout.groupby("error_cluster")
        .agg(rows=("video_id", "size"), pass_rate=("y", "mean"))
        .reset_index()
        .to_dict("records")
    )
    action = fusion_metrics["machine_holdout_actions"]
    go = bool(
        fusion_metrics["machine_holdout_auc"] >= 0.85
        and action["keep_n"] >= 50
        and action["keep_precision"] >= 0.90
        and action["drop_n"] >= 50
        and action["drop_overturn"] <= 0.08
        and thresholds["keep_method"] == "target_met"
        and thresholds["drop_method"] == "target_met"
    )
    report = {
        "coverage": {
            "labels": len(labels),
            "storyboard_features": len(features),
            "joined": len(work),
            "machine_fit": len(machine_fit),
            "machine_holdout": len(machine_holdout),
            "human_holdout": len(human_holdout),
        },
        "rule": rule_metrics,
        "vlm": vlm_metrics,
        "fusion": fusion_metrics,
        "clusters": cluster_metrics,
        "go_full_pool": go,
        "go_criteria": {
            "machine_holdout_auc": 0.85,
            "keep_precision": 0.90,
            "keep_min_n": 50,
            "drop_max_overturn": 0.08,
            "drop_min_n": 50,
        },
    }
    (out / "benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

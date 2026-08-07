#!/usr/bin/env python3
"""冻结 CLIP embedding 的真人直播 Logistic Regression 训练器。"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "02_脚本"))
from core.visual_filter import (  # noqa: E402
    build_feature_matrix,
    choose_action_thresholds,
    load_embedding_rows,
    train_grouped_visual_model,
)


def _read_optional_scores(path: str | None, columns: list[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype={"video_id": str}, low_memory=False)
    return frame[[c for c in columns if c in frame.columns]].copy()


def _features(
    labels: pd.DataFrame,
    store: str,
    pos_scores: pd.DataFrame,
    neg_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    ids = labels["video_id"].astype(str).str.strip().tolist()
    embeddings, found = load_embedding_rows(store, ids)
    found_set = set(found)
    work = labels[labels["video_id"].astype(str).str.strip().isin(found_set)].copy()
    order = {video_id: i for i, video_id in enumerate(found)}
    work["_order"] = work["video_id"].map(order)
    work = work.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    if len(work) != len(embeddings):
        raise RuntimeError("标签与 embedding 对齐失败")
    work = work.merge(pos_scores, on="video_id", how="left")
    work = work.merge(neg_scores, on="video_id", how="left")
    features = build_feature_matrix(
        embeddings,
        pos_sim=pd.to_numeric(work.get("sim_score"), errors="coerce"),
        neg_sim=pd.to_numeric(work.get("neg_sim"), errors="coerce"),
        duration_seconds=pd.to_numeric(work.get("duration_seconds"), errors="coerce"),
    )
    return work, features


def _source_weights(source: pd.Series) -> np.ndarray:
    counts = source.fillna("unknown").astype(str).value_counts()
    return source.fillna("unknown").astype(str).map(
        {key: len(source) / (len(counts) * count) for key, count in counts.items()}
    ).to_numpy(dtype=float)


def _source_oof_profiles(
    y: np.ndarray,
    probabilities: np.ndarray,
    source: pd.Series,
    *,
    target_pass_rate: float,
    max_overturn: float,
    min_labels: int = 10,
) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    source_values = source.fillna("unknown").astype(str).to_numpy()
    for source_name in sorted(set(source_values)):
        mask = source_values == source_name
        if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
            continue
        profiles[source_name] = choose_action_thresholds(
            y[mask],
            probabilities[mask],
            target_pass_rate=target_pass_rate,
            max_overturn=max_overturn,
            min_keep_labels=min(min_labels, int(mask.sum())),
            min_drop_labels=min(min_labels, int(mask.sum())),
        )
        profiles[source_name]["oof_rows"] = int(mask.sum())
    return profiles


def _select_candidate_name(machine_only: dict, mixed: dict) -> str:
    mixed_profile = mixed.get("profiles", {}).get("machine", {})
    targets_met = (
        mixed_profile.get("keep_method") == "target_met"
        and mixed_profile.get("drop_method") == "target_met"
    )
    baseline_auc = float(machine_only.get("machine_oof_auc", float("-inf")))
    mixed_auc = float(mixed.get("machine_oof_auc", float("-inf")))
    return "mixed" if targets_met and mixed_auc > baseline_auc else "machine_only"


def _train_candidate(
    *,
    features: np.ndarray,
    y: np.ndarray,
    groups: pd.Series,
    source: pd.Series,
    mask: np.ndarray,
    weighted: bool,
    target_pass_rate: float,
    max_overturn: float,
    seed: int,
) -> dict:
    candidate_source = source[mask].reset_index(drop=True)
    model, oof = train_grouped_visual_model(
        features[mask],
        y[mask],
        groups[mask].to_numpy(),
        n_splits=5,
        seed=seed,
        sample_weight=_source_weights(candidate_source) if weighted else None,
    )
    profiles = _source_oof_profiles(
        y[mask],
        oof,
        candidate_source,
        target_pass_rate=target_pass_rate,
        max_overturn=max_overturn,
    )
    machine_mask = candidate_source.eq("machine").to_numpy()
    machine_auc = (
        float(roc_auc_score(y[mask][machine_mask], oof[machine_mask]))
        if machine_mask.any() and len(np.unique(y[mask][machine_mask])) == 2
        else float("nan")
    )
    return {
        "model": model,
        "profiles": profiles,
        "machine_oof_auc": machine_auc,
        "fit_rows": int(mask.sum()),
        "sources": candidate_source.value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 human_live 视觉过滤器")
    parser.add_argument("--labels", required=True, help="labels_prepared.csv")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--pos-sim")
    parser.add_argument("--neg-sim")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--target-pass-rate", type=float, default=0.85)
    parser.add_argument("--max-overturn", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.labels, dtype=str, low_memory=False)
    labels["video_id"] = labels["video_id"].astype(str).str.strip()
    labels["qc_bool"] = labels["qc_bool"].astype(str).str.lower().map(
        {"true": True, "false": False},
    )
    labels = labels[labels["split"].isin(["train", "calibration", "holdout"])].copy()
    pos = _read_optional_scores(args.pos_sim, ["video_id", "sim_score"])
    neg = _read_optional_scores(args.neg_sim, ["video_id", "neg_sim"])
    work, features = _features(labels, args.embeddings, pos, neg)
    y = work["qc_bool"].astype(bool).astype(int).to_numpy()
    groups = work.get("label_group", work["video_id"]).fillna(work["video_id"]).astype(str)
    source = work.get("source", pd.Series("machine", index=work.index)).fillna("machine")

    train_mask = work["split"].eq("train").to_numpy()
    cal_mask = work["split"].eq("calibration").to_numpy()
    holdout_mask = work["split"].eq("holdout").to_numpy()
    if train_mask.sum() < 20 or len(np.unique(y[train_mask])) < 2:
        raise SystemExit("[ERROR] train 标签不足或缺少 T/F")

    fit_mask = train_mask | cal_mask
    machine_fit_mask = fit_mask & source.eq("machine").to_numpy()
    if machine_fit_mask.sum() < 20 or len(np.unique(y[machine_fit_mask])) < 2:
        raise SystemExit("[ERROR] machine 标签不足，无法训练 machine-only 基线")
    candidates = {
        "machine_only": _train_candidate(
            features=features,
            y=y,
            groups=groups,
            source=source,
            mask=machine_fit_mask,
            weighted=False,
            target_pass_rate=args.target_pass_rate,
            max_overturn=args.max_overturn,
            seed=args.seed,
        ),
        "mixed": _train_candidate(
            features=features,
            y=y,
            groups=groups,
            source=source,
            mask=fit_mask,
            weighted=True,
            target_pass_rate=args.target_pass_rate,
            max_overturn=args.max_overturn,
            seed=args.seed,
        ),
    }
    selected_name = _select_candidate_name(
        candidates["machine_only"], candidates["mixed"],
    )
    selected = candidates[selected_name]
    metrics: dict[str, object] = {
        "train_rows": int(train_mask.sum()),
        "calibration_rows": int(cal_mask.sum()),
        "holdout_rows": int(holdout_mask.sum()),
        "selected_candidate": selected_name,
        "selection_rule": (
            "mixed only when machine OOF AUC improves and keep/drop targets are met"
        ),
    }
    if holdout_mask.any():
        holdout_prob = selected["model"].predict_proba(features[holdout_mask])[:, 1]
        for source_name in sorted(source[holdout_mask].astype(str).unique()):
            source_mask = source[holdout_mask].astype(str).eq(source_name).to_numpy()
            source_y = y[holdout_mask][source_mask]
            if len(np.unique(source_y)) != 2:
                continue
            metrics[f"holdout_{source_name}"] = {
                "rows": int(source_mask.sum()),
                "auc": float(roc_auc_score(source_y, holdout_prob[source_mask])),
            }
    candidate_reports = {
        name: {
            key: value
            for key, value in candidate.items()
            if key != "model"
        }
        for name, candidate in candidates.items()
    }
    common_artifact = {
        "default_source": "machine",
        "feature_layout": "clip_embedding,pos_sim,neg_sim,log1p_duration",
        "embedding_dim": int(features.shape[1] - 3),
        "metrics": metrics,
        "candidates": candidate_reports,
    }
    for name, candidate in candidates.items():
        candidate_artifact = {
            **common_artifact,
            "model": candidate["model"],
            "profiles": candidate["profiles"],
            "candidate": name,
        }
        with (out / f"human_live_visual_lr_{name}.pkl").open("wb") as handle:
            pickle.dump(candidate_artifact, handle)
    artifact = {
        **common_artifact,
        "model": selected["model"],
        "profiles": selected["profiles"],
        "candidate": selected_name,
    }
    with (out / "human_live_visual_lr.pkl").open("wb") as handle:
        pickle.dump(artifact, handle)
    report = {
        "labels": str(Path(args.labels).resolve()),
        "embeddings": str(Path(args.embeddings).resolve()),
        "selected_candidate": selected_name,
        "profiles": selected["profiles"],
        "candidates": candidate_reports,
        "metrics": metrics,
        "sources": source.value_counts().to_dict(),
        "note": "来源仅用于权重、模型比较与分来源 OOF 阈值，不作为模型特征",
    }
    (out / "calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

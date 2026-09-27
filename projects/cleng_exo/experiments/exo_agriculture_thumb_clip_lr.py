#!/usr/bin/env python3
"""exo_agriculture 缩略图 CLIP embedding + LogisticRegression（channel group OOF）。

特征来自 0814 embedding store（免重编码）；人工 T/F → balanced LR。
闸门：OOF 上 T 误伤≤2% 且 drop precision≥0.9，最大化 F 召回。

用法:
  PYTHONPATH=02_脚本:experiments python experiments/exo_agriculture_thumb_clip_lr.py --calibrate \\
    --labels data/runs/exo_agriculture/machine_0814/03_qc/human270_v04_1be48617/exo农业_human_qc.csv \\
    --extra-labels data/runs/exo_agriculture/machine_0814/03_qc/human678_merged/labeled.csv \\
    -o data/runs/exo_agriculture/machine_0814/06_tools/clip_lr_v1/
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = PROJECT / "02_脚本"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT / "experiments"))

from core.visual_filter import (  # noqa: E402
    assign_actions,
    load_embedding_rows,
    train_grouped_visual_model,
)

DEFAULT_LABELS = (
    PROJECT
    / "data/runs/exo_agriculture/machine_0814/03_qc/human270_v04_1be48617/exo农业_human_qc.csv"
)
DEFAULT_EXTRA = (
    PROJECT
    / "data/runs/exo_agriculture/machine_0814/03_qc/human678_merged/labeled.csv"
)
DEFAULT_OUT = PROJECT / "data/runs/exo_agriculture/machine_0814/06_tools/clip_lr_v1"
DEFAULT_STORE = PROJECT / "data/assets/embeddings/exo_agriculture_0814_semantic_remain"
MODEL_PATH = PROJECT / "models/exo_agriculture_thumb_clip_lr.pkl"
CALIB_PATH = PROJECT / "models/exo_agriculture_thumb_clip_lr_calibration.json"


def normalize_label(value: object) -> str | None:
    v = str(value or "").strip().upper()
    if not v or v == "NAN":
        return None
    if v in {"T", "PASS"}:
        return "T"
    if v.startswith("F"):
        if "无法播放" in str(value) or "UNPLAYABLE" in v:
            return None
        return "F"
    return None


def label_from_labeled_row(row: pd.Series) -> str | None:
    if "human_label" in row.index:
        v = str(row.get("human_label", "") or "").strip().lower()
        if v == "pass":
            return "T"
        if v == "fail":
            return "F"
    if "qc_result" in row.index:
        return normalize_label(row.get("qc_result"))
    return None


def group_id(row: pd.Series) -> str:
    for col in ("channel", "source_ref", "video_id"):
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            return str(val).strip()
    return str(row.get("video_id", ""))


def load_label_frame(path: Path, *, tag: str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, low_memory=False)
    rows: list[dict] = []
    for _, r in raw.iterrows():
        lab = label_from_labeled_row(r)
        vid = str(r.get("video_id", "") or "").strip()
        if not vid or lab is None:
            continue
        rows.append({
            "video_id": vid,
            "title": r.get("title", ""),
            "channel": r.get("channel", ""),
            "source_ref": r.get("source_ref", ""),
            "duration_seconds": r.get("duration_seconds", ""),
            "label": lab,
            "y": 1 if lab == "T" else 0,
            "group_id": group_id(r),
            "label_source": tag,
        })
    return pd.DataFrame(rows).drop_duplicates("video_id", keep="first").reset_index(drop=True)


def pick_drop_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    max_t_hurt_rate: float = 0.02,
    max_t_hurt_abs: int = 2,
    min_drop_precision: float = 0.9,
    min_drop_labels: int = 5,
) -> dict:
    """选高置信 drop 阈值：p<=thr 丢弃；precision=丢弃中 F 占比。"""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(p)
    y, p = y[valid], p[valid]
    n_f = int((y == 0).sum())
    options: list[dict] = []
    for threshold in np.unique(p):
        drop = p <= threshold
        n_drop = int(drop.sum())
        if n_drop < min_drop_labels:
            continue
        t_hurt = int((y[drop] == 1).sum())
        t_hurt_rate = t_hurt / max(n_drop, 1)
        drop_precision = float((y[drop] == 0).mean())
        f_recall = int((y[drop] == 0).sum()) / max(n_f, 1)
        if t_hurt_rate > max_t_hurt_rate and t_hurt > max_t_hurt_abs:
            continue
        if drop_precision < min_drop_precision:
            continue
        options.append({
            "drop_threshold": float(threshold),
            "drop_n": n_drop,
            "drop_precision": round(drop_precision, 4),
            "t_hurt": t_hurt,
            "t_hurt_rate": round(t_hurt_rate, 4),
            "f_caught": int((y[drop] == 0).sum()),
            "f_recall": round(f_recall, 4),
        })
    if options:
        chosen = max(options, key=lambda x: (x["f_caught"], x["drop_precision"], -x["drop_threshold"]))
        chosen["method"] = "target_met"
        return chosen
    # best effort: minimize t_hurt while maximizing drop precision
    fallback: list[dict] = []
    for threshold in np.unique(p):
        drop = p <= threshold
        n_drop = int(drop.sum())
        if n_drop < max(1, min(min_drop_labels, len(y))):
            continue
        t_hurt = int((y[drop] == 1).sum())
        drop_precision = float((y[drop] == 0).mean())
        fallback.append({
            "drop_threshold": float(threshold),
            "drop_n": n_drop,
            "drop_precision": round(drop_precision, 4),
            "t_hurt": t_hurt,
            "t_hurt_rate": round(t_hurt / max(n_drop, 1), 4),
            "f_caught": int((y[drop] == 0).sum()),
            "f_recall": round(int((y[drop] == 0).sum()) / max(n_f, 1), 4),
        })
    if not fallback:
        raise RuntimeError("无法选择 drop 阈值")
    chosen = min(
        fallback,
        key=lambda x: (x["t_hurt"], -x["drop_precision"], -x["f_caught"]),
    )
    chosen["method"] = "best_effort"
    return chosen


def eval_on_mask(
    y: np.ndarray,
    oof: np.ndarray,
    *,
    drop_threshold: float,
) -> dict:
    drop = oof <= drop_threshold
    n_drop = int(drop.sum())
    t_hurt = int((y[drop] == 1).sum()) if n_drop else 0
    n_f = int((y == 0).sum())
    return {
        "n": int(len(y)),
        "n_t": int((y == 1).sum()),
        "n_f": n_f,
        "drop_threshold": drop_threshold,
        "drop_n": n_drop,
        "drop_precision": round(float((y[drop] == 0).mean()), 4) if n_drop else None,
        "t_hurt": t_hurt,
        "t_hurt_rate": round(t_hurt / max((y == 1).sum(), 1), 4),
        "f_caught": int((y[drop] == 0).sum()) if n_drop else 0,
        "f_recall": round(int((y[drop] == 0).sum()) / max(n_f, 1), 4) if n_f else None,
    }


def calibrate(
    *,
    labels_csv: Path,
    extra_labels_csv: Path | None,
    store_dir: Path,
    out_dir: Path,
    model_path: Path,
    calibration_path: Path,
) -> dict:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    primary = load_label_frame(labels_csv, tag="primary")
    extra = (
        load_label_frame(extra_labels_csv, tag="human678")
        if extra_labels_csv and extra_labels_csv.is_file()
        else pd.DataFrame()
    )
    primary_ids = set(primary["video_id"])
    if not extra.empty:
        extra = extra[~extra["video_id"].isin(primary_ids)].copy()

    combined = pd.concat([primary, extra], ignore_index=True)
    combined = combined.drop_duplicates("video_id", keep="first").reset_index(drop=True)
    print(
        f"[calibrate] primary={len(primary)} extra={len(extra)} "
        f"combined={len(combined)} T={int((combined.y==1).sum())} F={int((combined.y==0).sum())}",
        flush=True,
    )

    vecs, found = load_embedding_rows(store_dir, combined["video_id"].tolist())
    if len(found) < 30:
        raise RuntimeError(f"store 命中过少: {len(found)}")
    train = combined[combined["video_id"].isin(found)].copy().reset_index(drop=True)
    X = np.asarray(vecs, dtype=np.float32)
    y = train["y"].to_numpy(dtype=int)
    groups = train["group_id"].astype(str).to_numpy()
    hours = pd.to_numeric(train["duration_seconds"], errors="coerce").fillna(0).to_numpy(dtype=float) / 3600.0

    model, oof = train_grouped_visual_model(X, y, groups, n_splits=5, seed=42)

    from sklearn.metrics import average_precision_score, roc_auc_score

    auc = float(roc_auc_score(y, oof))
    ap = float(average_precision_score(y, oof))

    primary_mask = train["video_id"].isin(primary_ids).to_numpy()
    threshold_pick = pick_drop_threshold(y[primary_mask], oof[primary_mask])
    drop_thr = threshold_pick["drop_threshold"]
    actions = assign_actions(oof, keep_threshold=1.0, drop_threshold=drop_thr)

    train["clip_score"] = oof
    train["ml_action"] = actions
    train.to_csv(out_dir / "train_labels.csv", index=False)
    np.save(out_dir / "train_embeddings.npy", X)
    pd.DataFrame({
        "video_id": train["video_id"],
        "y": y,
        "clip_score": oof,
        "ml_action": actions,
        "group_id": groups,
        "hours": hours,
        "label": train["label"],
        "label_source": train["label_source"],
    }).to_csv(out_dir / "oof_scores.csv", index=False)

    primary_eval = eval_on_mask(y[primary_mask], oof[primary_mask], drop_threshold=drop_thr)
    combined_eval = eval_on_mask(y, oof, drop_threshold=drop_thr)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(
            {
                "pipeline": model,
                "encoder": "ViT-B-32/openai",
                "feature": "embedding_store",
                "embedding_store": str(store_dir.resolve()),
                "positive_class": "T",
                "labels_csv": str(labels_csv.resolve()),
                "extra_labels_csv": str(extra_labels_csv.resolve()) if extra_labels_csv else None,
                "drop_threshold": drop_thr,
            },
            fh,
        )

    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "labels_csv": str(labels_csv.resolve()),
        "extra_labels_csv": str(extra_labels_csv.resolve()) if extra_labels_csv else None,
        "embedding_store": str(store_dir.resolve()),
        "n_primary": int(len(primary)),
        "n_extra": int(len(extra)),
        "n_train": int(len(train)),
        "n_t": int((y == 1).sum()),
        "n_f": int((y == 0).sum()),
        "n_groups": int(len(np.unique(groups))),
        "oof_auc": round(auc, 4),
        "oof_ap": round(ap, 4),
        "drop_threshold": drop_thr,
        "threshold_pick": threshold_pick,
        "eval_primary_1be": primary_eval,
        "eval_combined": combined_eval,
        "model_path": str(model_path.resolve()),
        "calibration_path": str(calibration_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "notes": [
            "冻结 CLIP embedding store + balanced LR",
            "闸门=高置信 drop only（p<=drop_threshold）",
            "阈值在 primary(1be) OOF 上选：T误伤≤2%且drop precision≥0.9",
        ],
    }
    calibration_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "train_meta.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="exo_agriculture thumb CLIP+LR calibrate")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--extra-labels", type=Path, default=DEFAULT_EXTRA)
    ap.add_argument("--no-extra", action="store_true", help="不合并 human678")
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--calibration", type=Path, default=CALIB_PATH)
    args = ap.parse_args()
    if not args.calibrate:
        ap.print_help()
        sys.exit(2)
    extra = None if args.no_extra else args.extra_labels
    calibrate(
        labels_csv=args.labels,
        extra_labels_csv=extra,
        store_dir=args.store,
        out_dir=args.out_dir,
        model_path=args.model,
        calibration_path=args.calibration,
    )


if __name__ == "__main__":
    raise SystemExit(main())

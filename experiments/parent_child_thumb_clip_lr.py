#!/usr/bin/env python3
"""parent_child 缩略图 CLIP embedding + LogisticRegression（channel group split）。

第一版：人工 T/F → open_clip ViT-B-32 冻结 embedding → balanced LR。
评估看 OOF Top 1/5/10/20/30% precision/hours/recall，不做 prompt prototype。

用法:
  PYTHONPATH=02_脚本:experiments python experiments/parent_child_thumb_clip_lr.py --calibrate \\
    --labels /home/jqs/tmp/亲子互动人工_c31ff005_qc_result.csv \\
    -o data/runs/parent_child/machine_0818_lt50/06_tools/clip_lr_v1/
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

from core.exemplar_sim import ClipEncoder, fetch_thumbnails_batch  # noqa: E402
from core.visual_filter import train_grouped_visual_model  # noqa: E402

DEFAULT_LABELS = Path("/home/jqs/tmp/亲子互动人工_c31ff005_qc_result.csv")
DEFAULT_OUT = (
    PROJECT
    / "data/runs/parent_child/machine_0818_lt50/06_tools/clip_lr_v1"
)
MODEL_PATH = PROJECT / "models/parent_child_thumb_clip_lr.pkl"
CALIB_PATH = PROJECT / "models/parent_child_thumb_clip_lr_calibration.json"
TOP_PCTS = (1, 5, 10, 20, 30)


def normalize_label(value: object) -> str | None:
    v = str(value or "").strip().upper()
    if not v or v == "NAN":
        return None
    if v in {"T", "PASS"}:
        return "T"
    if v.startswith("F"):
        # "F | 无法播放" → 剔除（坏样本）
        if "无法播放" in str(value) or "UNPLAYABLE" in v:
            return None
        return "F"
    return None


def group_id(row: pd.Series) -> str:
    for col in ("channel", "source_ref", "video_id"):
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            return str(val).strip()
    return str(row.get("video_id", ""))


def top_pct_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    hours: np.ndarray,
    *,
    pcts: tuple[int, ...] = TOP_PCTS,
) -> dict:
    """按 score 降序切 Top k%：precision / hours / recall。"""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    hours = np.asarray(hours, dtype=float)
    order = np.argsort(-scores)
    y = labels[order]
    h = hours[order]
    n = len(y)
    n_pos = int((labels == 1).sum())
    total_h = float(hours.sum())
    out: dict[str, dict] = {}
    for pct in pcts:
        k = max(1, int(round(n * pct / 100.0)))
        top_y = y[:k]
        top_h = h[:k]
        tp = int((top_y == 1).sum())
        out[f"top_{pct}pct"] = {
            "n": k,
            "precision": round(tp / max(k, 1), 4),
            "hours": round(float(top_h.sum()), 2),
            "hours_share": round(float(top_h.sum()) / max(total_h, 1e-9), 4),
            "recall": round(tp / max(n_pos, 1), 4),
            "n_pos_in_top": tp,
        }
    out["n_total"] = n
    out["n_pos"] = n_pos
    out["total_hours"] = round(total_h, 2)
    return out


def calibrate(
    *,
    labels_csv: Path,
    out_dir: Path,
    model_path: Path,
    calibration_path: Path,
    cache_dir: Path,
    thumb_workers: int = 16,
    batch_size: int = 64,
) -> dict:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(labels_csv, dtype=str, low_memory=False)
    if "video_id" not in raw.columns or "qc_result" not in raw.columns:
        raise ValueError("人工表需含 video_id / qc_result")

    rows: list[dict] = []
    skipped: list[dict] = []
    for _, r in raw.iterrows():
        lab = normalize_label(r.get("qc_result"))
        vid = str(r.get("video_id", "") or "").strip()
        if not vid:
            skipped.append({"reason": "no_video_id", "qc_result": r.get("qc_result")})
            continue
        if lab is None:
            skipped.append({"video_id": vid, "reason": "bad_or_unplayable", "qc_result": r.get("qc_result")})
            continue
        rows.append({
            "video_id": vid,
            "title": r.get("title", ""),
            "channel": r.get("channel", ""),
            "source_ref": r.get("source_ref", ""),
            "duration_seconds": r.get("duration_seconds", ""),
            "label": lab,
            "y": 1 if lab == "T" else 0,
            "fail_type": "",
            "group_id": group_id(r),
        })

    frame = pd.DataFrame(rows).drop_duplicates("video_id", keep="first").reset_index(drop=True)
    print(f"[calibrate] labeled usable: {len(frame)}  T={int((frame.y==1).sum())} F={int((frame.y==0).sum())}")

    video_ids = frame["video_id"].tolist()
    paths = fetch_thumbnails_batch(video_ids, cache_dir, workers=thumb_workers)
    ok_mask = [p is not None and Path(p).is_file() and Path(p).stat().st_size >= 1500 for p in paths]
    thumb_paths = [str(p) if ok else "" for p, ok in zip(paths, ok_mask)]
    frame["thumb_path"] = thumb_paths

    train = frame[frame["thumb_path"] != ""].copy().reset_index(drop=True)
    n_fail_thumb = int((frame["thumb_path"] == "").sum())
    print(f"[calibrate] thumbs ok: {len(train)}  fail: {n_fail_thumb}")

    if len(train) < 30:
        raise RuntimeError(f"可用缩略图过少: {len(train)}")

    encoder = ClipEncoder()
    emb_list: list[np.ndarray] = []
    ok_paths = [Path(p) for p in train["thumb_path"].tolist()]
    for start in range(0, len(ok_paths), batch_size):
        chunk = ok_paths[start : start + batch_size]
        emb_list.append(encoder.encode_paths(chunk, batch_size=batch_size))
        print(f"  encode {min(start + len(chunk), len(ok_paths))}/{len(ok_paths)}", flush=True)
    X = np.concatenate(emb_list, axis=0).astype(np.float32)
    y = train["y"].to_numpy(dtype=int)
    groups = train["group_id"].astype(str).to_numpy()
    hours = pd.to_numeric(train["duration_seconds"], errors="coerce").fillna(0).to_numpy(dtype=float) / 3600.0

    model, oof = train_grouped_visual_model(X, y, groups, n_splits=5, seed=42)
    metrics = top_pct_metrics(y, oof, hours)

    from sklearn.metrics import average_precision_score, roc_auc_score

    auc = float(roc_auc_score(y, oof))
    ap = float(average_precision_score(y, oof))

    train_labels_path = out_dir / "train_labels.csv"
    emb_path = out_dir / "train_embeddings.npy"
    oof_path = out_dir / "oof_scores.csv"
    metrics_path = out_dir / "top_pct_metrics.json"
    meta_path = out_dir / "train_meta.json"

    train.to_csv(train_labels_path, index=False)
    np.save(emb_path, X)
    pd.DataFrame({
        "video_id": train["video_id"],
        "y": y,
        "clip_score": oof,
        "group_id": groups,
        "hours": hours,
        "label": train["label"],
    }).to_csv(oof_path, index=False)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(
            {
                "pipeline": model,
                "encoder": "ViT-B-32/openai",
                "feature": "open_clip_image_embedding",
                "positive_class": "T",
                "labels_csv": str(labels_csv.resolve()),
            },
            fh,
        )

    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "labels_csv": str(labels_csv.resolve()),
        "n_raw": int(len(raw)),
        "n_train": int(len(train)),
        "n_t": int((y == 1).sum()),
        "n_f": int((y == 0).sum()),
        "n_groups": int(len(np.unique(groups))),
        "n_thumb_fail": n_fail_thumb,
        "n_skipped": len(skipped),
        "oof_auc": round(auc, 4),
        "oof_ap": round(ap, 4),
        "top_pct": metrics,
        "model_path": str(model_path.resolve()),
        "calibration_path": str(calibration_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "notes": [
            "冻结 open_clip embedding + balanced LR",
            "StratifiedGroupKFold by channel/source_ref/video_id",
            "KPI=Top% precision/hours/recall；ml/clip_score 非交付 KPI",
        ],
        "skipped_sample": skipped[:20],
    }
    calibration_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="parent_child thumb CLIP + LR calibrate")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--calibration", type=Path, default=CALIB_PATH)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="缩略图缓存；默认 out-dir/thumb_cache",
    )
    ap.add_argument("--thumb-workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    if not args.calibrate:
        ap.print_help()
        sys.exit(2)
    cache = args.cache_dir or (args.out_dir / "thumb_cache")
    calibrate(
        labels_csv=args.labels,
        out_dir=args.out_dir,
        model_path=args.model,
        calibration_path=args.calibration,
        cache_dir=cache,
        thumb_workers=args.thumb_workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())

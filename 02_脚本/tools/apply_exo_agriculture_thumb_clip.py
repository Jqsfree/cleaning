#!/usr/bin/env python3
"""对 exo_agriculture keep 池用 CLIP embedding store + LR 探针打分并过滤。

模式：
- 默认：只丢高置信负例（p <= drop_threshold）
- --keep-threshold / --target-keep-pass-rate：keep 需 score >= keep_threshold（面向 keep 合格率 KPI）

用法:
  PYTHONPATH=02_脚本 python 02_脚本/tools/apply_exo_agriculture_thumb_clip.py \\
    data/runs/exo_agriculture/machine_0814/06_tools/v11_pass_blacklist_v04/农业采集_0814_clip_pass_v11_clean_0821.csv \\
    -o data/runs/exo_agriculture/machine_0814/06_tools/v04_thumb_lr_v2/ \\
    --from-scored data/runs/exo_agriculture/machine_0814/06_tools/v04_thumb_lr_v1/pass_scored.csv \\
    --target-keep-pass-rate 0.70 \\
    --eval-labels data/runs/exo_agriculture/machine_0814/03_qc/human270_v04_1be48617/03_qc/labeled.csv
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

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from core.visual_filter import assign_actions, load_embedding_rows  # noqa: E402

DEFAULT_MODEL = _REPO_ROOT / "models/exo_agriculture_thumb_clip_lr.pkl"
DEFAULT_STORE = _REPO_ROOT / "data/assets/embeddings/exo_agriculture_0814_semantic_remain"
DEFAULT_CALIB = _REPO_ROOT / "models/exo_agriculture_thumb_clip_lr_calibration.json"


def load_model(model_path: Path) -> tuple[object, float]:
    with model_path.open("rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict):
        pipe = obj.get("pipeline", obj)
        thr = float(obj.get("drop_threshold", 0.0))
        return pipe, thr
    return obj, 0.0


def load_drop_threshold(model_path: Path, calibration_path: Path) -> float:
    _, thr = load_model(model_path)
    if thr > 0:
        return thr
    if calibration_path.is_file():
        data = json.loads(calibration_path.read_text(encoding="utf-8"))
        return float(data.get("drop_threshold", 0.0))
    return 0.0


def load_labels_y(labels_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path, dtype=str, low_memory=False)
    if "human_label" in labels.columns:
        lab = labels["human_label"].str.lower().map({"pass": 1, "fail": 0})
    else:
        qc = labels["qc_result"].astype(str).str.strip().str.upper()
        lab = qc.map(lambda x: 1 if x == "T" else (0 if x.startswith("F") else np.nan))
    return labels.assign(y=lab).dropna(subset=["y"])


def pick_keep_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    target_pass_rate: float = 0.70,
    min_keep_labels: int = 20,
) -> dict:
    """在人工标上选 keep 阈值：score>=thr 时合格率≥target，且保留量最大。"""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    valid = np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) < min_keep_labels:
        raise ValueError("标量过少，无法选 keep 阈值")

    best: dict | None = None
    for threshold in np.unique(p):
        keep = p >= threshold
        n_keep = int(keep.sum())
        if n_keep < min_keep_labels:
            continue
        rate = float(y[keep].mean())
        if rate + 1e-9 < target_pass_rate:
            continue
        cand = {
            "keep_threshold": float(threshold),
            "keep_n": n_keep,
            "keep_pass_rate": round(rate, 4),
            "n_t_kept": int((y[keep] == 1).sum()),
            "n_f_kept": int((y[keep] == 0).sum()),
            "n_t_total": int((y == 1).sum()),
            "n_f_total": int((y == 0).sum()),
            "method": "target_met",
        }
        if best is None or n_keep > best["keep_n"] or (
            n_keep == best["keep_n"] and rate > best["keep_pass_rate"]
        ):
            best = cand
    if best is not None:
        return best

    # 未达标：取合格率最高的阈值（至少 min_keep_labels）
    fallback: dict | None = None
    for threshold in np.unique(p):
        keep = p >= threshold
        n_keep = int(keep.sum())
        if n_keep < max(5, min(min_keep_labels, len(y))):
            continue
        rate = float(y[keep].mean())
        cand = {
            "keep_threshold": float(threshold),
            "keep_n": n_keep,
            "keep_pass_rate": round(rate, 4),
            "n_t_kept": int((y[keep] == 1).sum()),
            "n_f_kept": int((y[keep] == 0).sum()),
            "n_t_total": int((y == 1).sum()),
            "n_f_total": int((y == 0).sum()),
            "method": "best_effort",
        }
        if fallback is None or rate > fallback["keep_pass_rate"] or (
            rate == fallback["keep_pass_rate"] and n_keep > fallback["keep_n"]
        ):
            fallback = cand
    if fallback is None:
        raise RuntimeError("无法选择 keep 阈值")
    return fallback


def keep_mask_from_thresholds(
    scores: pd.Series,
    *,
    keep_threshold: float | None,
    drop_threshold: float,
) -> pd.Series:
    s = pd.to_numeric(scores, errors="coerce")
    if keep_threshold is not None:
        return s.isna() | (s < keep_threshold)
    return s <= drop_threshold


def eval_keep_on_labels(
    merged: pd.DataFrame,
    *,
    drop_mask: np.ndarray,
    keep_threshold: float | None,
    drop_threshold: float,
) -> dict:
    y = merged["y"].astype(int).to_numpy()
    n_f = int((y == 0).sum())
    n_t = int((y == 1).sum())
    keep = ~drop_mask
    return {
        "n": int(len(merged)),
        "keep_threshold": keep_threshold,
        "drop_threshold": drop_threshold if keep_threshold is None else None,
        "keep_n": int(keep.sum()),
        "keep_pass_rate": round(float(y[keep].mean()), 4) if keep.any() else None,
        "n_t_kept": int((y[keep] == 1).sum()) if keep.any() else 0,
        "n_f_kept": int((y[keep] == 0).sum()) if keep.any() else 0,
        "t_hurt_rate": round(int((y[drop_mask] == 1).sum()) / max(n_t, 1), 4) if drop_mask.any() else 0.0,
        "f_recall": round(int((y[drop_mask] == 0).sum()) / max(n_f, 1), 4) if n_f else None,
        "drop_n": int(drop_mask.sum()),
    }


def hours(frame: pd.DataFrame) -> float:
    if "duration_seconds" not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame["duration_seconds"], errors="coerce").fillna(0).sum() / 3600.0)


def score_batch(
    video_ids: list[str],
    *,
    store_dir: Path,
    pipe,
) -> tuple[np.ndarray, np.ndarray]:
    vecs, found = load_embedding_rows(store_dir, video_ids)
    scores = np.full(len(video_ids), np.nan, dtype=np.float64)
    status = np.full(len(video_ids), "no_embedding", dtype=object)
    if not found:
        return scores, status
    vid_to_local = {vid: i for i, vid in enumerate(found)}
    rows = [vid_to_local[vid] for vid in video_ids if vid in vid_to_local]
    idxs = [i for i, vid in enumerate(video_ids) if vid in vid_to_local]
    if rows:
        X = np.asarray(vecs, dtype=np.float32)
        proba = pipe.predict_proba(X)[:, 1]
        for j, ii in enumerate(idxs):
            scores[ii] = float(proba[j])
            status[ii] = "ok"
    return scores, status


def apply_filter(
    input_csv: Path,
    *,
    out_dir: Path,
    model_path: Path,
    store_dir: Path,
    calibration_path: Path,
    chunksize: int = 50_000,
    eval_labels: Path | None = None,
    keep_threshold: float | None = None,
    target_keep_pass_rate: float | None = None,
    from_scored: Path | None = None,
) -> dict:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe, _ = load_model(model_path)
    drop_thr = load_drop_threshold(model_path, calibration_path)
    threshold_pick: dict | None = None

    if target_keep_pass_rate is not None:
        if not eval_labels or not eval_labels.is_file():
            raise ValueError("--target-keep-pass-rate 需要 --eval-labels")
        scored_for_cal = from_scored
        if scored_for_cal is None:
            oof = Path(calibration_path).parent.parent / (
                "data/runs/exo_agriculture/machine_0814/06_tools/clip_lr_v1/oof_scores.csv"
            )
            if not oof.is_file():
                oof = _REPO_ROOT / "data/runs/exo_agriculture/machine_0814/06_tools/clip_lr_v1/oof_scores.csv"
            scored_for_cal = oof if oof.is_file() else None
        if scored_for_cal and scored_for_cal.is_file():
            labels = load_labels_y(eval_labels)
            primary_ids = set(labels["video_id"].astype(str))
            sc = pd.read_csv(scored_for_cal, dtype=str, low_memory=False)
            score_col = "thumb_lr_score" if "thumb_lr_score" in sc.columns else "clip_score"
            sc[score_col] = pd.to_numeric(sc[score_col], errors="coerce")
            sc = sc[sc["video_id"].astype(str).isin(primary_ids)]
            merged_cal = labels.merge(
                sc[["video_id", score_col]].rename(columns={score_col: "thumb_lr_score"}),
                on="video_id",
            )
            threshold_pick = pick_keep_threshold(
                merged_cal["y"].astype(int).to_numpy(),
                merged_cal["thumb_lr_score"].to_numpy(),
                target_pass_rate=target_keep_pass_rate,
            )
            keep_threshold = threshold_pick["keep_threshold"]
        elif keep_threshold is None:
            raise ValueError("无法从 OOF/scored 选阈；请传 --from-scored 或 --keep-threshold")

    pass_path = out_dir / f"{input_csv.stem}_thumb_lr_pass.csv"
    drop_path = out_dir / f"{input_csv.stem}_thumb_lr_drop.csv"
    scored_path = out_dir / "pass_scored.csv"

    n_in = 0
    n_pass = 0
    n_drop = 0
    n_no_emb = 0
    write_pass = True
    write_drop = True
    write_scored = True
    if from_scored:
        for p in (pass_path, drop_path, scored_path):
            if p.exists():
                p.unlink()

    def process_chunk(chunk: pd.DataFrame, *, rescore: bool) -> pd.DataFrame:
        nonlocal n_no_emb
        chunk["video_id"] = chunk["video_id"].astype(str).str.strip()
        if rescore:
            scores, status = score_batch(chunk["video_id"].tolist(), store_dir=store_dir, pipe=pipe)
            chunk = chunk.copy()
            chunk["thumb_lr_score"] = scores
            chunk["thumb_lr_status"] = status
        else:
            chunk["thumb_lr_score"] = pd.to_numeric(chunk["thumb_lr_score"], errors="coerce")
            if "thumb_lr_status" not in chunk.columns:
                chunk["thumb_lr_status"] = np.where(
                    chunk["thumb_lr_score"].notna(), "ok", "no_embedding",
                )
        if keep_threshold is not None:
            drop_mask = keep_mask_from_thresholds(
                chunk["thumb_lr_score"], keep_threshold=keep_threshold, drop_threshold=drop_thr,
            )
            chunk["thumb_lr_action"] = np.where(drop_mask, "highconf_drop", "keep_candidate")
        else:
            chunk["thumb_lr_action"] = assign_actions(
                chunk["thumb_lr_score"], keep_threshold=1.0, drop_threshold=drop_thr,
            )
            drop_mask = chunk["thumb_lr_action"].eq("highconf_drop")
        n_no_emb += int((chunk["thumb_lr_status"] != "ok").sum())
        return chunk, drop_mask

    if from_scored and from_scored.is_file():
        for chunk in pd.read_csv(from_scored, chunksize=chunksize, low_memory=False):
            n_in += len(chunk)
            chunk, drop_mask = process_chunk(chunk, rescore=False)
            pass_mask = ~drop_mask
            n_drop += int(drop_mask.sum())
            n_pass += int(pass_mask.sum())
            chunk.to_csv(scored_path, mode="w" if write_scored else "a", header=write_scored, index=False)
            write_scored = False
            chunk.loc[pass_mask].to_csv(pass_path, mode="w" if write_pass else "a", header=write_pass, index=False)
            write_pass = False
            chunk.loc[drop_mask].to_csv(drop_path, mode="w" if write_drop else "a", header=write_drop, index=False)
            write_drop = False
            print(f"  chunk rows={n_in:,} pass={n_pass:,} drop={n_drop:,}", flush=True)
    else:
        for chunk in pd.read_csv(input_csv, chunksize=chunksize, low_memory=False):
            n_in += len(chunk)
            chunk, drop_mask = process_chunk(chunk, rescore=True)
            pass_mask = ~drop_mask
            n_drop += int(drop_mask.sum())
            n_pass += int(pass_mask.sum())
            chunk.to_csv(scored_path, mode="w" if write_scored else "a", header=write_scored, index=False)
            write_scored = False
            chunk.loc[pass_mask].to_csv(pass_path, mode="w" if write_pass else "a", header=write_pass, index=False)
            write_pass = False
            chunk.loc[drop_mask].to_csv(drop_path, mode="w" if write_drop else "a", header=write_drop, index=False)
            write_drop = False
            print(f"  chunk rows={n_in:,} pass={n_pass:,} drop={n_drop:,}", flush=True)

    eval_primary = None
    if eval_labels and eval_labels.is_file():
        labels = load_labels_y(eval_labels)
        scored = pd.read_csv(scored_path, usecols=["video_id", "thumb_lr_score", "thumb_lr_action"], low_memory=False)
        merged = labels.merge(scored, on="video_id", how="inner")
        drop = merged["thumb_lr_action"].eq("highconf_drop").to_numpy()
        eval_primary = eval_keep_on_labels(
            merged,
            drop_mask=drop,
            keep_threshold=keep_threshold,
            drop_threshold=drop_thr,
        )

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(input_csv.resolve()),
        "from_scored": str(from_scored.resolve()) if from_scored else None,
        "out_dir": str(out_dir.resolve()),
        "model": str(model_path.resolve()),
        "store": str(store_dir.resolve()),
        "mode": "keep_threshold" if keep_threshold is not None else "drop_only",
        "keep_threshold": keep_threshold,
        "target_keep_pass_rate": target_keep_pass_rate,
        "threshold_pick": threshold_pick,
        "drop_threshold": drop_thr if keep_threshold is None else None,
        "n_in": n_in,
        "n_pass": n_pass,
        "n_drop": n_drop,
        "n_no_embedding_kept": n_no_emb,
        "hours_in": None,
        "pass_path": str(pass_path.resolve()),
        "drop_path": str(drop_path.resolve()),
        "scored_path": str(scored_path.resolve()),
        "eval_primary": eval_primary,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    # recompute hours from outputs (streaming-friendly)
    if pass_path.exists():
        summary["hours_pass"] = round(
            hours(pd.read_csv(pass_path, usecols=["duration_seconds"], low_memory=False)), 2,
        )
    if drop_path.exists():
        summary["hours_drop"] = round(
            hours(pd.read_csv(drop_path, usecols=["duration_seconds"], low_memory=False)), 2,
        )
    if pass_path.exists() and drop_path.exists():
        summary["hours_in"] = round(
            (summary.get("hours_pass") or 0) + (summary.get("hours_drop") or 0), 2,
        )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="exo_agriculture thumb LR apply/filter")
    ap.add_argument("input", type=Path, help="keep CSV")
    ap.add_argument("-o", "--out-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--chunksize", type=int, default=50_000)
    ap.add_argument("--eval-labels", type=Path, default=None)
    ap.add_argument("--keep-threshold", type=float, default=None, help="keep 需 score>=该值")
    ap.add_argument(
        "--target-keep-pass-rate", type=float, default=None,
        help="在 eval-labels 上选 keep 阈值使合格率达标（如 0.70）",
    )
    ap.add_argument(
        "--from-scored", type=Path, default=None,
        help="已有 pass_scored.csv，跳过重编码",
    )
    args = ap.parse_args()
    if not args.model.exists():
        print(f"[ERROR] 缺少模型: {args.model}")
        return 2
    apply_filter(
        args.input,
        out_dir=args.out_dir,
        model_path=args.model,
        store_dir=args.store,
        calibration_path=args.calibration,
        chunksize=args.chunksize,
        eval_labels=args.eval_labels,
        keep_threshold=args.keep_threshold,
        target_keep_pass_rate=args.target_keep_pass_rate,
        from_scored=args.from_scored,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

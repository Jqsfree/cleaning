#!/usr/bin/env python3
"""缩略图封面文字过多 — 本地 CV 特征 + LR certain-noise 过滤。

用法:
  # 用人工 270 标定并落盘模型
  02_脚本/tools/score_thumb_text_heavy.py --calibrate /home/jqs/tmp/exo医疗_614991c7_qc_result.csv \\
    --save-model models/exo_medical_cover_text_lr.pkl \\
    --calibration-json models/exo_medical_cover_text_calibration.json

  # 对 MiniLM keep 全量打分
  02_脚本/tools/score_thumb_text_heavy.py \\
    data/runs/exo_medical/machine_0813/06_tools/text_semantic/*_thumb_T_keep.csv \\
    -o data/runs/exo_medical/machine_0813/06_tools/cover_text/ \\
    --model models/exo_medical_cover_text_lr.pkl \\
    --drop-threshold 0.85
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from core.thumb_text_heavy import (  # noqa: E402
    FEATURE_NAMES,
    composite_text_score,
    extract_features_from_bytes,
    extract_features_from_path,
    features_to_vector,
)
from qc.vision_thumb import THUMB_SUFFIXES, download_thumbnail  # noqa: E402

DEFAULT_CALIBRATION = _REPO_ROOT / "models/exo_medical_cover_text_calibration.json"
DEFAULT_MODEL = _REPO_ROOT / "models/exo_medical_cover_text_lr.pkl"
HUMAN_LABELS_DEFAULT = Path("/home/jqs/tmp/exo医疗_614991c7_qc_result.csv")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def resolve_label_column(df: pd.DataFrame) -> str:
    for col in ("qc_result", "human_label", "label"):
        if col not in df.columns:
            continue
        vals = df[col].dropna().astype(str).str.strip().str.upper()
        if set(vals.unique()) & {"T", "F", "PASS", "FAIL"}:
            return col
    raise ValueError("标定 CSV 需含 qc_result / human_label（T/F）")


def normalize_label(value: str) -> str:
    v = str(value).strip().upper()
    if v in {"T", "PASS"}:
        return "T"
    if v in {"F", "FAIL"}:
        return "F"
    raise ValueError(f"未知标签: {value!r}")


def cache_path(cache_dir: Path, video_id: str, suffix: str) -> Path:
    return cache_dir / f"{video_id}_{suffix}.jpg"


def ensure_thumbnail(
    video_id: str,
    *,
    cache_dir: Path,
    download: bool = True,
) -> tuple[Path | None, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for suffix in THUMB_SUFFIXES:
        path = cache_path(cache_dir, video_id, suffix)
        if path.exists() and path.stat().st_size >= 1500:
            return path, ""
    if not download:
        return None, "missing_thumbnail"
    data, suffix = download_thumbnail(video_id, str(cache_dir))
    if data is None or suffix is None:
        return None, "download_failed"
    path = cache_path(cache_dir, video_id, suffix)
    if path.exists():
        return path, ""
    return None, "download_failed"


def score_image_path(path: Path) -> tuple[dict[str, float] | None, str]:
    feats = extract_features_from_path(path)
    if feats is None:
        return None, "decode_error"
    return feats, ""


def score_video_id(
    video_id: str,
    *,
    cache_dir: Path,
    download: bool,
) -> tuple[dict[str, float] | None, str]:
    path, err = ensure_thumbnail(video_id, cache_dir=cache_dir, download=download)
    if path is not None:
        return score_image_path(path)
    if not download:
        return None, err
    data, _suffix = download_thumbnail(video_id, str(cache_dir))
    if data is None:
        return None, err or "download_failed"
    feats = extract_features_from_bytes(data)
    if feats is None:
        return None, "decode_error"
    return feats, ""


def build_pipeline(*, C: float = 0.5) -> Pipeline:
    return Pipeline([
        ("sc", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=C,
                random_state=42,
            ),
        ),
    ])


def pick_drop_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    min_precision: float = 0.85,
    max_t_kill_rate: float = 0.03,
) -> dict:
    """OOF 概率上选 certain-noise 阈值（p_F 越高越像文字封面 F）。"""
    n_t = int((y_true == 0).sum())
    n_f = int((y_true == 1).sum())
    candidates = []
    for thr in np.unique(np.round(proba, 4)):
        pred = proba >= thr
        tp = int(((pred) & (y_true == 1)).sum())
        fp = int(((pred) & (y_true == 0)).sum())
        fn = int(((~pred) & (y_true == 1)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        t_kill = fp / max(n_t, 1)
        candidates.append({
            "drop_threshold": float(thr),
            "n_drop": int(pred.sum()),
            "catch_f": tp,
            "n_f": n_f,
            "kill_t": fp,
            "n_t": n_t,
            "precision": float(prec),
            "recall_f": float(rec),
            "t_kill_rate": float(t_kill),
        })

    strict = [
        c for c in candidates
        if c["precision"] >= min_precision and c["t_kill_rate"] <= max_t_kill_rate
    ]
    pool = strict if strict else candidates
    best = max(pool, key=lambda c: (c["precision"], c["catch_f"], -c["kill_t"]))
    return best


def calibrate_from_labels(
    labels_csv: Path,
    *,
    cache_dir: Path,
    save_model: Path,
    calibration_json: Path,
    thumb_workers: int = 16,
) -> dict:
    df = read_table(labels_csv)
    label_col = resolve_label_column(df)
    labels = []
    feat_rows = []
    errors = []

    video_ids = df["video_id"].astype(str).tolist()

    def _one(vid: str) -> tuple[str, dict[str, float] | None, str]:
        feats, err = score_video_id(vid, cache_dir=cache_dir, download=True)
        return vid, feats, err

    results: dict[str, tuple[dict[str, float] | None, str]] = {}
    with ThreadPoolExecutor(max_workers=thumb_workers) as ex:
        futs = {ex.submit(_one, vid): vid for vid in video_ids}
        for fut in as_completed(futs):
            vid, feats, err = fut.result()
            results[vid] = (feats, err)

    for row in df.itertuples(index=False):
        vid = str(row.video_id)
        feats, err = results.get(vid, (None, "missing"))
        if feats is None:
            errors.append({"video_id": vid, "error": err})
            continue
        try:
            lab = normalize_label(getattr(row, label_col))
        except ValueError:
            continue
        feat_rows.append(feats)
        labels.append(1 if lab == "F" else 0)

    if len(feat_rows) < 20:
        raise RuntimeError(f"可用标定样本过少: {len(feat_rows)}")

    X = np.vstack([features_to_vector(f) for f in feat_rows])
    y = np.asarray(labels, dtype=int)
    pipe = build_pipeline()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y, oof))
    picked = pick_drop_threshold(y, oof)
    alt_75 = next(
        (c for c in sorted(
            [
                {
                    "drop_threshold": float(thr),
                    "n_drop": int((oof >= thr).sum()),
                    "catch_f": int(((oof >= thr) & (y == 1)).sum()),
                    "kill_t": int(((oof >= thr) & (y == 0)).sum()),
                }
                for thr in [0.75]
            ],
            key=lambda c: c["drop_threshold"],
        )),
        None,
    )

    pipe.fit(X, y)
    save_model.parent.mkdir(parents=True, exist_ok=True)
    with open(save_model, "wb") as f:
        pickle.dump(
            {
                "pipeline": pipe,
                "feature_names": list(FEATURE_NAMES),
                "positive_class": "text_heavy_fail",
                "labels_csv": str(labels_csv.resolve()),
            },
            f,
        )

    payload = {
        "labels_csv": str(labels_csv.resolve()),
        "n_labeled": int(len(y)),
        "n_f": int(y.sum()),
        "n_t": int((y == 0).sum()),
        "oof_auc_f": round(auc, 4),
        "picked_threshold": picked,
        "alt_threshold_0.75": alt_75,
        "model_path": str(save_model.resolve()),
        "feature_names": list(FEATURE_NAMES),
        "errors": errors,
        "note": "p_text_heavy = P(人工F|封面特征); drop when >= drop_threshold",
    }
    calibration_json.parent.mkdir(parents=True, exist_ok=True)
    calibration_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def load_scorer(model_path: Path) -> Pipeline:
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, Pipeline):
        return obj
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"]
    raise TypeError(f"无法识别的模型格式: {model_path}")


def predict_p_f(pipe: Pipeline, feats: dict[str, float]) -> float:
    x = features_to_vector(feats).reshape(1, -1)
    return float(pipe.predict_proba(x)[0, 1])


def load_drop_threshold(
    explicit: float | None,
    calibration_json: Path,
) -> float:
    if explicit is not None:
        return explicit
    if calibration_json.exists():
        data = json.loads(calibration_json.read_text(encoding="utf-8"))
        return float(data["picked_threshold"]["drop_threshold"])
    return 0.85


def score_dataframe(
    df: pd.DataFrame,
    *,
    cache_dir: Path,
    pipe: Pipeline | None,
    drop_threshold: float,
    thumb_workers: int,
    download: bool,
) -> pd.DataFrame:
    video_ids = df["video_id"].astype(str).tolist()
    rows: list[dict] = []

    def _one(vid: str) -> tuple[str, dict[str, float] | None, str]:
        feats, err = score_video_id(vid, cache_dir=cache_dir, download=download)
        return vid, feats, err

    results: dict[str, tuple[dict[str, float] | None, str]] = {}
    with ThreadPoolExecutor(max_workers=thumb_workers) as ex:
        futs = {ex.submit(_one, vid): vid for vid in video_ids}
        done = 0
        for fut in as_completed(futs):
            vid, feats, err = fut.result()
            results[vid] = (feats, err)
            done += 1
            if done % 2000 == 0:
                print(f"  thumbs {done:,}/{len(video_ids):,}", flush=True)

    for idx, row in df.iterrows():
        vid = str(row["video_id"])
        feats, err = results.get(vid, (None, "missing"))
        out = row.to_dict()
        out["cover_text_error"] = err
        if feats is None:
            out["cover_text_score"] = ""
            out["cover_text_heuristic"] = ""
            out["p_text_heavy"] = ""
            out["cover_text_action"] = "keep_error"
            for name in FEATURE_NAMES:
                out[f"cover_{name}"] = ""
        else:
            out["cover_text_heuristic"] = round(composite_text_score(feats), 6)
            for name in FEATURE_NAMES:
                out[f"cover_{name}"] = feats[name]
            if pipe is not None:
                p_f = predict_p_f(pipe, feats)
                out["p_text_heavy"] = round(p_f, 6)
                action = "highconf_drop" if p_f >= drop_threshold else "keep"
                out["cover_text_action"] = action
                out["cover_text_score"] = out["p_text_heavy"]
            else:
                h = float(out["cover_text_heuristic"])
                out["p_text_heavy"] = ""
                out["cover_text_score"] = out["cover_text_heuristic"]
                out["cover_text_action"] = (
                    "highconf_drop" if h >= drop_threshold else "keep"
                )
        rows.append(out)

    return pd.DataFrame(rows)


def stem_from_input(path: Path) -> str:
    name = path.stem
    for suffix in ("_thumb_T_keep", "_keep", "_quality"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def write_outputs(
    scored: pd.DataFrame,
    *,
    output_dir: Path,
    input_path: Path,
    drop_threshold: float,
    model_path: Path | None,
    report_only: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = stem_from_input(input_path)
    drop = scored[scored["cover_text_action"] == "highconf_drop"].copy()
    keep = scored[~scored["cover_text_action"].eq("highconf_drop")].copy()
    errors = scored[scored["cover_text_action"] == "keep_error"].copy()

    summary = {
        "input": str(input_path.resolve()),
        "n_input": int(len(scored)),
        "n_drop": int(len(drop)),
        "n_keep": int(len(keep)),
        "n_keep_error": int(len(errors)),
        "drop_threshold": drop_threshold,
        "model": str(model_path.resolve()) if model_path else None,
        "report_only": report_only,
        "note": "p_text_heavy / cover_text_action not deliver KPI",
    }

    if report_only:
        for alt in (0.75, 0.80, 0.85, 0.90):
            if "p_text_heavy" not in scored.columns:
                break
            p = pd.to_numeric(scored["p_text_heavy"], errors="coerce")
            m = p >= alt
            summary[f"would_drop_p>={alt}"] = int(m.sum())
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    scored_path = output_dir / f"{stem}_cover_text_scored.csv"
    drop_path = output_dir / f"{stem}_cover_text_drop.csv"
    keep_path = output_dir / f"{stem}_cover_text_keep.csv"
    scored.to_csv(scored_path, index=False)
    drop.to_csv(drop_path, index=False)
    keep.to_csv(keep_path, index=False)

    summary.update({
        "scored_csv": str(scored_path.resolve()),
        "drop_csv": str(drop_path.resolve()),
        "keep_csv": str(keep_path.resolve()),
    })
    apply_json = output_dir / "apply.json"
    apply_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="缩略图封面文字过多 CV 打分")
    p.add_argument("input", nargs="?", help="候选 CSV/Parquet（需 video_id）")
    p.add_argument("-o", "--output-dir", help="输出目录")
    p.add_argument("--cache-dir", default="qc_thumb_cache")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument(
        "--calibration-json",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    p.add_argument("--drop-threshold", type=float, default=None)
    p.add_argument("--calibrate", type=Path, help="人工 T/F CSV，训练并落盘模型")
    p.add_argument(
        "--save-model",
        type=Path,
        default=DEFAULT_MODEL,
    )
    p.add_argument("--thumb-workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)

    if args.calibrate:
        payload = calibrate_from_labels(
            args.calibrate,
            cache_dir=cache_dir,
            save_model=args.save_model,
            calibration_json=args.calibration_json,
            thumb_workers=args.thumb_workers,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if not args.input or not args.output_dir:
        p.error("打分需 input 与 -o/--output-dir（或仅用 --calibrate）")

    t0 = time.perf_counter()
    input_path = Path(args.input)
    df = read_table(input_path)
    if "video_id" not in df.columns:
        raise SystemExit("[ERROR] 输入缺少 video_id")
    if args.limit:
        df = df.head(args.limit).copy()

    drop_threshold = load_drop_threshold(args.drop_threshold, args.calibration_json)
    pipe = None
    if args.model.exists():
        pipe = load_scorer(args.model)
    elif args.drop_threshold is None:
        print(
            f"[WARN] 模型不存在 {args.model}，仅用启发式 composite_text_score",
            flush=True,
        )

    scored = score_dataframe(
        df,
        cache_dir=cache_dir,
        pipe=pipe,
        drop_threshold=drop_threshold,
        thumb_workers=args.thumb_workers,
        download=not args.no_download,
    )
    write_outputs(
        scored,
        output_dir=Path(args.output_dir),
        input_path=input_path,
        drop_threshold=drop_threshold,
        model_path=args.model if args.model.exists() else None,
        report_only=args.report_only,
    )
    print(f"elapsed_sec={time.perf_counter() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()

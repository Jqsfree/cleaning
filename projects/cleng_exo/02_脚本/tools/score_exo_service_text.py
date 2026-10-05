#!/usr/bin/env python3
"""商业服务 MiniLM 文本打分 — title → ml_score。

score < drop_threshold → ml_auto_drop。默认阈值来自 calibration.recall。

用法:
  PYTHONPATH=02_脚本:.venv  # experiments on path via repo root
  .venv/bin/python3 02_脚本/tools/score_exo_service_text.py \\
    data/runs/exo_service/machine_0813/06_tools/cascade_v2/商业服务_quality_commercial_service_candidates_0915.csv \\
    -o data/runs/exo_service/machine_0813/06_tools/text_semantic/candidates_ml_scored_0915.csv \\
    --split-keep-drop
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import exo_service_minilm_text_classifier as svc  # noqa: E402

DEFAULT_MODEL = _REPO_ROOT / "models/exo_service_minilm_text_clf_v0.pkl"
DEFAULT_CALIB = _REPO_ROOT / "models/exo_service_minilm_text_clf_v0_calibration.json"
DEFAULT_BATCH_SIZE = 256


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_threshold(calib_path: Path, *, legacy_strict: bool = False) -> float:
    if not calib_path.is_file():
        return 0.25
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    if legacy_strict:
        strict = calib.get("strict") or {}
        return float(strict.get("drop_threshold", 0.20))
    recall = calib.get("recall") or {}
    if recall.get("drop_threshold") is not None:
        return float(recall["drop_threshold"])
    strict = calib.get("strict") or {}
    return float(strict.get("drop_threshold", 0.25))


def load_model(model_path: Path):
    svc.MiniLMEncoder.__module__ = "exo_service_minilm_text_classifier"
    sys.modules.setdefault("exo_service_minilm_text_classifier", svc)
    importlib.import_module("exo_service_minilm_text_classifier")
    with model_path.open("rb") as fh:
        return pickle.load(fh)


def _get_encoder(model):
    if isinstance(model, Pipeline) and "emb" in model.named_steps:
        return model.named_steps["emb"]
    return None


def configure_encoder(model, *, device: str | None, batch_size: int) -> tuple[str, int]:
    emb = _get_encoder(model)
    if emb is None:
        return "n/a", batch_size
    emb.batch_size = int(batch_size)
    emb.device = device
    if hasattr(emb, "_model"):
        emb._model = None
    if hasattr(emb, "_load"):
        emb._load()
    resolved = getattr(emb, "_resolved_device", None) or svc.resolve_torch_device(device)
    return resolved, int(getattr(emb, "batch_size", batch_size))


def _sanitize_text_cells(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != object and not pd.api.types.is_string_dtype(out[col]):
            continue
        out[col] = out[col].map(
            lambda x: x if not isinstance(x, str) else
            x.replace("\x00", "")
             .replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
             .replace("\u2028", " ").replace("\u2029", " ")
        )
    return out


def _to_csv(df: pd.DataFrame, path: Path, *, mode: str = "w", header: bool = True) -> None:
    _sanitize_text_cells(df).to_csv(
        path, mode=mode, header=header, index=False,
        encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL, lineterminator="\n",
    )


def predict_scores(model, texts: list[str]) -> np.ndarray:
    return model.predict_proba(texts)[:, 1]


def score_dataframe(df: pd.DataFrame, model, *, threshold: float) -> pd.DataFrame:
    if "title" not in df.columns:
        raise ValueError("输入 CSV 需要 title 列")
    out = df.copy()
    texts = out.apply(svc.build_text, axis=1).tolist()
    scores = predict_scores(model, texts)
    out["ml_score"] = scores
    out["ml_drop_threshold"] = threshold
    out["ml_auto_drop"] = scores < threshold
    rescue = out["title"].map(svc.should_rescue_service)
    out.loc[rescue & out["ml_auto_drop"], "ml_auto_drop"] = False
    out["ml_rescued"] = rescue & (scores < threshold)
    return out


def score_file(
    inp: Path,
    out: Path,
    model,
    *,
    threshold: float,
    chunksize: int,
    sample: int,
    split_keep_drop: bool = False,
) -> dict:
    keep_path = drop_path = None
    if split_keep_drop:
        if "_ml_scored" in out.stem:
            keep_path = out.with_name(out.stem.replace("_ml_scored", "_ml_keep", 1) + out.suffix)
            drop_path = out.with_name(out.stem.replace("_ml_scored", "_ml_drop", 1) + out.suffix)
        else:
            keep_path = out.with_name(f"{out.stem}_ml_keep{out.suffix}")
            drop_path = out.with_name(f"{out.stem}_ml_drop{out.suffix}")

    out.parent.mkdir(parents=True, exist_ok=True)
    if sample > 0:
        df = pd.read_csv(inp, encoding="utf-8-sig", nrows=sample)
        scored = score_dataframe(df, model, threshold=threshold)
        _to_csv(scored, out)
        n_drop = int(scored["ml_auto_drop"].sum())
        if split_keep_drop and keep_path and drop_path:
            _to_csv(scored.loc[~scored["ml_auto_drop"]], keep_path)
            _to_csv(scored.loc[scored["ml_auto_drop"]], drop_path)
        _log(f"预览 {len(scored):,} → {out} auto_drop={n_drop:,}")
        return {"n_rows": len(scored), "n_drop": n_drop, "output": str(out)}

    if out.exists():
        out.unlink()
    if split_keep_drop and keep_path and drop_path:
        for p in (keep_path, drop_path):
            if p.exists():
                p.unlink()

    n_rows = n_drop = n_keep = 0
    hours_keep = hours_drop = 0.0
    first = True
    t0 = time.perf_counter()
    for chunk in pd.read_csv(inp, encoding="utf-8-sig", chunksize=chunksize):
        scored = score_dataframe(chunk, model, threshold=threshold)
        drop_mask = scored["ml_auto_drop"].astype(bool)
        n_drop += int(drop_mask.sum())
        n_keep += int((~drop_mask).sum())
        if "duration_seconds" in scored.columns:
            dur = pd.to_numeric(scored["duration_seconds"], errors="coerce").fillna(0)
            hours_keep += float(dur.loc[~drop_mask].sum()) / 3600
            hours_drop += float(dur.loc[drop_mask].sum()) / 3600
        _to_csv(scored, out, mode="a", header=first)
        if split_keep_drop and keep_path and drop_path:
            _to_csv(scored.loc[~drop_mask], keep_path, mode="a", header=first)
            _to_csv(scored.loc[drop_mask], drop_path, mode="a", header=first)
        first = False
        n_rows += len(scored)
        _log(f"  已打分 {n_rows:,} ({time.perf_counter() - t0:.0f}s)")

    _log(f"完成 {out}  n={n_rows:,} drop={n_drop:,} ({100 * n_drop / max(n_rows, 1):.1f}%)")
    if split_keep_drop and keep_path and drop_path:
        _log(f"keep: {keep_path}  ({n_keep:,} 行, {hours_keep:,.1f} h)")
        _log(f"drop: {drop_path}  ({n_drop:,} 行, {hours_drop:,.1f} h)")
    return {
        "input": str(inp),
        "output": str(out),
        "n_rows": n_rows,
        "n_keep": n_keep,
        "n_drop": n_drop,
        "hours_keep": round(hours_keep, 1),
        "hours_drop": round(hours_drop, 1),
        "keep": str(keep_path) if keep_path else None,
        "drop": str(drop_path) if drop_path else None,
        "threshold": threshold,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="exo_service MiniLM 文本打分")
    ap.add_argument("input", help="输入 CSV")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--legacy-strict", action="store_true")
    ap.add_argument("--chunksize", type=int, default=20000)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--split-keep-drop", action="store_true")
    args = ap.parse_args()

    if not args.model.is_file():
        print(f"[ERROR] 缺少模型: {args.model}", file=sys.stderr)
        return 1
    threshold = (
        args.threshold if args.threshold is not None
        else load_threshold(args.calibration, legacy_strict=args.legacy_strict)
    )
    model = load_model(args.model)
    resolved, bs = configure_encoder(model, device=args.device, batch_size=args.batch_size)
    _log(f"encoder ready device={resolved} batch_size={bs} τ={threshold}")

    summary = score_file(
        Path(args.input), args.output, model,
        threshold=threshold,
        chunksize=args.chunksize,
        sample=args.sample,
        split_keep_drop=args.split_keep_drop,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

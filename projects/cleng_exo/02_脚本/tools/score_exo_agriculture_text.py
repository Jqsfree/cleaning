#!/usr/bin/env python3
"""农业 MiniLM 文本打分 — title → ml_score。

score < drop_threshold 则 ml_auto_drop。
默认阈值来自 calibration.t_like（τ 越高越像人标 T；文本治理 v2）；
无 t_like 时回退 recall；--legacy-strict 用旧 strict；--legacy-recall 强制 recall。
支持 LR Pipeline 与 legacy HarvestPrototypeClf。

用法:
  # 单文件
  python3 02_脚本/tools/score_exo_agriculture_text.py \\
    /Users/muse/Downloads/humen-农业/exo农业_f8266cf6_qc_result.csv

  # 指定输出
  python3 02_脚本/tools/score_exo_agriculture_text.py INPUT.csv -o OUT.csv

  # 断点续跑（输出已存在时跳过已写行）
  python3 02_脚本/tools/score_exo_agriculture_text.py INPUT.csv -o OUT.csv --resume

  # 目录下全部 *qc_result.csv（人标源）
  python3 02_脚本/tools/score_exo_agriculture_text.py \\
    --glob '/Users/muse/Downloads/humen-农业/*qc_result.csv'

  # 预览前 500 行 + 打印最低分 20 条
  python3 02_脚本/tools/score_exo_agriculture_text.py INPUT.csv --sample 500 --low 20
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

import exo_agriculture_text_classifier as ag  # noqa: E402

DEFAULT_MODEL = _REPO_ROOT / "models/exo_agriculture_text_clf_f.pkl"
DEFAULT_CALIB = _REPO_ROOT / "models/exo_agriculture_text_clf_f_calibration.json"
DEFAULT_BATCH_SIZE = 256


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_threshold(
    calib_path: Path,
    *,
    legacy_strict: bool = False,
    legacy_recall: bool = False,
) -> float:
    if not calib_path.is_file():
        return 0.49
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    if legacy_strict:
        strict = calib.get("strict") or {}
        return float(strict.get("drop_threshold", strict.get("threshold", 0.08)))
    if legacy_recall:
        recall = calib.get("recall") or {}
        if recall.get("drop_threshold") is not None:
            return float(recall["drop_threshold"])
    t_like = calib.get("t_like") or calib.get("stage2_on_keep") or {}
    if t_like.get("drop_threshold") is not None:
        return float(t_like["drop_threshold"])
    recall = calib.get("recall") or {}
    if recall.get("drop_threshold") is not None:
        return float(recall["drop_threshold"])
    strict = calib.get("strict") or {}
    return float(strict.get("drop_threshold", strict.get("threshold", 0.08)))


def load_model(model_path: Path):
    ag.MiniLMEncoder.__module__ = "exo_agriculture_text_classifier"
    sys.modules.setdefault("exo_agriculture_text_classifier", ag)
    importlib.import_module("exo_agriculture_text_classifier")
    with model_path.open("rb") as fh:
        return pickle.load(fh)


def _get_encoder(model):
    if isinstance(model, Pipeline) and "emb" in model.named_steps:
        return model.named_steps["emb"]
    if hasattr(model, "encoder"):
        return model.encoder
    return None


def configure_encoder(
    model,
    *,
    device: str | None,
    batch_size: int,
) -> tuple[str, int]:
    """Override pickled emb defaults and preload onto the chosen device."""
    emb = _get_encoder(model)
    if emb is None:
        _log("WARN: 未找到 MiniLMEncoder，--device/--batch-size 无效")
        return "n/a", batch_size
    emb.batch_size = int(batch_size)
    emb.device = device  # may be absent on older pickles; set explicitly
    if hasattr(emb, "_model"):
        emb._model = None
    if hasattr(emb, "_load"):
        emb._load()
    resolved = getattr(emb, "_resolved_device", None) or ag.resolve_torch_device(device)
    return resolved, int(getattr(emb, "batch_size", batch_size))


def count_csv_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return sum(1 for _ in reader)


def count_auto_drops(path: Path) -> int:
    n = 0
    for chunk in pd.read_csv(
        path, encoding="utf-8-sig", usecols=["ml_auto_drop"],
        chunksize=100_000, engine="python",
    ):
        col = chunk["ml_auto_drop"]
        if col.dtype == bool:
            n += int(col.sum())
        else:
            n += int(col.astype(str).str.lower().isin(["true", "1"]).sum())
    return n


def _sanitize_text_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse newlines/NUL in object cols so CSV round-trips reliably."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != object and not pd.api.types.is_string_dtype(out[col]):
            continue
        s = out[col]
        out[col] = (
            s.map(lambda x: x if not isinstance(x, str) else
                  x.replace("\x00", "")
                   .replace("\r\n", " ")
                   .replace("\n", " ")
                   .replace("\r", " ")
                   .replace("\u2028", " ")
                   .replace("\u2029", " "))
        )
    return out


def _to_csv(df: pd.DataFrame, path: Path, *, mode: str = "w", header: bool = True) -> None:
    _sanitize_text_cells(df).to_csv(
        path,
        mode=mode,
        header=header,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )


def predict_scores(model, texts: list[str]) -> np.ndarray:
    if isinstance(model, Pipeline):
        return model.predict_proba(texts)[:, 1]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    raise TypeError(f"不支持的模型类型: {type(model)!r}")


def score_dataframe(
    df: pd.DataFrame,
    model,
    *,
    threshold: float,
) -> pd.DataFrame:
    if "title" not in df.columns:
        raise ValueError("输入 CSV 需要 title 列")
    out = df.copy()
    texts = out.apply(ag.build_text, axis=1).tolist()
    scores = predict_scores(model, texts)
    out["ml_score"] = scores
    out["ml_drop_threshold"] = threshold
    out["ml_auto_drop"] = scores < threshold
    if "title" in out.columns:
        rescue = out["title"].map(ag.should_rescue_crop_harvest)
        out.loc[rescue & out["ml_auto_drop"], "ml_auto_drop"] = False
        out["ml_rescued"] = rescue & (scores < threshold)
    else:
        out["ml_rescued"] = False
    return out


def score_file(
    inp: Path,
    out: Path,
    model,
    *,
    threshold: float,
    chunksize: int,
    sample: int,
    resume: bool = False,
    split_keep_drop: bool = False,
) -> dict:
    if not inp.is_file():
        raise FileNotFoundError(f"输入不存在: {inp}")

    emb = _get_encoder(model)
    device = getattr(emb, "_resolved_device", None) or "unknown"
    batch_size = getattr(emb, "batch_size", DEFAULT_BATCH_SIZE) if emb is not None else DEFAULT_BATCH_SIZE
    _log(f"读入: {inp}")
    _log(f"模型: {DEFAULT_MODEL.name}  阈值 drop < {threshold}  device={device}  batch_size={batch_size}")

    keep_path = drop_path = None
    if split_keep_drop:
        if "_ml_scored" in out.stem:
            keep_path = out.with_name(out.stem.replace("_ml_scored", "_ml_keep", 1) + out.suffix)
            drop_path = out.with_name(out.stem.replace("_ml_scored", "_ml_drop", 1) + out.suffix)
        else:
            keep_path = out.with_name(f"{out.stem}_ml_keep{out.suffix}")
            drop_path = out.with_name(f"{out.stem}_ml_drop{out.suffix}")

    if sample > 0:
        df = pd.read_csv(inp, encoding="utf-8-sig", nrows=sample)
        scored = score_dataframe(df, model, threshold=threshold)
        _to_csv(scored, out)
        n_drop = int(scored["ml_auto_drop"].sum())
        if split_keep_drop and keep_path and drop_path:
            _to_csv(scored.loc[~scored["ml_auto_drop"]], keep_path)
            _to_csv(scored.loc[scored["ml_auto_drop"]], drop_path)
        _log(f"预览 {len(scored):,} 行 → {out}  (auto_drop={n_drop:,})")
        return {"input": str(inp), "output": str(out), "n_rows": len(scored), "n_drop": n_drop}

    out.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    n_drop = 0
    first = True
    if resume and out.is_file():
        n_done = count_csv_data_rows(out)
        if n_done > 0:
            n_drop = count_auto_drops(out)
            first = False
            _log(f"resume from {n_done:,} 行 (已有 auto_drop={n_drop:,})")
            if split_keep_drop:
                _log("WARN: --resume 与 --split-keep-drop 同用时不重建 keep/drop，仅追加 scored")
        else:
            out.unlink()
            first = True
    elif out.exists():
        out.unlink()

    if split_keep_drop and keep_path and drop_path and not resume:
        for p in (keep_path, drop_path):
            if p.exists():
                p.unlink()

    n_rows = n_done
    n_keep = 0
    hours_keep = hours_drop = 0.0
    skipped = 0
    t0 = time.perf_counter()
    for chunk in pd.read_csv(inp, encoding="utf-8-sig", chunksize=chunksize):
        if skipped + len(chunk) <= n_done:
            skipped += len(chunk)
            continue
        if skipped < n_done:
            chunk = chunk.iloc[n_done - skipped :]
            skipped = n_done
        scored = score_dataframe(chunk, model, threshold=threshold)
        drop_mask = scored["ml_auto_drop"].astype(bool)
        n_drop += int(drop_mask.sum())
        n_keep += int((~drop_mask).sum())
        if "duration_seconds" in scored.columns:
            dur = pd.to_numeric(scored["duration_seconds"], errors="coerce").fillna(0)
            hours_keep += float(dur.loc[~drop_mask].sum()) / 3600
            hours_drop += float(dur.loc[drop_mask].sum()) / 3600
        _to_csv(scored, out, mode="a", header=first)
        if split_keep_drop and keep_path and drop_path and not (resume and n_done > 0):
            _to_csv(scored.loc[~drop_mask], keep_path, mode="a", header=first)
            _to_csv(scored.loc[drop_mask], drop_path, mode="a", header=first)
        first = False
        n_rows += len(scored)
        _log(f"  已打分 {n_rows:,} 行 ({time.perf_counter() - t0:.0f}s)")

    _log(f"完成: {out}  共 {n_rows:,} 行  auto_drop={n_drop:,} ({n_drop / max(n_rows, 1) * 100:.2f}%)")
    if split_keep_drop and keep_path and drop_path and not (resume and n_done > 0):
        _log(f"keep: {keep_path}  ({n_keep:,} 行, {hours_keep:,.1f} h)")
        _log(f"drop: {drop_path}  ({n_drop:,} 行, {hours_drop:,.1f} h)")
    summary = {"input": str(inp), "output": str(out), "n_rows": n_rows, "n_drop": n_drop}
    if split_keep_drop and keep_path and drop_path:
        summary.update({
            "n_keep": n_keep,
            "keep": str(keep_path),
            "drop": str(drop_path),
            "hours_keep": round(hours_keep, 1),
            "hours_drop": round(hours_drop, 1),
        })
    return summary


def print_low_scores(df: pd.DataFrame, n: int) -> None:
    if n <= 0 or "ml_score" not in df.columns:
        return
    idx = np.argsort(df["ml_score"].to_numpy())[:n]
    print(f"\n最低分 {min(n, len(idx))} 条（人工抽检）:")
    for i in idx:
        title = str(df.iloc[i].get("title", ""))[:100]
        print(f"  [{df.iloc[i]['ml_score']:.4f}] {title}")


def main() -> int:
    ap = argparse.ArgumentParser(description="exo_agriculture MiniLM 文本打分")
    ap.add_argument("input", nargs="?", help="输入 CSV（含 title；可选 channel）")
    ap.add_argument("-o", "--output", type=Path, help="输出 CSV（默认 *_ml_scored.csv）")
    ap.add_argument(
        "--glob", dest="glob_pattern", type=str,
        help="批量打分 glob，如 '/Users/muse/Downloads/humen-农业/*qc_result.csv'",
    )
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--threshold", type=float, default=None, help="覆盖 calibration 阈值")
    ap.add_argument(
        "--legacy-strict", action="store_true",
        help="使用 calibration.strict（旧宁漏勿杀）",
    )
    ap.add_argument(
        "--legacy-recall", action="store_true",
        help="使用 calibration.recall（旧宁杀勿漏默认），忽略 t_like",
    )
    ap.add_argument("--chunksize", type=int, default=30000)
    ap.add_argument("--sample", type=int, default=0, help="只处理前 N 行（预览）")
    ap.add_argument("--low", type=int, default=0, help="打印分数最低的 N 条")
    ap.add_argument(
        "--device", type=str, default=None,
        help="推理设备（默认自动: mps→cuda→cpu）",
    )
    ap.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"MiniLM encode batch size（默认 {DEFAULT_BATCH_SIZE}）",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="若输出已存在则跳过已写行并追加（不断点则删旧重写）",
    )
    ap.add_argument(
        "--split-keep-drop", action="store_true",
        help="同步写出 *_ml_keep_* / *_ml_drop_*（按 ml_auto_drop）",
    )
    args = ap.parse_args()

    if not args.input and not args.glob_pattern:
        ap.error("请指定 input 或 --glob")

    threshold = args.threshold if args.threshold is not None else load_threshold(
        args.calibration,
        legacy_strict=args.legacy_strict,
        legacy_recall=args.legacy_recall,
    )
    model = load_model(args.model)
    resolved, batch_size = configure_encoder(
        model, device=args.device, batch_size=args.batch_size,
    )
    _log(f"encoder ready: device={resolved} batch_size={batch_size}")

    paths: list[Path] = []
    if args.glob_pattern:
        import glob as globmod
        paths = sorted(Path(p) for p in globmod.glob(args.glob_pattern))
    elif args.input:
        paths = [Path(args.input)]

    if not paths:
        print(f"[ERROR] glob 无匹配: {args.glob_pattern!r}", file=sys.stderr)
        return 1

    summaries = []
    for inp in paths:
        out = args.output
        if out is None:
            out = inp.with_name(f"{inp.stem}_ml_scored.csv")
        elif len(paths) > 1:
            out = out.parent / f"{inp.stem}_ml_scored.csv"

        summary = score_file(
            inp, out, model,
            threshold=threshold,
            chunksize=args.chunksize,
            sample=args.sample,
            resume=args.resume,
            split_keep_drop=args.split_keep_drop,
        )
        summaries.append(summary)

        if args.low > 0 and args.sample > 0:
            preview = pd.read_csv(out, encoding="utf-8-sig")
            print_low_scores(preview, args.low)

    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

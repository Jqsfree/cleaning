#!/usr/bin/env python3
"""商品开箱 MiniLM 文本打分 — title → ml_score。

默认阈值来自 calibration.t_like（τ 越高越像人标 T）。

用法:
  python3 02_脚本/tools/score_exo_unbox_text.py INPUT.csv -o OUT.csv
  python3 02_脚本/tools/score_exo_unbox_text.py INPUT.csv --threshold 0.5
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
sys.path.insert(0, str(_SCRIPT_DIR))

import exo_unbox_text_classifier as ls  # noqa: E402

DEFAULT_MODEL = _REPO_ROOT / "models/exo_unbox_text_clf_f.pkl"
DEFAULT_CALIB = _REPO_ROOT / "models/exo_unbox_text_clf_f_calibration.json"
DEFAULT_BATCH_SIZE = 256


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_threshold(
    calib_path: Path,
    *,
    legacy_strict: bool = False,
    legacy_recall: bool = False,
) -> float:
    if legacy_strict and legacy_recall: raise ValueError("Choose one threshold mode")
    if not calib_path.is_file(): raise FileNotFoundError("Calibration missing; supply --threshold explicitly")
    calib=json.loads(calib_path.read_text(encoding="utf-8"))
    key="strict" if legacy_strict else "recall" if legacy_recall else "t_like"
    section=calib.get(key) or (calib.get("stage2_on_keep") if key=="t_like" else {}) or {}
    value=section.get("drop_threshold",section.get("threshold"))
    if value is None: raise ValueError("Requested calibration threshold missing: "+key+"; supply --threshold explicitly")
    return float(value)


def load_model(model_path: Path):
    ls.MiniLMEncoder.__module__ = "exo_unbox_text_classifier"
    sys.modules.setdefault("exo_unbox_text_classifier", ls)
    importlib.import_module("exo_unbox_text_classifier")
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
    resolved = getattr(emb, "_resolved_device", None) or ls.resolve_torch_device(device)
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
    texts = out.apply(ls.build_text, axis=1).tolist()
    scores = predict_scores(model, texts)
    out["ml_score"] = scores
    out["ml_drop_threshold"] = threshold
    out["ml_auto_drop"] = scores < threshold
    if "title" in out.columns:
        rescue = out["title"].map(ls.should_rescue_labor)
        out.loc[rescue & out["ml_auto_drop"], "ml_auto_drop"] = False
        out["ml_rescued"] = rescue & (scores < threshold)
    else:
        out["ml_rescued"] = False
    return out


def score_file(inp, out, model, *, threshold, chunksize, sample, resume=False, split_keep_drop=False, run_identity=None):
    from core.scoring_checkpoint import score_csv
    import math
    if not math.isfinite(threshold) or not 0<=threshold<=1: raise ValueError("Invalid scoring threshold")
    if not run_identity: raise ValueError("Model/code identity required; use the CLI or provide run_identity")
    split_paths=None
    if split_keep_drop:
        if "_ml_scored" in out.stem:
            split_paths=tuple(out.with_name(out.stem.replace("_ml_scored",suffix,1)+out.suffix) for suffix in ("_ml_keep","_ml_drop"))
        else:
            split_paths=tuple(out.with_name(out.stem+suffix+out.suffix) for suffix in ("_ml_keep","_ml_drop"))
    return score_csv(inp,out,lambda frame:score_dataframe(frame,model,threshold=threshold),
                     identity=dict(run_identity,threshold=threshold),chunksize=chunksize,sample=sample,
                     resume=resume,split_paths=split_paths)


def print_low_scores(df: pd.DataFrame, n: int) -> None:
    if n <= 0 or "ml_score" not in df.columns:
        return
    idx = np.argsort(df["ml_score"].to_numpy())[:n]
    print(f"\n最低分 {min(n, len(idx))} 条（人工抽检）:")
    for i in idx:
        title = str(df.iloc[i].get("title", ""))[:100]
        print(f"  [{df.iloc[i]['ml_score']:.4f}] {title}")


def main() -> int:
    ap = argparse.ArgumentParser(description="exo_unbox MiniLM 文本打分（仅 title）")
    ap.add_argument("input", nargs="?", help="输入 CSV（含 title；可选 channel）")
    ap.add_argument("-o", "--output", type=Path, help="输出 CSV（默认 *_ml_scored.csv）")
    ap.add_argument(
        "--glob", dest="glob_pattern", type=str,
        help="批量打分 glob，如 '/Users/muse/Downloads/humen-商品开箱/*qc_result.csv'",
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
        help="校验输入、模型、阈值和已提交输出后续跑；旧无 checkpoint 输出拒绝续跑",
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
    from core.data_profile import file_hash
    model_sha=file_hash(args.model)
    model = load_model(args.model)
    if file_hash(args.model)!=model_sha: raise ValueError("Model changed while loading")
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
            run_identity={"model_sha256":model_sha,"calibration_sha256":file_hash(args.calibration) if args.calibration.exists() else None,
                          "text_code_sha256":file_hash(ls.__file__),"scoring_code_sha256":file_hash(__file__),
                          "device":resolved,"encoder_batch_size":batch_size},
        )
        if file_hash(args.model)!=model_sha: raise ValueError("Model changed during scoring")
        summaries.append(summary)

        if args.low > 0 and args.sample > 0:
            preview = pd.read_csv(out, encoding="utf-8-sig")
            print_low_scores(preview, args.low)

    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stage -1：元数据过滤器（CLIP 之前）。

对 title/keyword/duration/view 打分，阈值默认 0.3；blocklist keyword 硬拒。
特征构造与 experiments/train_metadata_filter.py 完全一致。

用法:
  PYTHONPATH=02_脚本 python 02_脚本/tools/apply_exo_agriculture_metadata_filter.py \\
    data/runs/exo_agriculture/machine_0814/06_tools/农业采集_0814_semantic_remain.csv \\
    -o data/runs/exo_agriculture/machine_0814/06_tools/metadata_filter_0822/

  # 带人工标验收误杀率（超过 calibration 上限则 exit 2）
  PYTHONPATH=02_脚本 python 02_脚本/tools/apply_exo_agriculture_metadata_filter.py \\
    INPUT.csv -o OUT/ --eval-labels labeled.csv --halt-on-high-t-hurt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from core.metadata_filter import (  # noqa: E402
    assign_metadata_actions,
    eval_reject_overturn,
    keyword_bucket,
    load_blocklist_keywords,
    load_calibration,
    score_metadata_frame,
)

DEFAULT_MODEL_DIR = _REPO_ROOT / "models/exo_agriculture_metadata_filter"
DEFAULT_BLOCKLIST = DEFAULT_MODEL_DIR / "blocklist_keywords.csv"


def load_labels(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, low_memory=False)
    if "human_label" in raw.columns:
        lab = raw["human_label"].str.lower().map({"pass": 1, "fail": 0})
    elif "qc_result" in raw.columns:
        qc = raw["qc_result"].astype(str).str.strip().str.upper()
        lab = qc.map(lambda x: 1 if x == "T" else (0 if x.startswith("F") else np.nan))
    else:
        raise ValueError("eval-labels 需含 human_label 或 qc_result")
    out = raw.assign(y=lab).dropna(subset=["y"])
    out["video_id"] = out["video_id"].astype(str).str.strip()
    return out


def sample_reject_for_review(
    reject_df: pd.DataFrame,
    *,
    out_path: Path,
    n: int,
    seed: int,
) -> dict:
    if reject_df.empty or n <= 0:
        return {"n_sample": 0, "path": None}
    n_take = min(n, len(reject_df))
    part = reject_df.sample(n=n_take, random_state=seed).copy()
    part["review_route"] = "metadata_reject_overturn"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part.to_csv(out_path, index=False)
    return {"n_sample": int(n_take), "path": str(out_path.resolve())}


def apply_filter(
    input_csv: Path,
    *,
    out_dir: Path,
    model_dir: Path,
    blocklist_path: Path,
    threshold: float,
    chunksize: int,
    sample_reject: int,
    seed: int,
    eval_labels: Path | None,
    halt_on_high_t_hurt: bool,
) -> dict:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    blocklist = load_blocklist_keywords(blocklist_path)
    calib = load_calibration(model_dir)

    pass_path = out_dir / f"{input_csv.stem}_metadata_pass.csv"
    reject_path = out_dir / f"{input_csv.stem}_metadata_reject.csv"
    scored_path = out_dir / "metadata_scored.csv"
    review_path = out_dir / f"{input_csv.stem}_metadata_reject_review_sample.csv"

    n_in = 0
    n_pass = 0
    n_reject = 0
    n_blocklist = 0
    n_score_reject = 0
    write_pass = write_reject = write_scored = True
    bucket_acc: dict[str, dict[str, int]] = {}

    def _acc_bucket(part: pd.DataFrame) -> None:
        for bucket, sub in part.groupby(part["keyword"].map(keyword_bucket)):
            if bucket not in bucket_acc:
                bucket_acc[bucket] = {"n": 0, "n_reject": 0}
            bucket_acc[bucket]["n"] += len(sub)
            bucket_acc[bucket]["n_reject"] += int(sub["metadata_action"].eq("metadata_reject").sum())

    for chunk in pd.read_csv(input_csv, chunksize=chunksize, low_memory=False):
        n_in += len(chunk)
        scores = score_metadata_frame(chunk, model_dir=model_dir)
        scored = assign_metadata_actions(
            chunk, scores, blocklist=blocklist, threshold=threshold,
        )
        pass_mask = scored["metadata_action"].eq("metadata_pass")
        reject_mask = ~pass_mask
        n_pass += int(pass_mask.sum())
        n_reject += int(reject_mask.sum())
        n_blocklist += int(scored["metadata_reject_reason"].eq("blocklist_keyword").sum())
        n_score_reject += int(scored["metadata_reject_reason"].eq("score_below_threshold").sum())
        _acc_bucket(scored)

        scored.to_csv(scored_path, mode="w" if write_scored else "a", header=write_scored, index=False)
        write_scored = False
        scored.loc[pass_mask].to_csv(pass_path, mode="w" if write_pass else "a", header=write_pass, index=False)
        write_pass = False
        scored.loc[reject_mask].to_csv(
            reject_path, mode="w" if write_reject else "a", header=write_reject, index=False,
        )
        write_reject = False
        print(
            f"  chunk cumulative: in={n_in:,} pass={n_pass:,} reject={n_reject:,}",
            flush=True,
        )

    reject_df = pd.read_csv(reject_path, low_memory=False) if reject_path.exists() else pd.DataFrame()
    review = sample_reject_for_review(
        reject_df, out_path=review_path, n=sample_reject, seed=seed,
    )

    bucket_report = {}
    for bucket, acc in bucket_acc.items():
        bucket_report[bucket] = {
            "n": acc["n"],
            "n_reject": acc["n_reject"],
            "reject_rate": round(acc["n_reject"] / max(acc["n"], 1), 4),
        }

    overturn = None
    halt = False
    halt_reason = None
    if eval_labels and eval_labels.is_file():
        labels = load_labels(eval_labels)
        scored_all = pd.read_csv(scored_path, usecols=[
            "video_id", "metadata_action", "metadata_score", "metadata_reject_reason",
        ], low_memory=False)
        overturn = eval_reject_overturn(labels, scored_all)
        max_rate = float(calib.get("max_positive_reject_rate", 0.10))
        if halt_on_high_t_hurt and overturn["t_hurt_rate"] > max_rate:
            halt = True
            halt_reason = (
                f"T误杀率 {overturn['t_hurt_rate']:.1%} > 上限 {max_rate:.1%} "
                f"(训练期预期约 {calib.get('expected_positive_reject_rate', 0.07):.1%})"
            )

    clip_saved_pct = round(100.0 * n_reject / max(n_in, 1), 2)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "metadata_filter",
        "input": str(input_csv.resolve()),
        "out_dir": str(out_dir.resolve()),
        "model_dir": str(model_dir.resolve()),
        "blocklist_path": str(blocklist_path.resolve()),
        "threshold": threshold,
        "n_candidates": n_in,
        "n_metadata_pass": n_pass,
        "n_metadata_reject": n_reject,
        "n_reject_blocklist": n_blocklist,
        "n_reject_score": n_score_reject,
        "clip_candidates_saved": n_reject,
        "clip_saved_pct": clip_saved_pct,
        "clip_will_process": n_pass,
        "keyword_bucket_reject": bucket_report,
        "pass_path": str(pass_path.resolve()),
        "reject_path": str(reject_path.resolve()),
        "scored_path": str(scored_path.resolve()),
        "reject_review_sample": review,
        "calibration": calib,
        "eval_overturn": overturn,
        "halt": halt,
        "halt_reason": halt_reason,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_path = out_dir / "apply.log"
    log_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if halt:
        print(f"\n[HALT] {halt_reason}", file=sys.stderr)
        raise SystemExit(2)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="exo_agriculture Stage -1 metadata filter")
    ap.add_argument("input", type=Path, help="CLIP 前候选 CSV")
    ap.add_argument("-o", "--out-dir", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--blocklist", type=Path, default=DEFAULT_BLOCKLIST)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--chunksize", type=int, default=50_000)
    ap.add_argument("--sample-reject", type=int, default=100, help="metadata_reject 抽检条数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-labels", type=Path, default=None)
    ap.add_argument(
        "--halt-on-high-t-hurt", action="store_true",
        help="有人工标且 T误杀率超 calibration 上限时 exit 2",
    )
    args = ap.parse_args()
    if not (args.model_dir / "clf.joblib").is_file():
        print(f"[ERROR] 缺少模型: {args.model_dir}", file=sys.stderr)
        return 2
    apply_filter(
        args.input,
        out_dir=args.out_dir,
        model_dir=args.model_dir,
        blocklist_path=args.blocklist,
        threshold=args.threshold,
        chunksize=args.chunksize,
        sample_reject=args.sample_reject,
        seed=args.seed,
        eval_labels=args.eval_labels,
        halt_on_high_t_hurt=args.halt_on_high_t_hurt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

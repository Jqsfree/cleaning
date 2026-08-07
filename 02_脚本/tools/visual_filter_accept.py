#!/usr/bin/env python3
"""真人直播视觉过滤：准备独立验收样本与验证验收门。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.run_manifest import maybe_update_stage  # noqa: E402
from core.visual_filter import (  # noqa: E402
    acceptance_decision,
    exclude_video_ids,
    normalize_qc_result,
)


def _read(path: str) -> pd.DataFrame:
    if Path(path).suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def _sample_size(
    population: int,
    *,
    confidence: float = 0.90,
    margin: float = 0.05,
    p: float = 0.5,
) -> int:
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    n0 = z * z * p * (1 - p) / (margin * margin)
    n = n0 / (1 + (n0 - 1) / max(population, 1))
    return min(population, math.ceil(n))


def _prepare(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    keep = _read(args.keep)
    drop = _read(args.drop)
    excluded: set[str] = set()
    if args.exclude_labels:
        prior = _read(args.exclude_labels)
        excluded = set(prior["video_id"].astype(str).str.strip())
    keep = exclude_video_ids(keep, excluded)
    drop = exclude_video_ids(drop, excluded)
    keep_n = _sample_size(
        len(keep), confidence=args.confidence, margin=args.margin,
    )
    keep_sample = keep.sample(n=keep_n, random_state=args.seed).copy()
    drop_n = min(args.drop_n, len(drop))
    drop_sample = drop.sample(n=drop_n, random_state=args.seed + 1).copy()
    for frame, kind in ((keep_sample, "acceptance"), (drop_sample, "drop_overturn")):
        frame["qc_result"] = ""
        frame["qc_dimension"] = kind
    keep_path = out / "acceptance_keep_sample.csv"
    drop_path = out / "drop_overturn_sample.csv"
    keep_sample.to_csv(keep_path, index=False)
    drop_sample.to_csv(drop_path, index=False)
    summary = {
        "keep_population": len(keep),
        "keep_sample": len(keep_sample),
        "drop_population": len(drop),
        "drop_sample": len(drop_sample),
        "confidence": args.confidence,
        "margin": args.margin,
        "excluded_prior_labels": len(excluded),
        "keep_output": str(keep_path),
        "drop_output": str(drop_path),
    }
    (out / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _verify(args: argparse.Namespace) -> None:
    keep_labels = _read(args.keep_labels)
    drop_labels = _read(args.drop_labels)
    keep_pool = _read(args.keep_pool)
    keep_qc = normalize_qc_result(keep_labels["qc_result"])
    drop_qc = normalize_qc_result(drop_labels["qc_result"])
    keep_valid = keep_qc.notna()
    drop_valid = drop_qc.notna()
    pass_count = int((keep_qc[keep_valid] == True).sum())  # noqa: E712
    overturn_count = int((drop_qc[drop_valid] == True).sum())  # noqa: E712
    hours = pd.to_numeric(
        keep_pool.get("duration_seconds"), errors="coerce",
    ).sum() / 3600
    result = acceptance_decision(
        pass_count=pass_count,
        labeled_count=int(keep_valid.sum()),
        kept_hours=float(hours),
        overturn_count=overturn_count,
        drop_labeled_count=int(drop_valid.sum()),
        confidence=args.confidence,
        min_pass_lower=args.min_pass_lower,
        min_hours=args.min_hours,
        max_overturn=args.max_overturn,
    )
    report = {
        **result,
        "pass_count": pass_count,
        "keep_labeled": int(keep_valid.sum()),
        "overturn_count": overturn_count,
        "drop_labeled": int(drop_valid.sum()),
        "criteria": {
            "confidence": args.confidence,
            "min_pass_lower": args.min_pass_lower,
            "min_hours": args.min_hours,
            "max_overturn": args.max_overturn,
        },
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "acceptance_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    maybe_update_stage(
        out,
        "human_live_visual_accept",
        paths={"report": str(report_path), "keep_pool": str(args.keep_pool)},
        stats=report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="真人直播视觉过滤验收")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--keep", required=True)
    prepare.add_argument("--drop", required=True)
    prepare.add_argument("--exclude-labels")
    prepare.add_argument("-o", "--output-dir", required=True)
    prepare.add_argument("--confidence", type=float, default=0.90)
    prepare.add_argument("--margin", type=float, default=0.05)
    prepare.add_argument("--drop-n", type=int, default=50)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(func=_prepare)

    verify = sub.add_parser("verify")
    verify.add_argument("--keep-labels", required=True)
    verify.add_argument("--drop-labels", required=True)
    verify.add_argument("--keep-pool", required=True)
    verify.add_argument("-o", "--output-dir", required=True)
    verify.add_argument("--confidence", type=float, default=0.90)
    verify.add_argument("--min-pass-lower", type=float, default=0.85)
    verify.add_argument("--min-hours", type=float, default=80_000)
    verify.add_argument("--max-overturn", type=float, default=0.08)
    verify.set_defaults(func=_verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

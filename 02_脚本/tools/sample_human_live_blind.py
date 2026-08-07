#!/usr/bin/env python3
"""从规则 keep 池生成与训练标签隔离的时长/频道分层盲样。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.human_live_multiframe import make_blind_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="真人直播独立盲样")
    parser.add_argument("pool")
    parser.add_argument("--exclude-labels", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-n", type=int, default=385)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pool_path = Path(args.pool)
    pool = (
        pd.read_parquet(pool_path)
        if pool_path.suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(pool_path, dtype=str, low_memory=False)
    )
    labels = pd.read_csv(args.exclude_labels, dtype={"video_id": str})
    excluded = set(labels["video_id"].dropna().astype(str))
    sample = make_blind_sample(pool, excluded, n=args.n, seed=args.seed)
    sample["qc_result"] = ""
    sample["qc_note"] = ""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False)
    summary = {
        "pool": str(pool_path.resolve()),
        "excluded_labels": len(excluded),
        "rows": len(sample),
        "unique_channels": int(
            sample["channel"].fillna("").astype(str).nunique()
        ),
        "duration_hours": round(
            float(pd.to_numeric(sample["duration_seconds"], errors="coerce").sum())
            / 3600,
            2,
        ),
        "strata": sample["sample_stratum"].value_counts().sort_index().to_dict(),
        "seed": args.seed,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Discover a CSV/Parquet file's title topics and recurring title patterns.

This script does not decide which topics are unwanted and never drops rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.title_distribution import analyze_title_distribution


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV, TSV or Parquet with video_id and title")
    parser.add_argument("-o", "--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--topics", type=int, default=20, help="Requested number of exploratory topics")
    parser.add_argument("--sample-size", type=int, default=10000,
                        help="Uniform row reservoir for fitting; full input is still assigned")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-features", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder", help="Optional existing local sentence-transformer directory")
    args = parser.parse_args(argv)
    report = analyze_title_distribution(
        args.input, args.out, topics=args.topics, sample_size=args.sample_size,
        batch_size=args.batch_size, seed=args.seed, max_features=args.max_features,
        encoder=args.encoder,
    )
    print(json.dumps({"status": report["status"], "output": str(args.out.resolve()),
                      "intake": report["intake"], "topics_fitted": report["parameters"]["topics_fitted"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

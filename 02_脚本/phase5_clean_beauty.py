#!/usr/bin/env python3
"""
phase5_clean_beauty.py — 美妆规则清洗入口

用法:
  python3 phase5_clean_beauty.py data/runs/beauty/001_baseline/xxx_raw.parquet -o data/runs/beauty/005_clean/run01
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from categories.beauty.cleaner import clean


def main():
    parser = argparse.ArgumentParser(description="美妆规则清洗")
    parser.add_argument("input", help="baseline parquet 文件")
    parser.add_argument("-o", "--output-dir", default="data/runs/beauty/005_clean/run01")
    parser.add_argument("-r", "--run", default="run01")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    raw_stem = os.path.splitext(os.path.basename(args.input))[0]
    stem = raw_stem.replace("_raw", "") if "_raw" in raw_stem else raw_stem

    summary = clean(
        input_path=args.input,
        stem=stem,
        output_dir=args.output_dir,
        raw_name=stem,
        run=args.run,
    )

    print()
    print("=" * 62)
    print("  Phase 5 — 美妆规则清洗 完成")
    print("=" * 62)
    print(f"  总行数:     {summary['total_rows']:>12,}")
    print(f"  保留:       {summary['total_keep']:>12,}  ({summary['retention_pct']}%)")
    print(f"  移除:       {summary['total_drop']:>12,}")
    print(f"  耗时:       {summary['elapsed_sec']:>11.1f}s")
    print(f"  产物:       {args.output_dir}/")
    print("=" * 62)


if __name__ == "__main__":
    main()

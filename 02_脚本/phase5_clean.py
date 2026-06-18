#!/usr/bin/env python3
"""
phase5_clean.py — 规则清洗：对 baseline parquet 应用黑/白名单规则

用法:
  python3 phase5_clean.py data/runs/001_baseline/xxx_raw.parquet -o data/runs/005_clean/run01/
"""

import sys, os, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from categories.language_teaching.cleaner import clean


def main():
    parser = argparse.ArgumentParser(description="规则清洗：应用黑/白名单规则过滤")
    parser.add_argument("input", help="baseline parquet 文件")
    parser.add_argument("-o", "--output-dir", default="data/runs/005_clean/run01",
                        help="输出目录")
    parser.add_argument("--keep-score", type=int, default=None,
                        help="high 阈值（默认从规则读取）")
    parser.add_argument("--gray-low", type=int, default=None,
                        help="gray 低分阈值")
    parser.add_argument("--med-min", type=int, default=None,
                        help="medium 最低分")
    parser.add_argument("-r", "--run", default="run01",
                        help="run 名称")
    parser.add_argument("--no-medium", action="store_true",
                        help="只保留 high，丢弃 medium")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    raw_stem = os.path.splitext(os.path.basename(args.input))[0]
    # strip _raw suffix for cleaner naming
    stem = raw_stem.replace("_raw", "") if "_raw" in raw_stem else raw_stem

    summary = clean(
        input_path=args.input,
        stem=stem,
        output_dir=args.output_dir,
        raw_name=stem,
        run=args.run,
        keep_score=args.keep_score,
        gray_low=args.gray_low,
        med_min=args.med_min,
        no_medium=args.no_medium,
    )

    print()
    print("=" * 62)
    print("  Phase 5 — 规则清洗 完成")
    print("=" * 62)
    print(f"  总行数:     {summary['total_rows']:>12,}")
    print(f"  保留:       {summary['total_keep']:>12,}  ({summary['retention_pct']}%)")
    print(f"    high:     {summary['total_keep_high']:>12,}")
    print(f"    medium:   {summary['total_keep_medium']:>12,}")
    print(f"  移除:       {summary['total_drop']:>12,}")
    print(f"  耗时:       {summary['elapsed_sec']:>11.1f}s")
    print(f"  产物:       {args.output_dir}/")
    print("=" * 62)


if __name__ == "__main__":
    main()

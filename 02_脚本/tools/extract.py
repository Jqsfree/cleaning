#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
"""
extract_by_field.py — 从 CSV 中按指定字段值提取记录

用法:
  python3 extract_by_field.py input.csv -c qc_status -v pass
  python3 extract_by_field.py input.csv -c keyword -v "双人对话"
  python3 extract_by_field.py input.csv -c video_id -v abc123,def456
  python3 extract_by_field.py input.csv -c qc_status -v pass -o custom.csv
"""

import argparse, sys, os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="按字段值提取 CSV 记录")
    parser.add_argument("input", help="输入 CSV")
    parser.add_argument("-c", "--column", required=True, help="要匹配的列名")
    parser.add_argument("-v", "--value", required=True, help="匹配值，多个用逗号分隔")
    parser.add_argument("-o", "--output", default=None,
                        help="输出路径 (默认: {原名}_{column}_{value}.csv)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    df = pd.read_csv(args.input, dtype=str)

    if args.column not in df.columns:
        print(f"[ERROR] 列 '{args.column}' 不存在。可用列: {', '.join(df.columns)}")
        sys.exit(1)

    values = [v.strip() for v in args.value.split(",")]
    subset = df[df[args.column].isin(values)]

    if args.output:
        out = args.output
    else:
        stem = os.path.splitext(args.input)[0]
        tag = args.value.replace(",", "_").replace(" ", "")
        out = f"{stem}_{tag}.csv"

    subset.to_csv(out, index=False)

    if "duration_seconds" in df.columns:
        all_h = df["duration_seconds"].astype(float).sum() / 3600
        sub_h = subset["duration_seconds"].astype(float).sum() / 3600
        print(f"输入: {len(df):,} 条 ({all_h:,.1f}h)")
        print(f"匹配: {len(subset):,} 条 ({sub_h:,.1f}h)  ← {(len(subset)/max(len(df),1)*100):.1f}%")
    else:
        print(f"输入: {len(df):,} 条")
        print(f"匹配: {len(subset):,} 条  ← {(len(subset)/max(len(df),1)*100):.1f}%")
    print(f"输出: {out}")


if __name__ == "__main__":
    main()

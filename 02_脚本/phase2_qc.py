#!/usr/bin/env python3
"""
phase2_qc.py -- SOP Phase 2: LLM 文本质检入口

这是 chunk_text_qc_v2.py 的薄封装，为 SOP Phase 2 提供专用入口。
实际 QC 逻辑在 chunk_text_qc_v2.run_text_qc() 中。

用法:
  python3 phase2_qc.py audit_sample_v1.parquet -o data/runs/002_audit/ -w 20
  python3 phase2_qc.py audit_sample_v1.parquet -o data/runs/002_audit/ -w 20 --force
"""

import sys, os, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk_text_qc_v2 import run_text_qc, DEFAULT_MODEL, DEFAULT_WORKERS


def main():
    parser = argparse.ArgumentParser(description="SOP Phase 2: LLM 文本质检")
    parser.add_argument("input", help="输入 CSV 或 Parquet 文件")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="输出目录（默认与输入同目录）")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="即使已有 QC 结果也重新运行")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))

    # --force: 清除已有 QC 结果
    if args.force:
        import pandas as pd
        ext = os.path.splitext(args.input)[1].lower()
        df = pd.read_parquet(args.input).fillna("").astype(str) if ext == ".parquet" \
            else pd.read_csv(args.input, dtype=str, low_memory=False).fillna("")
        for col in ["qc_text_result", "qc_text_model", "qc_run_id", "qc_error_reason"]:
            if col in df.columns:
                df[col] = ""
        if ext == ".parquet":
            df.to_parquet(args.input, index=False)
        else:
            df.to_csv(args.input, index=False, encoding="utf-8-sig")
        print(f"[FORCE] 已清除 QC 结果，重新运行")

    summary = run_text_qc(args.input, output_dir, model=args.model,
                          workers=args.workers, dry_run=args.dry_run)
    return summary


if __name__ == "__main__":
    main()

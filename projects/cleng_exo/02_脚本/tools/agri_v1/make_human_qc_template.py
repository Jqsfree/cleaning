#!/usr/bin/env python3
"""make_human_qc_template.py — 由 02_sample 抽样表生成农业 v1 §2 人工标注模板。

用途：`tools/batch_ops/sample_qc.py` 输出的抽样表只有候选池原始列，人工标注需要
`core.agri_v1_schema.audit_sample_columns()` 的固定列序（含空标注列）。

用法:
  PYTHONPATH=02_脚本 .venv/bin/python3 02_脚本/tools/agri_v1/make_human_qc_template.py \
      data/runs/exo_agriculture/machine_0818/02_sample/v1_human_qc_c90_srs/*_sample_*.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.agri_v1_schema import (  # noqa: E402
    audit_sample_columns,
    merge_agri_v1_context,
    normalize_agri_v1_frame,
)

DEFAULT_NOTE = (
    "填写 v1 §2 字段；valid = human_present=1 AND crop_interaction=1；"
    "标完后把判定写入 human_label（pass/fail）"
)


def build_template(sample_csv: Path, *, note: str = DEFAULT_NOTE) -> pd.DataFrame:
    raw = pd.read_csv(sample_csv, dtype=str)
    frame = normalize_agri_v1_frame(raw)
    frame = merge_agri_v1_context(frame, raw)

    for col in ("human_label", "qc_result"):
        frame[col] = frame[col] if col in frame.columns else ""
    frame["review_notes"] = note

    cols = audit_sample_columns()
    for col in cols:
        if col not in frame.columns:
            frame[col] = ""
    out = frame[cols].copy()

    dupes = [c for c in out.columns if list(out.columns).count(c) > 1]
    if dupes:
        raise ValueError(f"模板列重复: {sorted(set(dupes))}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="生成农业 v1 人工标注模板（去重列序）")
    ap.add_argument("sample_csv", type=Path,
                    help="tools/batch_ops/sample_qc.py 输出的抽样表 CSV")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="输出 CSV（默认: 同目录 {抽样文件名}_labeled_template.csv）")
    ap.add_argument("--note", default=DEFAULT_NOTE, help="review_notes 预填文本")
    args = ap.parse_args()

    if not args.sample_csv.exists():
        print(f"[ERROR] 文件不存在: {args.sample_csv}")
        sys.exit(1)

    out_path = args.out or args.sample_csv.with_name(
        f"{args.sample_csv.stem}_labeled_template.csv"
    )
    template = build_template(args.sample_csv, note=args.note)
    template.to_csv(out_path, index=False)

    print(f"样本行数: {len(template):,}")
    print(f"列({len(template.columns)}): {', '.join(template.columns)}")
    print(f"产物: {out_path}")


if __name__ == "__main__":
    main()

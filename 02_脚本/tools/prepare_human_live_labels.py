#!/usr/bin/env python3
"""规范真人直播标签，并按频道拆 train/calibration/holdout。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.visual_filter import split_labeled_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="准备真人直播视觉训练标签")
    parser.add_argument("input", help="人工 QC CSV")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--conflicts-json", help="规则验收 JSON（含 T_injured_titles）")
    parser.add_argument("--source", choices=["human", "machine"], default="machine")
    parser.add_argument("--batch", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input, dtype=str, low_memory=False)
    if not {"video_id", "qc_result"}.issubset(frame.columns):
        raise SystemExit("[ERROR] 输入需要 video_id、qc_result")
    if "source" not in frame.columns:
        frame["source"] = args.source
    else:
        frame["source"] = frame["source"].fillna(args.source)
    if "batch" not in frame.columns:
        frame["batch"] = args.batch

    conflict_titles: set[str] = set()
    if args.conflicts_json:
        report = json.loads(Path(args.conflicts_json).read_text(encoding="utf-8"))
        conflict_titles = {
            str(title).strip() for title in report.get("T_injured_titles", [])
        }
    conflict_ids = set(
        frame.loc[
            frame.get("title", pd.Series("", index=frame.index))
            .fillna("").astype(str).str.strip().isin(conflict_titles),
            "video_id",
        ].astype(str)
    )
    prepared = split_labeled_frame(
        frame,
        conflict_ids=conflict_ids,
        group_col="channel",
        seed=args.seed,
    )
    prepared.to_csv(out_dir / "labels_prepared.csv", index=False)
    prepared.loc[prepared["split"].eq("review")].to_csv(
        out_dir / "conflict_review.csv", index=False,
    )
    summary = {
        "input": str(Path(args.input).resolve()),
        "source": args.source,
        "batch": args.batch,
        "rows": len(prepared),
        "conflicts": int(prepared["split"].eq("review").sum()),
        "splits": prepared["split"].value_counts(dropna=False).to_dict(),
        "note": "holdout 仅模型独立评估；最终批次验收另抽随机样本",
    }
    (out_dir / "label_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

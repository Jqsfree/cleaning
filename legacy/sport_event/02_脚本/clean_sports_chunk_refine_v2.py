#!/usr/bin/env python3
"""
v2 精炼 — 对 clean_v2_all.csv 做第二轮过滤。

依赖: clean_sports_chunk_refine.py（同目录）

用法:
  python3 clean_sports_chunk_refine_v2.py <cleaned_all.csv>
"""

import csv, json, sys
from datetime import datetime, timezone
from pathlib import Path

# 引入同目录的精炼规则
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_sports_chunk_refine import refine_row

VERSION = "v2-qc"
OUTPUT_DIR = Path("output")


def main():
    if len(sys.argv) < 2:
        print("用法: python3 clean_sports_chunk_refine_v2.py <cleaned_all.csv>")
        print("示例: python3 clean_sports_chunk_refine_v2.py output/sports_chunk_01_clean_v2_all.csv")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"[ERROR] 文件不存在: {input_path}")
        sys.exit(1)

    stem = input_path.stem.replace("_clean_v2_all", "")
    out_clean = OUTPUT_DIR / f"{stem}_clean_{VERSION}_refined.csv"
    out_drop = OUTPUT_DIR / f"{stem}_clean_{VERSION}_refined_dropped.csv"
    out_ids = OUTPUT_DIR / f"{stem}_clean_{VERSION}_refined_ids.csv"
    out_summary = OUTPUT_DIR / f"{stem}_clean_{VERSION}_refined_summary.json"

    summary = {
        "version": VERSION, "input": str(input_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": {"r2_blacklist": {"dropped": 0}, "r2_context": {"dropped": 0}, "r2_weak_entity": {"dropped": 0}},
        "total_in": 0, "total_keep": 0, "total_drop": 0,
    }

    with input_path.open(newline="", encoding="utf-8", errors="replace") as fin, \
         out_clean.open("w", newline="", encoding="utf-8") as fclean, \
         out_drop.open("w", newline="", encoding="utf-8") as fdrop, \
         out_ids.open("w", encoding="utf-8") as fids:

        reader = csv.DictReader(fin)
        base = reader.fieldnames or []
        extra = ["refine_label", "refine_reason"]
        out_fields = base + [f for f in extra if f not in base]

        wc = csv.DictWriter(fclean, fieldnames=out_fields, extrasaction="ignore")
        wd = csv.DictWriter(fdrop, fieldnames=out_fields, extrasaction="ignore")
        wc.writeheader()
        wd.writeheader()
        fids.write("video_id\n")

        for i, row in enumerate(reader, 1):
            summary["total_in"] += 1
            label, reason = refine_row(row)
            row["refine_label"] = label
            row["refine_reason"] = reason
            if label == "keep":
                wc.writerow(row)
                fids.write(f"{row.get('video_id', '')}\n")
                summary["total_keep"] += 1
            else:
                wd.writerow(row)
                summary["total_drop"] += 1
                if reason.startswith("r2_blacklist"):
                    summary["steps"]["r2_blacklist"]["dropped"] += 1
                elif reason.startswith("r2_weak_entity"):
                    summary["steps"]["r2_weak_entity"]["dropped"] += 1
                else:
                    summary["steps"]["r2_context"]["dropped"] += 1
            if i % 100_000 == 0:
                print(f"  {i:,} | keep {summary['total_keep']:,} | drop {summary['total_drop']:,}")

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["outputs"] = {"clean": str(out_clean), "dropped": str(out_drop), "ids": str(out_ids)}
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    t = summary["total_in"]
    print(f"\n=== {VERSION} REFINE COMPLETE ===")
    print(f"  In:   {t:,}")
    print(f"  Keep: {summary['total_keep']:,} ({100*summary['total_keep']/max(t,1):.1f}%)")
    print(f"  Drop: {summary['total_drop']:,}")
    for v in summary["outputs"].values():
        print(f"  {v}")


if __name__ == "__main__":
    main()

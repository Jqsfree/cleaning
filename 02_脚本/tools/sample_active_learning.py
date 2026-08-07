#!/usr/bin/env python3
"""从视觉过滤结果抽边界、多样 keep 与 drop overturn 样本。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.visual_filter import select_active_learning_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="真人直播主动学习抽样")
    parser.add_argument("scored", help="human_live_visual_scored.parquet")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--boundary", type=int, default=150)
    parser.add_argument("--diverse-keep", type=int, default=100)
    parser.add_argument("--drop", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scored = pd.read_parquet(args.scored)
    index = pd.read_csv(
        Path(args.embeddings) / "index.csv", dtype={"video_id": str},
    )
    if scored["video_id"].astype(str).tolist() != index["video_id"].astype(str).tolist():
        raise SystemExit("[ERROR] scored 与 embedding index 顺序不一致")
    embeddings = np.load(Path(args.embeddings) / "embeddings.npy", mmap_mode="r")
    sample = select_active_learning_sample(
        scored,
        np.asarray(embeddings),
        n_boundary=args.boundary,
        n_diverse_keep=args.diverse_keep,
        n_drop=args.drop,
        seed=args.seed + args.round,
    )
    sample["qc_result"] = ""
    sample["qc_dimension"] = "human_live_visual"
    sample["active_round"] = args.round
    path = out / f"active_round{args.round:02d}.csv"
    sample.to_csv(path, index=False)
    summary = {
        "round": args.round,
        "rows": len(sample),
        "routes": sample["sample_route"].value_counts().to_dict(),
        "seed": args.seed + args.round,
        "output": str(path),
        "label": "qc_result 填 T/F/U；T=真人直播或完整回放",
    }
    (out / f"active_round{args.round:02d}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

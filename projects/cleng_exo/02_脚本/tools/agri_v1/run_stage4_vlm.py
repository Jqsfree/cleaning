#!/usr/bin/env python3
"""Stage 4：低置信 VLM 复核（v1 绿field 脚手架）。

仅处理 Stage2/3 低置信样本；二元问题：是否有真人直接接触农作物。
需配置 DashScope/API；默认 dry-run 产出待复核队列。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_SCRIPT))

from core.agri_v1_schema import audit_sample_columns, normalize_agri_v1_frame  # noqa: E402

VLM_PROMPT = (
    "这条视频缩略图/关键帧中，是否有真实人类用手或身体直接接触正在种植环境中的农作物/植物？"
    "只回答 human_present 与 crop_interaction（0/1），不要猜测具体动作类型。"
)


def build_gated_queue(
    stage23_csv: Path,
    *,
    out_dir: Path,
    confidence_col: str = "metadata_score",
    max_score: float = 0.5,
    n_cap: int = 500,
) -> dict:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(stage23_csv, dtype=str, low_memory=False)
    df["video_id"] = df["video_id"].astype(str).str.strip()

    low = df.copy()
    if confidence_col in df.columns:
        scores = pd.to_numeric(df[confidence_col], errors="coerce")
        low = df[scores <= max_score].copy()
    if len(low) > n_cap:
        low = low.sample(n=n_cap, random_state=42)

    queue = normalize_agri_v1_frame(low)
    queue["vlm_prompt"] = VLM_PROMPT
    queue["review_route"] = "stage4_vlm_gated"
    for col in audit_sample_columns():
        if col not in queue.columns:
            queue[col] = ""

    out_csv = out_dir / "stage4_vlm_queue.csv"
    queue.to_csv(out_csv, index=False)
    summary = {
        "stage": "stage4_vlm",
        "input": str(stage23_csv.resolve()),
        "n_pool": len(df),
        "n_queue": len(queue),
        "output": str(out_csv.resolve()),
        "status": "scaffold_dry_run",
        "note": "接入 qc/vision_thumb 多帧 + 新 prompt 后替换 dry-run",
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 4 VLM gated queue scaffold")
    ap.add_argument("stage23_csv", type=Path, help="Stage2/3 合并或 stage1 pass")
    ap.add_argument("-o", "--out-dir", type=Path, required=True)
    ap.add_argument("--confidence-col", default="metadata_score")
    ap.add_argument("--max-score", type=float, default=0.5)
    ap.add_argument("--cap", type=int, default=500)
    args = ap.parse_args()
    build_gated_queue(
        args.stage23_csv,
        out_dir=args.out_dir,
        confidence_col=args.confidence_col,
        max_score=args.max_score,
        n_cap=args.cap,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

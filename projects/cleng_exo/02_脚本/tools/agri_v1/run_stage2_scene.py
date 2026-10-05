#!/usr/bin/env python3
"""Stage 2：场景多分类（v1 绿field 脚手架）。

多帧 CLIP 特征 + scene_type 头；显式负类 no_human / human_no_interaction。
全量跑需在 Phase 0 HALT 解除后执行；默认 --sample 校准用。
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

from categories.exo_agriculture.cascade_clip import (  # noqa: E402
    ClipEncoder,
    load_cfg,
    score_frame,
)
from core.agri_v1_schema import normalize_agri_v1_frame  # noqa: E402

_SCENE_KEYWORD_HINTS: dict[str, str] = {
    "market_kitchen": r"market|kitchen|cook|称重|厨房",
    "game_cg": r"game|simulation|minecraft|游戏",
}

def run_stage2(
    input_csv: Path,
    *,
    out_dir: Path,
    n_sample: int,
    seed: int,
    cache_dir: str,
) -> dict:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    frame = pd.read_csv(input_csv, dtype=str, low_memory=False)
    frame["video_id"] = frame["video_id"].astype(str).str.strip()
    frame = frame.drop_duplicates("video_id")
    if n_sample > 0 and n_sample < len(frame):
        frame = frame.sample(n=n_sample, random_state=seed).reset_index(drop=True)

    encoder = ClipEncoder(cfg["meta"]["model"], cfg["meta"]["pretrained"])
    scored = score_frame(
        frame,
        cfg=cfg,
        encoder=encoder,
        cache_dir=cache_dir,
        batch_size=32,
        thumb_workers=8,
    )
    kw = scored.get("keyword", pd.Series("", index=scored.index)).fillna("").astype(str).str.lower()
    scene = pd.Series("agri_field", index=scored.index, dtype=str)
    for label, pattern in _SCENE_KEYWORD_HINTS.items():
        scene.loc[kw.str.contains(pattern, regex=True)] = label
    scene.loc[~scored["clip_thumb_ok"].fillna(False).astype(bool)] = "other"
    scored["scene_type"] = scene
    scored["human_present"] = scored["scene_type"].map(
        lambda s: "0" if s in {"no_human", "game_cg"} else ""
    )
    scored["crop_interaction"] = scored["scene_type"].map(
        lambda s: "0" if s in {"no_human", "human_no_interaction", "market_kitchen", "game_cg"} else ""
    )

    out_csv = out_dir / f"{input_csv.stem}_stage2_scene.csv"
    scored.to_csv(out_csv, index=False)
    dist = scored["scene_type"].value_counts().to_dict()
    summary = {
        "stage": "stage2_scene",
        "input": str(input_csv.resolve()),
        "n_run": len(scored),
        "scene_type_distribution": dist,
        "halt_check": {
            "market_kitchen_pct": round(
                dist.get("market_kitchen", 0) / max(len(scored), 1) * 100, 2
            ),
            "note": "market_kitchen >15% 连续两轮 → HALT 上报（spec §5/§7）",
        },
        "output": str(out_csv.resolve()),
        "status": "scaffold",
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2 scene_type scaffold")
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--out-dir", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default="qc_thumb_cache/exemplar_sim")
    args = ap.parse_args()
    run_stage2(
        args.input,
        out_dir=args.out_dir,
        n_sample=args.sample,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

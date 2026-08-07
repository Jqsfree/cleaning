#!/usr/bin/env python3
"""应用真人直播视觉 LR，输出 keep/uncertain/highconf_drop。"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.run_manifest import maybe_update_stage  # noqa: E402
from core.visual_filter import assign_actions, build_feature_matrix  # noqa: E402


def _read(path: str) -> pd.DataFrame:
    if Path(path).suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"video_id": str}, low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="应用 human_live 视觉过滤器")
    parser.add_argument("input", help="规则 keep CSV/Parquet")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--neg-sim")
    parser.add_argument("--source", choices=["machine", "human"], default="machine")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=20_000)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with Path(args.model).open("rb") as handle:
        artifact = pickle.load(handle)
    profiles = artifact["profiles"]
    profile = profiles.get(args.source) or profiles[artifact["default_source"]]
    keep_threshold = float(profile["keep_threshold"])
    drop_threshold = float(profile["drop_threshold"])

    pool = _read(args.input)
    pool["video_id"] = pool["video_id"].astype(str).str.strip()
    pool = pool.drop_duplicates("video_id").set_index("video_id")
    index = pd.read_csv(
        Path(args.embeddings) / "index.csv", dtype={"video_id": str},
    )
    ids = index["video_id"].astype(str).str.strip().tolist()
    pool = pool.reindex(ids)
    if pool.index.isna().any():
        raise SystemExit("[ERROR] embedding index 与输入无法对齐")
    vectors = np.load(Path(args.embeddings) / "embeddings.npy", mmap_mode="r")
    if len(vectors) != len(pool):
        raise SystemExit("[ERROR] embedding 行数与输入不一致")

    if args.neg_sim:
        neg = _read(args.neg_sim)
        neg["video_id"] = neg["video_id"].astype(str).str.strip()
        neg = neg.drop_duplicates("video_id").set_index("video_id")
        pool["neg_sim"] = pd.to_numeric(neg.reindex(ids)["neg_sim"], errors="coerce")
    elif "neg_sim" not in pool.columns:
        pool["neg_sim"] = np.nan
    if "sim_score" not in pool.columns:
        pool["sim_score"] = np.nan

    probabilities = np.full(len(pool), np.nan, dtype=np.float64)
    model = artifact["model"]
    for start in range(0, len(pool), args.batch_size):
        end = min(start + args.batch_size, len(pool))
        chunk = pool.iloc[start:end]
        features = build_feature_matrix(
            np.asarray(vectors[start:end], dtype=np.float32),
            pos_sim=pd.to_numeric(chunk["sim_score"], errors="coerce"),
            neg_sim=pd.to_numeric(chunk["neg_sim"], errors="coerce"),
            duration_seconds=pd.to_numeric(
                chunk.get("duration_seconds"), errors="coerce",
            ),
        )
        probabilities[start:end] = model.predict_proba(features)[:, 1]
        print(f"score {end}/{len(pool)}", flush=True)

    scored = pool.reset_index()
    scored["visual_prob"] = probabilities
    scored["ml_action"] = assign_actions(
        probabilities,
        keep_threshold=keep_threshold,
        drop_threshold=drop_threshold,
    )
    scored_path = out / "human_live_visual_scored.parquet"
    scored.to_parquet(scored_path, index=False)
    paths: dict[str, str] = {"scored": str(scored_path)}
    stats: dict[str, dict] = {}
    for action in ("keep_candidate", "uncertain", "highconf_drop"):
        subset = scored[scored["ml_action"].eq(action)].copy()
        path = out / f"{action}.parquet"
        subset.to_parquet(path, index=False)
        hours = pd.to_numeric(
            subset.get("duration_seconds"), errors="coerce",
        ).sum() / 3600
        stats[action] = {"rows": len(subset), "hours": round(float(hours), 1)}
        paths[action] = str(path)

    summary = {
        "input": str(Path(args.input).resolve()),
        "model": str(Path(args.model).resolve()),
        "source": args.source,
        "thresholds": {
            "keep": keep_threshold,
            "drop": drop_threshold,
            "calibration": profile,
        },
        "stats": stats,
        "note": "只有 keep_candidate 进入验收；uncertain 不自动丢",
    }
    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["summary"] = str(summary_path)
    maybe_update_stage(
        out,
        "human_live_visual_filter",
        paths=paths,
        stats={
            "keep_rows": stats["keep_candidate"]["rows"],
            "keep_hours": stats["keep_candidate"]["hours"],
            "uncertain_rows": stats["uncertain"]["rows"],
            "drop_rows": stats["highconf_drop"]["rows"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

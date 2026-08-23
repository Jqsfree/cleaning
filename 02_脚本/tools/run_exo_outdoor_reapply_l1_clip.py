#!/usr/bin/env python3
"""对已有 L1 clip 分数按新阈值重判（无需重跑 CLIP 编码）。"""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).resolve().parent.parent


def load_cfg(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def decide_l1(
    frame: pd.DataFrame,
    *,
    thresholds: dict[str, float],
    require: list[str],
) -> pd.Series:
    ok = frame["clip_thumb_ok"].fillna(False).astype(bool)
    passed = ok.copy()
    for key in require:
        col = f"clip_{key}"
        if col not in frame.columns:
            raise ValueError(f"缺少分数列: {col}")
        passed &= frame[col].astype(float) >= float(thresholds[key])
    out = pd.Series("no_thumb", index=frame.index, dtype=object)
    out.loc[ok & passed] = "clip_pass"
    out.loc[ok & ~passed] = "clip_fail"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="exo_outdoor L1 阈值重判")
    ap.add_argument("input", help="待重判 CSV（如 clip_negative_remain）")
    ap.add_argument(
        "--scored-source",
        required=True,
        help="已有 L1 clip_scored CSV（含 clip_q1/q2/q3）",
    )
    ap.add_argument("-o", "--output", required=True, help="输出目录")
    ap.add_argument(
        "--config",
        default=str(
            _SCRIPT / "categories" / "exo_outdoor" / "rules" / "cascade_outdoor_clip_l1_strict.toml"
        ),
    )
    ap.add_argument("--stem", default="", help="输出文件名前缀")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    require = list(cfg["decision"]["require"])
    thresholds = {k: float(v) for k, v in cfg["thresholds"].items()}
    score_cols = ["video_id", "clip_thumb_ok", *[f"clip_{k}" for k in require]]

    inp = pd.read_csv(args.input, dtype=str, low_memory=False)
    src = pd.read_csv(args.scored_source, dtype=str, low_memory=False)
    if "video_id" not in inp.columns:
        raise ValueError("input 需要 video_id")
    missing = [c for c in score_cols if c not in src.columns]
    if missing:
        raise ValueError(f"scored-source 缺少列: {missing}")

    inp["video_id"] = inp["video_id"].astype(str).str.strip()
    src = src[score_cols].drop_duplicates("video_id", keep="last")
    src["video_id"] = src["video_id"].astype(str).str.strip()

    base = inp.drop(columns=[c for c in inp.columns if c.startswith("clip_")], errors="ignore")
    merged = base.merge(src, on="video_id", how="left")
    miss = merged["clip_thumb_ok"].isna().sum()
    if miss:
        raise ValueError(f"有 {miss} 条在 scored-source 中找不到 L1 分数")

    merged["clip_decision_l1_strict"] = decide_l1(
        merged, thresholds=thresholds, require=require,
    )
    merged["clip_l1_strict_remain"] = merged["clip_decision_l1_strict"] != "clip_fail"

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or Path(args.input).stem
    date_tag = time.strftime("%m%d")

    scored_csv = out_dir / f"{stem}_l1strict_scored_{date_tag}.csv"
    remain_csv = out_dir / f"{stem}_l1strict_remain_{date_tag}.csv"
    drop_csv = out_dir / f"{stem}_l1strict_drop_{date_tag}.csv"
    summary_json = out_dir / f"{stem}_l1strict_summary_{date_tag}.json"

    merged.to_csv(scored_csv, index=False)
    remain = merged[merged["clip_l1_strict_remain"]].copy()
    drop = merged[~merged["clip_l1_strict_remain"]].copy()
    remain.to_csv(remain_csv, index=False)
    drop.to_csv(drop_csv, index=False)

    def wh(df: pd.DataFrame) -> float:
        if "duration_seconds" not in df.columns:
            return 0.0
        return float(df["duration_seconds"].fillna(0).astype(float).sum() / 3600 / 10000)

    summary = {
        "input": args.input,
        "scored_source": args.scored_source,
        "config": args.config,
        "thresholds": thresholds,
        "n_input": len(merged),
        "n_remain": len(remain),
        "n_drop": len(drop),
        "remain_hours_wan": round(wh(remain), 4),
        "drop_hours_wan": round(wh(drop), 4),
        "decision_counts": merged["clip_decision_l1_strict"].value_counts().to_dict(),
        "scored_csv": str(scored_csv),
        "remain_csv": str(remain_csv),
        "drop_csv": str(drop_csv),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

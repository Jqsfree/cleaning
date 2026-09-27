#!/usr/bin/env python3
"""exo_outdoor: clip_remain 二次视觉强负类 veto（L2）。"""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

_SCRIPT = Path(__file__).resolve().parent.parent

from core.exemplar_sim import ClipEncoder, fetch_thumbnails_batch  # noqa: E402
from core.log import log  # noqa: E402
from core.run_manifest import maybe_update_stage  # noqa: E402

_DEFAULT_CFG = (
    _SCRIPT / "categories" / "exo_outdoor" / "rules" / "cascade_outdoor_clip_negative.toml"
)


def load_cfg(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _fmt_quantile(q: float) -> str:
    if q < 0.01:
        return f"q{int(q * 1000):03d}"
    return f"q{int(q * 100):02d}"


def _score_batch(
    frame: pd.DataFrame,
    *,
    encoder: ClipEncoder,
    neg_text: np.ndarray,
    neg_prompts: list[str],
    pos_text: np.ndarray,
    pos_prompts: list[str],
    cache_dir: Path,
    thumb_workers: int,
    batch_size: int,
) -> pd.DataFrame:
    vids = frame["video_id"].astype(str).str.strip().tolist()
    paths = fetch_thumbnails_batch(vids, cache_dir, workers=thumb_workers)

    neg_max = np.full(len(frame), np.nan, dtype=np.float32)
    neg_top1 = np.full(len(frame), "", dtype=object)
    neg_top2 = np.full(len(frame), "", dtype=object)
    pos_max = np.full(len(frame), np.nan, dtype=np.float32)
    pos_top1 = np.full(len(frame), "", dtype=object)
    thumb_ok = np.array([p is not None for p in paths], dtype=bool)

    ok_idx = [i for i, p in enumerate(paths) if p is not None]
    for start in range(0, len(ok_idx), batch_size):
        chunk_idx = ok_idx[start : start + batch_size]
        imgs = [Image.open(paths[i]).convert("RGB") for i in chunk_idx]
        feats = encoder.encode_images(imgs)  # (B, D)

        sims_neg = feats @ neg_text.T  # (B, Nneg)
        sims_pos = feats @ pos_text.T  # (B, Npos)

        top1_neg_idx = sims_neg.argmax(axis=1)
        top1_neg_val = sims_neg.max(axis=1)
        top1_pos_idx = sims_pos.argmax(axis=1)
        top1_pos_val = sims_pos.max(axis=1)

        if sims_neg.shape[1] >= 2:
            part = np.argpartition(sims_neg, -2, axis=1)[:, -2:]
            part_vals = np.take_along_axis(sims_neg, part, axis=1)
            order = np.argsort(part_vals, axis=1)
            second_col = part[np.arange(len(part)), order[:, 0]]
        else:
            second_col = top1_neg_idx

        for j, ridx in enumerate(chunk_idx):
            neg_max[ridx] = float(top1_neg_val[j])
            neg_top1[ridx] = neg_prompts[int(top1_neg_idx[j])]
            neg_top2[ridx] = neg_prompts[int(second_col[j])]
            pos_max[ridx] = float(top1_pos_val[j])
            pos_top1[ridx] = pos_prompts[int(top1_pos_idx[j])]

    out = frame.copy()
    out["clip_thumb_ok"] = thumb_ok
    out["clip_negative_max"] = neg_max
    out["clip_negative_top1"] = neg_top1
    out["clip_negative_top2"] = neg_top2
    out["clip_positive_max"] = pos_max
    out["clip_positive_top1"] = pos_top1
    out["clip_margin"] = out["clip_negative_max"] - out["clip_positive_max"]
    return out


def _decide(df: pd.DataFrame, *, cfg: dict[str, Any]) -> pd.DataFrame:
    dec = cfg["decision"]
    neg_hi = float(dec["neg_hi_threshold"])
    margin_hi = float(dec["margin_threshold"])
    pos_hi = float(dec["pos_hi_threshold"])

    strong_neg = (
        df["clip_thumb_ok"].fillna(False).astype(bool)
        & (df["clip_negative_max"] >= neg_hi)
        & (df["clip_margin"] >= margin_hi)
    )
    strong_pos = (
        df["clip_thumb_ok"].fillna(False).astype(bool)
        & (df["clip_positive_max"] >= pos_hi)
        & (df["clip_positive_max"] > df["clip_negative_max"])
    )
    out = df.copy()
    out["clip_negative_action"] = "keep"
    out.loc[strong_neg, "clip_negative_action"] = "drop"
    out.loc[strong_pos, "clip_negative_action"] = "keep"

    out["clip_negative_rule"] = "default_keep"
    out.loc[strong_neg, "clip_negative_rule"] = "strong_negative_veto"
    out.loc[strong_pos, "clip_negative_rule"] = "strong_positive_guard"
    out.loc[~out["clip_thumb_ok"].fillna(False), "clip_negative_rule"] = "no_thumb_keep"
    return out


def _quantile_samples(
    scored: pd.DataFrame,
    *,
    quantiles: list[float],
    sample_n: int,
    seed: int,
    out_dir: Path,
    stem: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bucket_meta: list[dict[str, Any]] = []
    valid = scored[scored["clip_thumb_ok"].fillna(False)].copy()
    valid = valid.sort_values("clip_negative_max", ascending=False).reset_index(drop=True)

    for q in quantiles:
        take = max(1, int(np.ceil(len(valid) * q)))
        bucket = valid.head(take).copy()
        cutoff = float(bucket["clip_negative_max"].min()) if len(bucket) else None
        n_draw = min(sample_n, len(bucket))
        sampled = bucket.sample(n=n_draw, random_state=seed) if n_draw > 0 else bucket.head(0)
        tag = _fmt_quantile(q)
        sample_path = out_dir / f"{stem}_negative_top_{tag}_sample.csv"
        sampled.to_csv(sample_path, index=False)
        bucket_meta.append(
            {
                "quantile": q,
                "top_rows": len(bucket),
                "sample_rows": len(sampled),
                "cutoff_negative_max": cutoff,
                "sample_csv": str(sample_path),
            }
        )
    return {
        "quantiles": quantiles,
        "sample_n_each": sample_n,
        "buckets": bucket_meta,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="exo_outdoor L2 negative veto clip")
    ap.add_argument("input", help="clip_remain CSV")
    ap.add_argument("-o", "--output", required=True, help="输出目录")
    ap.add_argument("--config", default=str(_DEFAULT_CFG))
    ap.add_argument("--stem", default="", help="输出名前缀；默认取输入 stem")
    ap.add_argument("--cache-dir", default="qc_thumb_cache/exemplar_sim")
    ap.add_argument("--batch-rows", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--thumb-workers", type=int, default=16)
    ap.add_argument("--sample-n", type=int, default=150, help="每个分位桶抽样数")
    ap.add_argument("--quantiles", default="0.001,0.005,0.01,0.02,0.05")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.perf_counter()
    cfg_path = Path(args.config)
    cfg = load_cfg(cfg_path)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input, dtype=str, low_memory=False)
    if "video_id" not in frame.columns:
        raise ValueError("需要 video_id")
    frame["video_id"] = frame["video_id"].astype(str).str.strip()
    frame = frame.drop_duplicates("video_id").copy()
    n_total = len(frame)

    stem = args.stem or Path(args.input).stem
    if "_clip_remain" in stem:
        stem = stem.split("_clip_remain", 1)[0]
    date_tag = time.strftime("%m%d")

    scored_csv = out_dir / f"{stem}_clip_negative_scored_{date_tag}.csv"
    drop_csv = out_dir / f"{stem}_clip_negative_drop_{date_tag}.csv"
    remain_csv = out_dir / f"{stem}_clip_negative_remain_{date_tag}.csv"
    summary_json = out_dir / f"{stem}_clip_negative_summary.json"

    neg_prompts = list(cfg["negative"]["prompts"])
    pos_prompts = list(cfg["positive"]["prompts"])
    encoder = ClipEncoder(cfg["meta"]["model"], cfg["meta"]["pretrained"])
    neg_text = encoder.encode_text(neg_prompts)
    pos_text = encoder.encode_text(pos_prompts)

    parts: list[pd.DataFrame] = []
    for start in range(0, n_total, max(args.batch_rows, 1)):
        chunk = frame.iloc[start : start + max(args.batch_rows, 1)].copy()
        log(f"[L2 negative] batch {start + 1}-{start + len(chunk)}/{n_total}")
        scored = _score_batch(
            chunk,
            encoder=encoder,
            neg_text=neg_text,
            neg_prompts=neg_prompts,
            pos_text=pos_text,
            pos_prompts=pos_prompts,
            cache_dir=Path(args.cache_dir),
            thumb_workers=args.thumb_workers,
            batch_size=args.batch_size,
        )
        parts.append(scored)

    all_scored = pd.concat(parts, ignore_index=True) if parts else frame.head(0).copy()
    all_scored = _decide(all_scored, cfg=cfg)
    all_scored.to_csv(scored_csv, index=False)

    dropped = all_scored[all_scored["clip_negative_action"] == "drop"].copy()
    remain = all_scored[all_scored["clip_negative_action"] != "drop"].copy()
    dropped.to_csv(drop_csv, index=False)
    remain.to_csv(remain_csv, index=False)

    quantiles = [float(x) for x in args.quantiles.split(",") if x.strip()]
    sample_meta = _quantile_samples(
        all_scored,
        quantiles=quantiles,
        sample_n=args.sample_n,
        seed=args.seed,
        out_dir=out_dir / "samples",
        stem=stem,
    )

    summary = {
        "layer": cfg["meta"].get("layer", "outdoor_clip_negative_veto"),
        "cfg": str(cfg_path),
        "input": args.input,
        "n_input": n_total,
        "n_drop": int((all_scored["clip_negative_action"] == "drop").sum()),
        "n_remain": int((all_scored["clip_negative_action"] != "drop").sum()),
        "n_thumb_ok": int(all_scored["clip_thumb_ok"].fillna(False).astype(bool).sum()),
        "n_no_thumb": int((~all_scored["clip_thumb_ok"].fillna(False).astype(bool)).sum()),
        "decision_thresholds": cfg["decision"],
        "scored_csv": str(scored_csv),
        "drop_csv": str(drop_csv),
        "remain_csv": str(remain_csv),
        "sampling": sample_meta,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "notes": [
            "L2 只做强负类 veto；默认 keep",
            "no_thumb 默认 keep",
            "drop_precision 需人工抽样复核",
        ],
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    maybe_update_stage(
        out_dir,
        "harvest_clip_negative",
        paths={
            "scored": str(scored_csv),
            "drop": str(drop_csv),
            "remain": str(remain_csv),
            "summary": str(summary_json),
        },
        stats={
            "n": n_total,
            "n_drop": summary["n_drop"],
            "n_remain": summary["n_remain"],
            "n_no_thumb": summary["n_no_thumb"],
        },
    )
    log(
        f"[L2 negative] n={n_total:,} drop={summary['n_drop']:,} remain={summary['n_remain']:,} "
        f"no_thumb={summary['n_no_thumb']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

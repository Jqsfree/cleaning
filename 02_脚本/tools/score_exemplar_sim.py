#!/usr/bin/env python3
"""
tools/score_exemplar_sim.py — 候选 CSV 缩略图 vs 样例视频原型（支持分批+断点）

用法:
  # 全量分批（每批 5000，断点续跑）
  02_脚本/tools/score_exemplar_sim.py quality.csv \\
    --bank data/assets/exemplars/yt_live_scene/ \\
    -o $BATCH/06_tools/ \\
    --batch-rows 5000 --high 0.5537 --mid 0.4908

  # 调试
  02_脚本/tools/score_exemplar_sim.py quality.csv --bank … -o out/ -n 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.exemplar_sim import (  # noqa: E402
    ClipEncoder,
    assign_bands_by_quantile,
    load_bank,
    score_candidates,
)
from core.run_manifest import maybe_update_stage  # noqa: E402


def _read_table(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def _apply_band(scored: pd.DataFrame, high_t: float, mid_t: float) -> pd.DataFrame:
    def _band(s: float) -> str:
        if pd.isna(s):
            return "error"
        if s >= high_t:
            return "high"
        if s >= mid_t:
            return "mid"
        return "low"

    out = scored.copy()
    out["band"] = out["sim_score"].map(_band)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="样例原型相似度打分（可分批）")
    p.add_argument("input", help="候选 CSV/Parquet（需 video_id）")
    p.add_argument("--bank", required=True, help="exemplar bank 目录")
    p.add_argument("-o", "--output-dir", required=True, help="输出目录")
    p.add_argument("-n", "--limit", type=int, default=None, help="仅前 N 条（调试）")
    p.add_argument("--high", type=float, default=None, help="high 阈值（跨批请固定）")
    p.add_argument("--mid", type=float, default=None, help="mid 阈值（跨批请固定）")
    p.add_argument("--cache-dir", default="qc_thumb_cache/exemplar_sim")
    p.add_argument("--batch-size", type=int, default=64, help="GPU encode batch")
    p.add_argument(
        "--batch-rows", type=int, default=5000,
        help="每批处理行数并落盘 checkpoint（默认 5000）",
    )
    p.add_argument("--thumb-workers", type=int, default=16)
    p.add_argument(
        "--overwrite", action="store_true",
        help="忽略已有 checkpoint 重跑",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    proto, manifest, meta = load_bank(args.bank)
    exemplar_ids = manifest["exemplar_id"].astype(str).tolist()
    print(f"bank={args.bank}  exemplars={len(exemplar_ids)}  model={meta.get('model')}")

    df = _read_table(args.input)
    if "video_id" not in df.columns:
        print("[ERROR] 需要 video_id 列")
        sys.exit(2)
    df = df.copy()
    df["video_id"] = df["video_id"].astype(str).str.strip()
    if args.limit:
        df = df.head(args.limit).copy()
    all_ids = df["video_id"].tolist()
    print(f"candidates={len(all_ids)}  batch_rows={args.batch_rows}")

    stem = Path(args.input).stem
    date_tag = time.strftime("%m%d")
    ckpt_path = out_dir / f"{stem}_exemplar_sim.ckpt.csv"
    out_csv = out_dir / f"{stem}_exemplar_sim_{date_tag}.csv"
    keep_csv = out_dir / f"{stem}_exemplar_keep_highmid_{date_tag}.csv"
    sum_path = out_dir / f"{stem}_exemplar_sim_{date_tag}_summary.json"
    progress_path = out_dir / f"{stem}_exemplar_sim_progress.json"

    done: set[str] = set()
    parts: list[pd.DataFrame] = []
    if ckpt_path.is_file() and not args.overwrite:
        prev = pd.read_csv(ckpt_path, dtype={"video_id": str})
        parts.append(prev)
        done = set(prev["video_id"].astype(str).str.strip())
        print(f"[续跑] checkpoint {ckpt_path}  already={len(done)}")

    pending_ids = [vid for vid in all_ids if vid not in done]
    print(f"pending={len(pending_ids)}")

    encoder = ClipEncoder(meta.get("model", "ViT-B-32"), meta.get("pretrained", "openai"))

    high_t, mid_t = args.high, args.mid
    # 若未指定阈值：用已有 ckpt 或首批算分位，并写死到 progress 供后续批复用
    if high_t is None or mid_t is None:
        if progress_path.is_file():
            prog = json.loads(progress_path.read_text(encoding="utf-8"))
            high_t = high_t if high_t is not None else prog.get("high_threshold")
            mid_t = mid_t if mid_t is not None else prog.get("mid_threshold")
        if (high_t is None or mid_t is None) and parts:
            qh, qm = assign_bands_by_quantile(parts[0]["sim_score"])
            high_t = high_t if high_t is not None else qh
            mid_t = mid_t if mid_t is not None else qm

    batch_i = 0
    t0 = time.perf_counter()
    for start in range(0, len(pending_ids), args.batch_rows):
        batch_i += 1
        chunk = pending_ids[start : start + args.batch_rows]
        print(
            f"\n=== batch {batch_i}  rows={len(chunk)}  "
            f"global {start + len(done) + 1}-{start + len(done) + len(chunk)}/{len(all_ids)} ===",
            flush=True,
        )
        # 首批且无阈值：先打分再定阈值
        use_high = high_t if high_t is not None else 1.0
        use_mid = mid_t if mid_t is not None else 1.0
        scored = score_candidates(
            chunk,
            proto,
            exemplar_ids,
            encoder,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            high=use_high,
            mid=use_mid,
            thumb_workers=args.thumb_workers,
        )
        if high_t is None or mid_t is None:
            qh, qm = assign_bands_by_quantile(scored["sim_score"])
            high_t = high_t if high_t is not None else qh
            mid_t = mid_t if mid_t is not None else qm
            print(f"thresholds locked  high>={high_t:.4f}  mid>={mid_t:.4f}")
        scored = _apply_band(scored, float(high_t), float(mid_t))
        parts.append(scored)

        # 落盘 checkpoint（全量已完成部分）
        all_scored = pd.concat(parts, ignore_index=True)
        all_scored = all_scored.drop_duplicates(subset=["video_id"], keep="last")
        tmp = ckpt_path.with_suffix(".tmp.csv")
        all_scored.to_csv(tmp, index=False)
        tmp.replace(ckpt_path)

        elapsed = time.perf_counter() - t0
        rate = len(all_scored) / max(elapsed, 1e-6)
        rem = (len(all_ids) - len(all_scored)) / max(rate, 1e-6)
        progress = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "done": len(all_scored),
            "total": len(all_ids),
            "batch_rows": args.batch_rows,
            "high_threshold": high_t,
            "mid_threshold": mid_t,
            "rows_per_sec": round(rate, 2),
            "eta_sec": int(rem),
            "band_counts": all_scored["band"].value_counts().to_dict(),
        }
        progress_path.write_text(
            json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        print(
            f"  ckpt done={len(all_scored)}/{len(all_ids)}  "
            f"{rate:.1f} rows/s  eta≈{rem/3600:.1f}h  bands={progress['band_counts']}",
            flush=True,
        )

    all_scored = pd.concat(parts, ignore_index=True)
    all_scored = all_scored.drop_duplicates(subset=["video_id"], keep="last")
    all_scored = _apply_band(all_scored, float(high_t), float(mid_t))
    all_scored.to_csv(out_csv, index=False)

    keep = all_scored[all_scored["band"].isin(["high", "mid"])][
        ["video_id", "sim_score", "band", "nearest_exemplar_id"]
    ]
    base = df[df["video_id"].isin(set(keep["video_id"]))].copy()
    base = base.merge(keep, on="video_id", how="inner")
    base.to_csv(keep_csv, index=False)

    summary = {
        "input": str(args.input),
        "bank": str(args.bank),
        "n": len(all_scored),
        "high_threshold": high_t,
        "mid_threshold": mid_t,
        "batch_rows": args.batch_rows,
        "band_counts": all_scored["band"].value_counts().to_dict(),
        "thumb_ok": int(all_scored["thumb_ok"].sum()),
        "sim_mean": float(all_scored["sim_score"].dropna().mean())
        if all_scored["sim_score"].notna().any() else None,
        "output": str(out_csv),
        "keep_highmid": str(keep_csv),
        "checkpoint": str(ckpt_path),
        "note": "分批打分；liveBroadcastContent 非硬门禁；band 默认宁留 mid",
    }
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary["band_counts"], ensure_ascii=False))
    print(f"wrote {out_csv}")
    print(f"keep high+mid → {keep_csv}  (n={len(base)})")

    maybe_update_stage(
        out_dir,
        "exemplar_sim",
        paths={
            "scored": str(out_csv),
            "keep": str(keep_csv),
            "summary": str(sum_path),
            "checkpoint": str(ckpt_path),
        },
        stats={
            "n": len(all_scored),
            "n_keep": len(base),
            "high_threshold": high_t,
            "mid_threshold": mid_t,
            **{f"band_{k}": int(v) for k, v in summary["band_counts"].items()},
        },
    )


if __name__ == "__main__":
    main()

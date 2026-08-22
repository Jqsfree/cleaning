#!/usr/bin/env python3
"""对 MiniLM pass 池打 parent_child 缩略图 CLIP+LR 分数，并按小时排序切片。

用法:
  # 全量推理（断点）
  PYTHONPATH=02_脚本 python 02_脚本/tools/apply_parent_child_thumb_clip.py \\
    data/runs/parent_child/machine_0818_lt50/06_tools/minilm_v3/亲子互动_<50%_run02_minilm_v3_pass_0820.csv \\
    -o data/runs/parent_child/machine_0818_lt50/06_tools/clip_lr_v1/ \\
    --model models/parent_child_thumb_clip_lr.pkl

  # 仅排序切片（已有 pass_scored.csv）
  PYTHONPATH=02_脚本 python 02_脚本/tools/apply_parent_child_thumb_clip.py --rank \\
    --scored data/runs/parent_child/machine_0818_lt50/06_tools/clip_lr_v1/pass_scored.csv \\
    -o data/runs/parent_child/machine_0818_lt50/06_tools/clip_lr_v1/
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from core.exemplar_sim import ClipEncoder, fetch_thumbnails_batch  # noqa: E402

DEFAULT_MODEL = _REPO_ROOT / "models/parent_child_thumb_clip_lr.pkl"
KEEP_COLS = (
    "video_id", "title", "channel", "duration_seconds",
    "ml_score", "ml_action", "url", "keyword",
)


def load_pipeline(model_path: Path):
    with model_path.open("rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"]
    return obj


def read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def load_done_ids(scored_path: Path) -> set[str]:
    if not scored_path.exists():
        return set()
    done: set[str] = set()
    for chunk in pd.read_csv(scored_path, usecols=["video_id"], chunksize=200_000):
        done.update(chunk["video_id"].astype(str).tolist())
    return done


def load_all_done_ids(out_dir: Path) -> set[str]:
    """合并 pass_scored.csv 与 pass_scored_shard*.csv 中已打分 video_id。"""
    done: set[str] = set()
    for path in sorted(out_dir.glob("pass_scored*.csv")):
        done.update(load_done_ids(path))
    return done


def score_chunk(
    df: pd.DataFrame,
    *,
    encoder: ClipEncoder,
    pipe,
    cache_dir: Path,
    thumb_workers: int,
    batch_size: int,
) -> pd.DataFrame:
    video_ids = df["video_id"].astype(str).tolist()
    paths = fetch_thumbnails_batch(video_ids, cache_dir, workers=thumb_workers)
    scores = np.full(len(video_ids), np.nan, dtype=np.float64)
    status = np.full(len(video_ids), "no_thumb", dtype=object)

    ok_idx = [
        i for i, p in enumerate(paths)
        if p is not None and Path(p).is_file() and Path(p).stat().st_size >= 1500
    ]
    for start in range(0, len(ok_idx), batch_size):
        chunk_i = ok_idx[start : start + batch_size]
        imgs_paths = [Path(paths[i]) for i in chunk_i]
        feats = encoder.encode_paths(imgs_paths, batch_size=batch_size)
        proba = pipe.predict_proba(feats)[:, 1]
        for j, ii in enumerate(chunk_i):
            scores[ii] = float(proba[j])
            status[ii] = "ok"

    out = df.copy()
    out["clip_score"] = scores
    out["clip_status"] = status
    return out


def prefetch_chunk(
    video_ids: list[str],
    *,
    cache_dir: Path,
    thumb_workers: int,
) -> dict:
    paths = fetch_thumbnails_batch(video_ids, cache_dir, workers=thumb_workers)
    ok = sum(
        1 for p in paths
        if p is not None and Path(p).is_file() and Path(p).stat().st_size >= 1500
    )
    return {"n": len(video_ids), "ok": ok}


def apply_full(
    input_csv: Path,
    *,
    out_dir: Path,
    model_path: Path,
    cache_dir: Path,
    chunksize: int,
    thumb_workers: int,
    batch_size: int,
    shard_index: int = 0,
    num_shards: int = 1,
    scored_name: str | None = None,
    prefetch_only: bool = False,
    device: str | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if scored_name:
        scored_path = out_dir / scored_name
    elif num_shards > 1:
        scored_path = out_dir / f"pass_scored_shard{shard_index}.csv"
    else:
        scored_path = out_dir / "pass_scored.csv"
    log_path = out_dir / (
        f"prefetch_shard{shard_index}.log" if prefetch_only and num_shards > 1
        else f"apply_shard{shard_index}.log" if num_shards > 1
        else ("prefetch.log" if prefetch_only else "apply.log")
    )

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    pipe = None
    encoder = None
    if not prefetch_only:
        pipe = load_pipeline(model_path)
        encoder = ClipEncoder(device=device)
    done = load_all_done_ids(out_dir) if not prefetch_only and num_shards == 1 else (
        load_done_ids(scored_path) if not prefetch_only else set()
    )
    log(
        f"input={input_csv} already_scored={len(done):,} "
        f"shard={shard_index}/{num_shards}"
        + (" PREFETCH_ONLY" if prefetch_only else "")
    )

    write_header = not scored_path.exists()
    n_new = 0
    n_seen = 0
    n_prefetch_ok = 0
    t0 = time.perf_counter()

    for i, chunk in enumerate(pd.read_csv(input_csv, chunksize=chunksize, low_memory=False), 1):
        if num_shards > 1 and (i - 1) % num_shards != shard_index:
            continue
        chunk["video_id"] = chunk["video_id"].astype(str)
        if prefetch_only:
            stats = prefetch_chunk(
                chunk["video_id"].tolist(),
                cache_dir=cache_dir,
                thumb_workers=thumb_workers,
            )
            n_prefetch_ok += stats["ok"]
            n_seen += stats["n"]
            if i % 5 == 0 or stats["n"] > 0:
                log(
                    f"prefetch chunk {i}: n={stats['n']:,} ok={stats['ok']:,} "
                    f"cumulative_ok≈{n_prefetch_ok:,} seen={n_seen:,}"
                )
            continue

        pending = chunk[~chunk["video_id"].isin(done)].copy()
        n_seen += len(chunk)
        if pending.empty:
            if i % 20 == 0:
                log(f"chunk {i}: skip all done (seen={n_seen:,})")
            continue
        scored = score_chunk(
            pending,
            encoder=encoder,
            pipe=pipe,
            cache_dir=cache_dir,
            thumb_workers=thumb_workers,
            batch_size=batch_size,
        )
        # keep useful columns + clip
        cols = [c for c in KEEP_COLS if c in scored.columns] + ["clip_score", "clip_status"]
        scored[cols].to_csv(
            scored_path, mode="w" if write_header else "a",
            header=write_header, index=False,
        )
        write_header = False
        done.update(scored["video_id"].astype(str).tolist())
        n_new += len(scored)
        log(
            f"chunk {i}: +{len(scored):,}  cumulative_new={n_new:,} "
            f"total_done≈{len(done):,}  ok={(scored.clip_status=='ok').sum():,}"
        )

    elapsed = time.perf_counter() - t0
    summary = {
        "input": str(input_csv.resolve()),
        "scored_csv": str(scored_path.resolve()) if not prefetch_only else None,
        "n_new": n_new,
        "n_done": len(done) if not prefetch_only else 0,
        "n_prefetch_seen": n_seen if prefetch_only else None,
        "n_prefetch_ok": n_prefetch_ok if prefetch_only else None,
        "elapsed_sec": round(elapsed, 1),
        "model": str(model_path.resolve()) if not prefetch_only else None,
        "cache_dir": str(cache_dir.resolve()),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "prefetch_only": prefetch_only,
    }
    if prefetch_only:
        summary_name = (
            f"prefetch_summary_shard{shard_index}.json"
            if num_shards > 1
            else "prefetch_summary.json"
        )
    else:
        summary_name = (
            f"apply_summary_shard{shard_index}.json"
            if num_shards > 1
            else "apply_summary.json"
        )
    (out_dir / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    log(f"APPLY_DONE {json.dumps(summary, ensure_ascii=False)}")
    return summary


def merge_all_scored_files(out_dir: Path, *, dest_name: str = "pass_scored.csv") -> Path:
    """合并 pass_scored.csv 与 pass_scored_shard*.csv → pass_scored.csv（video_id 去重，保留 clip_score 最高）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / dest_name
    parts = sorted(set(out_dir.glob("pass_scored*.csv")) - {dest})
    if not parts and dest.exists():
        return dest
    frames: list[pd.DataFrame] = []
    if dest.exists():
        frames.append(pd.read_csv(dest, low_memory=False))
    for part in parts:
        frames.append(pd.read_csv(part, low_memory=False))
    if not frames:
        raise SystemExit("无 scored 文件可合并")
    df = pd.concat(frames, ignore_index=True)
    df["video_id"] = df["video_id"].astype(str)
    df["clip_score"] = pd.to_numeric(df.get("clip_score"), errors="coerce")
    df = df.sort_values("clip_score", ascending=False, na_position="last")
    df = df.drop_duplicates(subset=["video_id"], keep="first")
    tmp = dest.with_suffix(".merge_tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(dest)
    return dest


def merge_shards(out_dir: Path, *, num_shards: int, dest_name: str = "pass_scored.csv") -> dict:
    """合并 pass_scored_shard*.csv → pass_scored.csv（按 video_id 去重）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / dest_name
    parts = sorted(out_dir.glob("pass_scored_shard*.csv"))
    if num_shards > 0 and len(parts) < num_shards:
        raise SystemExit(
            f"分片未齐: 期望 {num_shards} 个 pass_scored_shard*.csv，当前 {len(parts)}"
        )
    if not parts:
        raise SystemExit("无 pass_scored_shard*.csv 可合并")

    seen: set[str] = set()
    write_header = True
    n_rows = 0
    for part in parts:
        for chunk in pd.read_csv(part, chunksize=200_000, low_memory=False):
            chunk["video_id"] = chunk["video_id"].astype(str)
            chunk = chunk[~chunk["video_id"].isin(seen)]
            if chunk.empty:
                continue
            seen.update(chunk["video_id"].tolist())
            chunk.to_csv(dest, mode="w" if write_header else "a", header=write_header, index=False)
            write_header = False
            n_rows += len(chunk)

    summary = {
        "merged_from": [str(p.resolve()) for p in parts],
        "dest": str(dest.resolve()),
        "n_rows": n_rows,
        "n_unique_video_ids": len(seen),
    }
    (out_dir / "merge_shards_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def rank_export(
    scored_csv: Path,
    *,
    out_dir: Path,
    hour_budgets: list[float] | None = None,
    top_pcts: tuple[int, ...] = (5, 10, 20, 30),
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if hour_budgets is None:
        hour_budgets = [50_000.0, 100_000.0, 150_000.0, 200_000.0]

    usecols = None
    df = pd.read_csv(scored_csv, low_memory=False)
    df["clip_score"] = pd.to_numeric(df["clip_score"], errors="coerce")
    df["duration_seconds"] = pd.to_numeric(df.get("duration_seconds"), errors="coerce").fillna(0)
    ok = df[df["clip_status"].astype(str).eq("ok") & df["clip_score"].notna()].copy()
    ok = ok.sort_values("clip_score", ascending=False).reset_index(drop=True)
    ok["hours"] = ok["duration_seconds"] / 3600.0
    ok["cum_hours"] = ok["hours"].cumsum()
    n = len(ok)
    total_h = float(ok["hours"].sum())

    ranked_path = out_dir / "pass_scored_ranked.csv"
    ok.to_csv(ranked_path, index=False)

    pct_stats = {}
    for pct in top_pcts:
        k = max(1, int(round(n * pct / 100.0)))
        sub = ok.iloc[:k]
        out_p = out_dir / f"top_{pct}pct.csv"
        sub.to_csv(out_p, index=False)
        pct_stats[f"top_{pct}pct"] = {
            "n": int(len(sub)),
            "hours": round(float(sub["hours"].sum()), 2),
            "hours_share": round(float(sub["hours"].sum()) / max(total_h, 1e-9), 4),
            "min_clip_score": round(float(sub["clip_score"].min()), 4),
            "path": str(out_p),
        }

    hour_stats = {}
    for budget in hour_budgets:
        sub = ok[ok["cum_hours"] <= budget]
        if sub.empty:
            sub = ok.iloc[:1]
        # include first row that crosses budget
        if len(sub) < len(ok) and float(sub["cum_hours"].iloc[-1]) < budget:
            nxt = ok.iloc[len(sub) : len(sub) + 1]
            sub = pd.concat([sub, nxt], ignore_index=True)
        tag = int(budget) if budget >= 1000 else budget
        out_h = out_dir / f"top_hours_{tag}.csv"
        sub.to_csv(out_h, index=False)
        hour_stats[f"top_hours_{tag}"] = {
            "n": int(len(sub)),
            "hours": round(float(sub["hours"].sum()), 2),
            "min_clip_score": round(float(sub["clip_score"].min()), 4),
            "path": str(out_h),
        }

    summary = {
        "scored_csv": str(scored_csv.resolve()),
        "ranked_csv": str(ranked_path.resolve()),
        "n_scored": int(len(df)),
        "n_ok": n,
        "n_no_thumb": int((df["clip_status"].astype(str) != "ok").sum()),
        "total_hours_ok": round(total_h, 2),
        "top_pct": pct_stats,
        "top_hours": hour_stats,
        "score_quantiles": {
            str(q): round(float(ok["clip_score"].quantile(q)), 4)
            for q in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="parent_child thumb CLIP+LR apply/rank")
    ap.add_argument("input", nargs="?", help="MiniLM pass CSV")
    ap.add_argument("-o", "--out-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--chunksize", type=int, default=5000)
    ap.add_argument("--thumb-workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--rank", action="store_true", help="仅对已有 scored 排序切片")
    ap.add_argument("--merge-shards", action="store_true", help="合并 pass_scored_shard*.csv")
    ap.add_argument("--merge-all", action="store_true", help="合并 pass_scored*.csv 到 pass_scored.csv")
    ap.add_argument("--num-shards", type=int, default=1, help="并行分片数（chunk 轮询分配）")
    ap.add_argument("--shard-index", type=int, default=0, help="当前分片索引 [0, num-shards)")
    ap.add_argument("--scored-name", type=str, default=None, help="自定义 scored 输出文件名")
    ap.add_argument("--prefetch-only", action="store_true", help="仅并行下载缩略图到 cache")
    ap.add_argument("--device", type=str, default=None, help="ClipEncoder device，如 cuda / cpu")
    ap.add_argument("--scored", type=Path, default=None, help="rank 用的 scored CSV")
    args = ap.parse_args()

    cache = args.cache_dir or (args.out_dir / "thumb_cache")

    if args.merge_shards:
        merge_shards(args.out_dir, num_shards=args.num_shards)
        return
    if args.merge_all:
        merge_all_scored_files(args.out_dir)
        return

    if args.rank:
        merge_all_scored_files(args.out_dir)
        scored = args.scored or (args.out_dir / "pass_scored.csv")
        if not scored.exists():
            raise SystemExit(f"缺少 scored: {scored}")
        rank_export(scored, out_dir=args.out_dir)
        return

    if args.num_shards > 1 and not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"shard-index 须在 [0, {args.num_shards})")

    if not args.input:
        raise SystemExit("需要 input CSV，或使用 --rank / --merge-shards")
    if not args.prefetch_only and not Path(args.model).exists():
        raise SystemExit(f"缺少模型: {args.model}；先跑 experiments/parent_child_thumb_clip_lr.py --calibrate")

    apply_full(
        Path(args.input),
        out_dir=args.out_dir,
        model_path=Path(args.model),
        cache_dir=cache,
        chunksize=args.chunksize,
        thumb_workers=args.thumb_workers,
        batch_size=args.batch_size,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        scored_name=args.scored_name,
        prefetch_only=args.prefetch_only,
        device=args.device,
    )
    if args.num_shards == 1 and not args.prefetch_only:
        merge_all_scored_files(args.out_dir)
        ranked = args.out_dir / (args.scored_name or "pass_scored.csv")
        if ranked.exists():
            rank_export(ranked, out_dir=args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""exo_entertainment clean keep · MiniLM(title+channel) 编码 + MiniBatchKMeans 聚类。

产物默认落 work/exo_entertainment_text_cluster_MMDD/（scratch，非交付 KPI）。
丢簇须等金标后再跑 exo_entertainment_cluster_gold_filter.py。

用法:
  .venv/bin/python3 experiments/exo_entertainment_text_cluster.py \\
    data/runs/exo_entertainment/machine_0818/05_clean/run01_v01/娱乐表演_merged_0818_clean_0917.csv \\
    -o work/exo_entertainment_text_cluster_0918 --device mps --batch-size 512 --k 60 --resume
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "experiments"))

# 复用 cook 同款 MiniLMEncoder（同一模型；娱乐尚无自训 clf）
import exo_cook_text_classifier as ck  # noqa: E402

DEFAULT_INPUT = (
    PROJECT
    / "data/runs/exo_entertainment/machine_0818/05_clean/run01_v01"
    / "娱乐表演_merged_0818_clean_0917.csv"
)
DIM = 384
TOKEN_RE = re.compile(r"[A-Za-z\u4e00-\u9fff\u0900-\u097F\u0C00-\u0C7F\u0B80-\u0BFF]{2,}")
STOP = {
    "the", "and", "with", "for", "from", "this", "that", "video", "live",
    "show", "dance", "performance", "music", "song", "official",
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_texts_df(df: pd.DataFrame) -> list[str]:
    title = df["title"].fillna("").astype(str) if "title" in df.columns else ""
    channel = df["channel"].fillna("").astype(str) if "channel" in df.columns else ""
    return (
        title
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .tolist()
    )


def encode_file(
    input_path: Path,
    out_dir: Path,
    *,
    device: str | None,
    batch_size: int,
    chunksize: int,
    resume: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings.f16.npy"
    index_path = out_dir / "index.parquet"
    progress_path = out_dir / "encode_progress.json"

    n_total = 0
    for chunk in pd.read_csv(input_path, dtype=str, chunksize=chunksize, usecols=["video_id"]):
        n_total += len(chunk)
    _log(f"input rows: {n_total:,}")

    start_row = 0
    # 断点续跑：只要有 progress + embeddings memmap 即可（合并前只有 index_part_*）
    if resume and progress_path.is_file() and emb_path.is_file():
        prog = json.loads(progress_path.read_text())
        start_row = int(prog.get("rows_done", 0))
        _log(f"resume from row {start_row:,}")
        if start_row >= n_total and index_path.is_file():
            _log("encode already complete")
            return {"n_rows": n_total, "emb_path": str(emb_path), "index_path": str(index_path)}
        if start_row >= n_total and not index_path.is_file():
            _log("embeddings complete — merging index parts only")
            start_row = n_total  # skip encode loop; fall through to merge

    if start_row == 0 or not emb_path.is_file():
        emb = np.lib.format.open_memmap(
            emb_path, mode="w+", dtype=np.float16, shape=(n_total, DIM)
        )
        del emb
        for p in out_dir.glob("index_part_*.parquet"):
            p.unlink()
    elif start_row > 0 and start_row < n_total:
        # keep existing index_part_* for rows already done; drop parts that overlap resume window
        # parts are written one-per-chunk in order — safest: delete parts whose max row >= start_row
        for p in sorted(out_dir.glob("index_part_*.parquet")):
            try:
                mx = int(pd.read_parquet(p, columns=["row"])["row"].max())
            except Exception:
                p.unlink(missing_ok=True)
                continue
            if mx >= start_row:
                p.unlink(missing_ok=True)

    emb = np.lib.format.open_memmap(emb_path, mode="r+", dtype=np.float16, shape=(n_total, DIM))
    encoder = ck.MiniLMEncoder(batch_size=batch_size, device=device)
    encoder.fit([])

    header = pd.read_csv(input_path, nrows=0, dtype=str).columns.tolist()
    usecols = [
        c
        for c in ("video_id", "title", "channel", "duration_seconds", "ml_score", "keyword")
        if c in header
    ]
    if "video_id" not in usecols or "title" not in usecols:
        raise ValueError("input needs video_id + title")

    row_cursor = 0
    existing_parts = sorted(out_dir.glob("index_part_*.parquet"))
    part_i = 0
    if existing_parts:
        # continue numbering after highest existing part id
        try:
            part_i = max(int(p.stem.split("_")[-1]) for p in existing_parts) + 1
        except ValueError:
            part_i = len(existing_parts)
    t0 = time.perf_counter()
    if start_row < n_total:
        for chunk in pd.read_csv(input_path, dtype=str, chunksize=chunksize, usecols=usecols):
            n = len(chunk)
            end = row_cursor + n
            if end <= start_row:
                row_cursor = end
                continue
            local_start = max(0, start_row - row_cursor)
            sub = chunk.iloc[local_start:].copy()
            abs_start = row_cursor + local_start
            texts = build_texts_df(sub)
            vecs = encoder.transform(texts)
            if vecs.shape[1] != DIM:
                raise RuntimeError(f"unexpected dim {vecs.shape[1]} != {DIM}")
            emb[abs_start : abs_start + len(sub)] = vecs.astype(np.float16)
            emb.flush()

            idx = sub.copy()
            idx.insert(0, "row", np.arange(abs_start, abs_start + len(sub), dtype=np.int64))
            part_path = out_dir / f"index_part_{part_i:04d}.parquet"
            idx.to_parquet(part_path, index=False)
            part_i += 1

            rows_done = abs_start + len(sub)
            progress_path.write_text(
                json.dumps(
                    {
                        "rows_done": rows_done,
                        "n_total": n_total,
                        "elapsed_sec": round(time.perf_counter() - t0, 1),
                    },
                    indent=2,
                )
                + "\n"
            )
            _log(
                f"encoded {rows_done:,}/{n_total:,} "
                f"({100 * rows_done / n_total:.1f}%) "
                f"elapsed {time.perf_counter() - t0:.0f}s"
            )
            row_cursor = end

    parts = sorted(out_dir.glob("index_part_*.parquet"))
    if not parts:
        raise RuntimeError("no index parts written")
    index_df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    index_df = index_df.sort_values("row").drop_duplicates("row", keep="last").reset_index(drop=True)
    if len(index_df) != n_total:
        _log(f"WARN index rows {len(index_df)} != n_total {n_total}")
    index_df.to_parquet(index_path, index=False)
    for p in parts:
        p.unlink(missing_ok=True)

    progress_path.write_text(
        json.dumps({"rows_done": n_total, "n_total": n_total, "status": "done"}, indent=2) + "\n"
    )
    _log(f"encode done → {emb_path} + {index_path}")
    return {"n_rows": n_total, "emb_path": str(emb_path), "index_path": str(index_path)}


def cluster_and_report(
    out_dir: Path,
    *,
    k: int,
    seed: int,
    samples_per_cluster: int,
) -> dict:
    emb_path = out_dir / "embeddings.f16.npy"
    index_path = out_dir / "index.parquet"
    emb = np.load(emb_path, mmap_mode="r")
    n, dim = emb.shape
    _log(f"clustering K={k} on {n:,} x {dim}")

    mbk = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=min(10000, max(1000, n // 20)),
        n_init=3,
        max_iter=200,
        reassignment_ratio=0.01,
    )
    batch = min(50000, n)
    rng = np.random.RandomState(seed)
    order = rng.permutation(n) if n > batch else np.arange(n)
    for pass_i in range(3):
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            X = np.asarray(emb[idx], dtype=np.float32)
            mbk.partial_fit(X)
        _log(f"  kmeans pass {pass_i + 1}/3")

    labels = np.empty(n, dtype=np.int32)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        labels[start:end] = mbk.predict(np.asarray(emb[start:end], dtype=np.float32))

    index = pd.read_parquet(index_path)
    if len(index) != n:
        if "row" in index.columns:
            index = index.sort_values("row").reset_index(drop=True)
        if len(index) != n:
            raise RuntimeError(f"index/emb size mismatch {len(index)} vs {n}")

    index = index.copy()
    index["cluster_id"] = labels
    dur = pd.to_numeric(index.get("duration_seconds"), errors="coerce")
    index["_hours"] = dur / 3600.0

    labels_path = out_dir / "labels.parquet"
    index[["video_id", "cluster_id"]].to_parquet(labels_path, index=False)

    summaries = []
    sample_rows = []
    for cid, sub in index.groupby("cluster_id"):
        titles = sub["title"].fillna("").astype(str).tolist()
        channels = sub["channel"].fillna("").astype(str).tolist()
        bag = Counter()
        for t in titles + channels:
            for tok in TOKEN_RE.findall(t.lower()):
                if tok in STOP:
                    continue
                bag[tok] += 1
        top_terms = " ".join(w for w, _ in bag.most_common(12))
        top_channels = sub["channel"].fillna("(empty)").astype(str).value_counts().head(5)
        top_ch_str = " | ".join(f"{c}({nn})" for c, nn in top_channels.items())
        summaries.append(
            {
                "cluster_id": int(cid),
                "n": int(len(sub)),
                "hours": round(float(sub["_hours"].sum(skipna=True)), 1),
                "share_pct": round(100.0 * len(sub) / n, 2),
                "top_terms": top_terms,
                "top_channels": top_ch_str,
                "mean_ml_score": round(
                    float(pd.to_numeric(sub.get("ml_score"), errors="coerce").mean()), 4
                )
                if "ml_score" in sub.columns
                else None,
            }
        )
        take = sub.sample(n=min(samples_per_cluster, len(sub)), random_state=seed + int(cid))
        for _, r in take.iterrows():
            sample_rows.append(
                {
                    "cluster_id": int(cid),
                    "video_id": r.get("video_id"),
                    "title": r.get("title"),
                    "channel": r.get("channel"),
                    "ml_score": r.get("ml_score"),
                    "duration_seconds": r.get("duration_seconds"),
                }
            )

    summary_df = pd.DataFrame(summaries).sort_values("n", ascending=False)
    summary_path = out_dir / "cluster_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    samples_path = out_dir / "cluster_samples.csv"
    pd.DataFrame(sample_rows).to_csv(samples_path, index=False)

    total_hours = float(index["_hours"].sum(skipna=True))
    meta = {
        "category": "exo_entertainment",
        "n_rows": n,
        "dim": dim,
        "k": k,
        "seed": seed,
        "encoder": ck.MINILM_NAME,
        "feature_fields": list(ck.FEATURE_FIELDS),
        "total_hours": round(total_hours, 1),
        "inertia": float(mbk.inertia_) if hasattr(mbk, "inertia_") else None,
        "paths": {
            "embeddings": str(emb_path),
            "index": str(index_path),
            "labels": str(labels_path),
            "summary": str(summary_path),
            "samples": str(samples_path),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    notes = f"""# exo_entertainment text MiniLM 聚类（探索）

- 池行数: **{n:,}**；时长约 **{total_hours:,.0f} h**
- 编码: `{ck.MINILM_NAME}`（title+channel，L2）；KMeans K=**{k}**
- 用途: 扫主题簇；**金标入库后再跑 gold_filter 丢簇**；非交付 KPI
- 读法: `cluster_summary.csv` → `cluster_samples.csv`

## 最大簇（Top 10）

| cluster | n | hours | top_terms |
|--------:|--:|------:|-----------|
"""
    for _, r in summary_df.head(10).iterrows():
        notes += f"| {int(r.cluster_id)} | {int(r.n):,} | {r.hours} | {r.top_terms[:80]} |\n"
    notes += "\n等人标后: `experiments/exo_entertainment_cluster_gold_filter.py --cluster-dir <本目录>`\n"
    (out_dir / "notes.md").write_text(notes, encoding="utf-8")

    _log(f"report → {summary_path.name}, {samples_path.name}, notes.md")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="exo_entertainment MiniLM encode + MiniBatchKMeans")
    ap.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    ap.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=PROJECT / "work" / f"exo_entertainment_text_cluster_{time.strftime('%m%d')}",
    )
    ap.add_argument("--device", default=None, help="mps|cuda|cpu（默认自动）")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--chunksize", type=int, default=5000)
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples-per-cluster", type=int, default=15)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--encode-only", action="store_true")
    ap.add_argument("--cluster-only", action="store_true")
    args = ap.parse_args()

    inp = args.input.resolve()
    out_dir = args.out_dir.resolve()
    if not inp.is_file():
        raise SystemExit(f"input missing: {inp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.cluster_only:
        enc = encode_file(
            inp,
            out_dir,
            device=args.device,
            batch_size=args.batch_size,
            chunksize=args.chunksize,
            resume=args.resume,
        )
        (out_dir / "encode_meta.json").write_text(
            json.dumps({"input": str(inp), **enc}, ensure_ascii=False, indent=2) + "\n"
        )

    if not args.encode_only:
        meta = cluster_and_report(
            out_dir,
            k=args.k,
            seed=args.seed,
            samples_per_cluster=args.samples_per_cluster,
        )
        meta["input"] = str(inp)
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"k": meta["k"], "n_rows": meta["n_rows"], "hours": meta["total_hours"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

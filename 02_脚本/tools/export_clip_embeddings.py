#!/usr/bin/env python3
"""将候选缩略图编码成可续跑的 float16 CLIP embedding store。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.exemplar_sim import ClipEncoder, fetch_thumbnails_batch  # noqa: E402


def _read(path: str) -> pd.DataFrame:
    if Path(path).suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def _ids_hash(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for video_id in ids:
        digest.update(video_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 CLIP 缩略图 embedding")
    parser.add_argument("input", help="CSV/Parquet（需 video_id）")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--cache-dir", default="qc_thumb_cache/exemplar_sim")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-rows", type=int, default=5000)
    parser.add_argument("--encode-batch", type=int, default=64)
    parser.add_argument("--thumb-workers", type=int, default=24)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = _read(args.input)
    if "video_id" not in frame.columns:
        raise SystemExit("[ERROR] 输入需要 video_id")
    frame = frame.drop_duplicates("video_id").copy()
    frame["video_id"] = frame["video_id"].astype(str).str.strip()
    if args.limit:
        frame = frame.head(args.limit).copy()
    ids = frame["video_id"].tolist()
    fingerprint = _ids_hash(ids)

    index_path = out / "index.csv"
    array_path = out / "embeddings.npy"
    ok_path = out / "thumb_ok.npy"
    progress_path = out / "progress.json"
    meta_path = out / "meta.json"

    done = 0
    if (
        not args.overwrite
        and array_path.exists()
        and index_path.exists()
        and progress_path.exists()
    ):
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("ids_sha256") != fingerprint:
            raise SystemExit("[ERROR] store 与当前输入 video_id 不一致；请用 --overwrite")
        done = int(progress.get("done", 0))
        embeddings = np.lib.format.open_memmap(array_path, mode="r+")
        thumb_ok = np.lib.format.open_memmap(ok_path, mode="r+")
        print(f"[续跑] done={done}/{len(ids)}")
    else:
        pd.DataFrame({
            "row": np.arange(len(ids), dtype=np.int64),
            "video_id": ids,
        }).to_csv(index_path, index=False)
        embeddings = np.lib.format.open_memmap(
            array_path, mode="w+", dtype=np.float16, shape=(len(ids), 512),
        )
        embeddings[:] = np.nan
        thumb_ok = np.lib.format.open_memmap(
            ok_path, mode="w+", dtype=np.bool_, shape=(len(ids),),
        )
        thumb_ok[:] = False
        progress_path.write_text(
            json.dumps(
                {"done": 0, "total": len(ids), "ids_sha256": fingerprint},
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

    encoder = ClipEncoder(args.model, args.pretrained)
    started = time.time()
    for start in range(done, len(ids), args.batch_rows):
        end = min(start + args.batch_rows, len(ids))
        chunk_ids = ids[start:end]
        paths = fetch_thumbnails_batch(
            chunk_ids,
            Path(args.cache_dir),
            workers=args.thumb_workers,
        )
        local_ok = [i for i, path in enumerate(paths) if path is not None]
        for offset in range(0, len(local_ok), args.encode_batch):
            local_rows = local_ok[offset : offset + args.encode_batch]
            images = [Image.open(paths[i]).convert("RGB") for i in local_rows]
            vectors = encoder.encode_images(images)
            global_rows = [start + i for i in local_rows]
            embeddings[global_rows] = vectors.astype(np.float16)
            thumb_ok[global_rows] = True
        embeddings.flush()
        thumb_ok.flush()
        elapsed = max(time.time() - started, 1e-6)
        processed = end - done
        progress = {
            "done": end,
            "total": len(ids),
            "ids_sha256": fingerprint,
            "rows_per_sec": round(processed / elapsed, 2),
            "thumb_ok": int(np.count_nonzero(thumb_ok[:end])),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        progress_path.write_text(
            json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"embedding {end}/{len(ids)} "
            f"ok={progress['thumb_ok']} rate={progress['rows_per_sec']}",
            flush=True,
        )

    meta = {
        "input": str(Path(args.input).resolve()),
        "rows": len(ids),
        "dim": 512,
        "dtype": "float16",
        "model": args.model,
        "pretrained": args.pretrained,
        "ids_sha256": fingerprint,
        "thumb_ok": int(np.count_nonzero(thumb_ok)),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

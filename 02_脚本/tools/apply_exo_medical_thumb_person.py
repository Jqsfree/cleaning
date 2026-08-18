#!/usr/bin/env python3
"""exo_medical 缩略图人体框过滤（qc_thumb_cache 命名）。"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.transforms.functional import pil_to_tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.human_live_multiframe import classify_thumbnail_person  # noqa: E402
from core.thumb_cache import resolve_thumbnail_path  # noqa: E402


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_model(device: torch.device):
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    return fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(device).eval()


def score_chunk(
    chunk: pd.DataFrame,
    *,
    cache_dir: Path,
    model,
    device: torch.device,
    batch_size: int,
    score_threshold: float,
    min_person_area: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    pending: list[tuple[int, str, Image.Image]] = []

    def flush() -> None:
        if not pending:
            return
        tensors = [pil_to_tensor(image).float().div(255).to(device) for _, _, image in pending]
        with torch.inference_mode():
            outputs = model(tensors)
        for (source_index, video_id, image), output in zip(pending, outputs, strict=True):
            keep = (output["labels"] == 1) & (output["scores"] >= score_threshold)
            boxes = output["boxes"][keep].detach().cpu().numpy().tolist()
            result = classify_thumbnail_person(
                boxes,
                frame_size=(image.width, image.height),
                min_person_area_ratio=min_person_area,
            )
            rows.append({
                "source_index": source_index,
                "video_id": video_id,
                "thumb_person_action": result["action"],
                "thumb_person_reason": result["reason"],
                "thumb_person_count": result["person_count"],
                "thumb_person_max_area": result["max_person_ratio"],
                "thumb_person_error": "",
            })
        pending.clear()

    for source_index, row in chunk.iterrows():
        video_id = str(row["video_id"])
        image_path = resolve_thumbnail_path(video_id, cache_dir)
        try:
            if image_path is None:
                raise FileNotFoundError("missing_thumbnail")
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            flush()
            error = "missing_thumbnail" if image_path is None else f"decode_error:{type(exc).__name__}"
            result = classify_thumbnail_person(None, frame_size=None, error=error)
            rows.append({
                "source_index": source_index,
                "video_id": video_id,
                "thumb_person_action": result["action"],
                "thumb_person_reason": result["reason"],
                "thumb_person_count": 0,
                "thumb_person_max_area": 0.0,
                "thumb_person_error": error,
            })
            continue
        pending.append((source_index, video_id, image))
        if len(pending) >= batch_size:
            flush()
    flush()
    return pd.DataFrame(rows).sort_values("source_index").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="exo_medical 缩略图人体框过滤")
    parser.add_argument("input")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--cache-dir", default="qc_thumb_cache")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=5_000)
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--min-person-area", type=float, default=0.08)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    source = read_frame(Path(args.input)).reset_index(drop=True)
    if args.limit:
        source = source.head(args.limit).copy()
    if "video_id" not in source.columns:
        raise SystemExit("[ERROR] 输入缺少 video_id")

    output = Path(args.output_dir)
    parts_dir = output / "parts"
    if args.overwrite and output.exists():
        shutil.rmtree(output)
    parts_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"rows={len(source):,} device={device}", flush=True)
    model = load_model(device)

    part_paths: list[Path] = []
    for start in range(0, len(source), args.chunk_size):
        stop = min(start + args.chunk_size, len(source))
        part_path = parts_dir / f"part_{start:09d}_{stop:09d}.parquet"
        part_paths.append(part_path)
        if part_path.exists():
            print(f"skip {stop:,}/{len(source):,}", flush=True)
            continue
        features = score_chunk(
            source.iloc[start:stop],
            cache_dir=Path(args.cache_dir),
            model=model,
            device=device,
            batch_size=args.batch_size,
            score_threshold=args.score_threshold,
            min_person_area=args.min_person_area,
        )
        features.to_parquet(part_path, index=False)
        write_json(output / "progress.json", {
            "status": "running",
            "processed": stop,
            "total": len(source),
            "device": str(device),
        })
        print(f"thumb {stop:,}/{len(source):,}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    features = pd.concat(
        [pd.read_parquet(path) for path in part_paths],
        ignore_index=True,
    ).sort_values("source_index").reset_index(drop=True)
    features.to_parquet(output / "thumb_person_features.parquet", index=False)

    enriched = source.copy()
    for column in features.columns:
        if column not in {"source_index", "video_id"}:
            enriched[column] = features[column].to_numpy()
    drop = enriched[enriched["thumb_person_action"].eq("highconf_drop")].copy()
    remain = enriched[~enriched["thumb_person_action"].eq("highconf_drop")].copy()
    drop.to_csv(output / "highconf_drop.csv", index=False)
    remain.to_csv(output / "remain.csv", index=False)

    elapsed = time.perf_counter() - started
    summary = {
        "input": str(Path(args.input).resolve()),
        "rows": len(source),
        "remain": len(remain),
        "drop": len(drop),
        "min_person_area": args.min_person_area,
        "remain_path": str(output / "remain.csv"),
        "drop_path": str(output / "highconf_drop.csv"),
        "features_path": str(output / "thumb_person_features.parquet"),
        "elapsed_sec": round(elapsed, 1),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

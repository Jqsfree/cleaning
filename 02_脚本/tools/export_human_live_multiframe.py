#!/usr/bin/env python3
"""抓取真人直播 storyboard，并导出人体占比与 SigLIP 2 多帧特征。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.human_live_multiframe import (  # noqa: E402
    classify_multiframe_rule,
    parse_vlm_label,
    summarize_person_frames,
)
from core.yt_dlp_auth import resolve_yt_dlp_auth  # noqa: E402
from qc.vision_storyboard import (  # noqa: E402
    RunConfig,
    detect_js_runtimes,
    load_cached_frames,
    prepare_auth_for_run,
    prepare_storyboard,
    resolve_sb_prefer_order,
)

PROMPTS = [
    "a livestream frame with a real person filling much of the image",
    "a real person talking directly to a webcam or camera",
    "an in-real-life livestream scene with clearly visible people",
    "video game gameplay or a screen recording",
    "a television studio interview or pre-recorded talk show",
    "animation, movie footage, music video, or sports broadcast",
]
VLM_PROMPT = """Judge these six uniformly sampled video frames.
Return T only when at least four frames show a real person clearly and the
person is a substantial visual subject. Return F for gameplay (including a
small facecam), screen recordings, animation, movies, sports, slides, or when
people appear in too few frames. Livestream clips may pass. Return U only when
the frames are genuinely ambiguous. Output exactly one letter: T, F, or U."""


def _read(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def fetch_storyboards(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / ".sb_cache"
    index_path = output / "storyboard_index.csv"
    source = _read(args.input).drop_duplicates("video_id").copy()
    source["video_id"] = source["video_id"].astype(str).str.strip()
    if args.limit:
        source = source.head(args.limit).copy()
    status = source[["video_id"]].copy()
    status["storyboard_status"] = ""
    status["storyboard_error"] = ""
    status["frame_count"] = 0
    status["elapsed_sec"] = np.nan
    if index_path.exists() and not args.overwrite:
        prior = pd.read_csv(index_path, dtype={"video_id": str})
        keep = [
            column for column in status.columns
            if column in prior.columns and column != "video_id"
        ]
        status = status.drop(columns=keep).merge(
            prior[["video_id", *keep]].drop_duplicates("video_id", keep="last"),
            on="video_id",
            how="left",
        )
        for column in ("storyboard_status", "storyboard_error"):
            status[column] = status[column].fillna("")
        status["frame_count"] = pd.to_numeric(
            status["frame_count"], errors="coerce",
        ).fillna(0).astype(int)

    auth = resolve_yt_dlp_auth(
        args.cookies,
        args.cookies_from_browser,
        default_browser="chrome",
    )
    run_id = time.strftime("%Y%m%d_%H%M%S")
    auth, _, _ = prepare_auth_for_run(
        auth, run_id, str(output), prefetch=not args.no_prefetch_cookies,
    )
    js_runtimes = detect_js_runtimes()
    run_cfg = RunConfig(
        meta_sleep_sec=args.meta_sleep,
        use_sidecar=False,
        quiet=True,
        sb_cache_dir=str(cache_dir),
    )
    prefer = resolve_sb_prefer_order(args.sb_prefer, sb_only=True)
    pending = status.index[
        ~status["storyboard_status"].eq("ok")
        & (
            args.retry_errors
            | status["storyboard_status"].eq("")
        )
    ].tolist()

    def _one(idx: int) -> tuple[int, str, str, int, float]:
        video_id = str(status.at[idx, "video_id"])
        started = time.perf_counter()
        prepared = prepare_storyboard(
            video_id,
            auth,
            n_frames=args.frames,
            sb_prefer_order=prefer,
            js_runtimes=js_runtimes,
            run_cfg=run_cfg,
        )
        frames = load_cached_frames(str(cache_dir), video_id, args.frames)
        elapsed = time.perf_counter() - started
        if prepared.error or not frames or len(frames) != args.frames:
            return (
                idx,
                "error",
                prepared.error or "incomplete_frames",
                len(frames or []),
                elapsed,
            )
        return idx, "ok", "", len(frames), elapsed

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_one, idx): idx for idx in pending}
        for future in as_completed(futures):
            idx, state, error, count, elapsed = future.result()
            status.at[idx, "storyboard_status"] = state
            status.at[idx, "storyboard_error"] = error
            status.at[idx, "frame_count"] = count
            status.at[idx, "elapsed_sec"] = round(elapsed, 3)
            completed += 1
            if completed % args.checkpoint_every == 0:
                _atomic_csv(status, index_path)
                print(f"storyboard {completed}/{len(pending)}", flush=True)
    _atomic_csv(status, index_path)
    summary = {
        "input": str(Path(args.input).resolve()),
        "rows": len(status),
        "frames": args.frames,
        "status": status["storyboard_status"].value_counts().to_dict(),
        "errors": status.loc[
            status["storyboard_status"].ne("ok"), "storyboard_error"
        ].value_counts().head(20).to_dict(),
        "elapsed_sec_sum": round(
            float(pd.to_numeric(status["elapsed_sec"], errors="coerce").sum()), 1,
        ),
    }
    (output / "storyboard_fetch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _frame_records(index: pd.DataFrame, cache_dir: Path, n_frames: int) -> list[dict]:
    records: list[dict] = []
    for video_id in index.loc[index["storyboard_status"].eq("ok"), "video_id"]:
        frames = load_cached_frames(str(cache_dir), str(video_id), n_frames)
        if not frames or len(frames) != n_frames:
            continue
        for frame_idx, image in enumerate(frames):
            records.append({
                "video_id": str(video_id),
                "frame_idx": frame_idx,
                "image": image,
                "width": image.width,
                "height": image.height,
            })
    return records


def _detect_people(records: list[dict], batch_size: int, device: torch.device) -> None:
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )
    from torchvision.transforms.functional import pil_to_tensor

    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights,
    ).to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            tensors = [
                pil_to_tensor(row["image"]).float().div(255).to(device)
                for row in chunk
            ]
            outputs = model(tensors)
            for row, result in zip(chunk, outputs, strict=True):
                keep = (result["labels"] == 1) & (
                    result["scores"] >= 0.50
                )
                row["person_boxes"] = (
                    result["boxes"][keep].detach().cpu().numpy().tolist()
                )
                row["person_conf_max"] = float(
                    result["scores"][keep].max().item()
                ) if keep.any() else 0.0
            print(
                f"person {min(start + batch_size, len(records))}/{len(records)}",
                flush=True,
            )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _feature_tensor(output):
    """兼容 transformers 直接 Tensor 与 BaseModelOutputWithPooling。"""
    return output.pooler_output if hasattr(output, "pooler_output") else output


def _siglip_scores(
    records: list[dict],
    batch_size: int,
    device: torch.device,
    model_name: str,
) -> None:
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    text = processor(
        text=PROMPTS,
        padding="max_length",
        return_tensors="pt",
    )
    text = {key: value.to(device) for key, value in text.items()}
    with torch.inference_mode():
        text_features = _feature_tensor(model.get_text_features(**text))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            image_inputs = processor(
                images=[row["image"] for row in chunk],
                return_tensors="pt",
            )
            image_inputs = {
                key: value.to(device) for key, value in image_inputs.items()
            }
            image_features = _feature_tensor(
                model.get_image_features(**image_inputs),
            )
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True,
            )
            scores = torch.softmax(
                10.0 * image_features @ text_features.T,
                dim=1,
            ).float().cpu().numpy()
            for row, values in zip(chunk, scores, strict=True):
                row["siglip_human_live"] = float(values[0])
                row["siglip_talking"] = float(values[1])
                row["siglip_irl"] = float(values[2])
                row["siglip_game"] = float(values[3])
                row["siglip_studio"] = float(values[4])
                row["siglip_other_media"] = float(values[5])
            print(
                f"siglip {min(start + batch_size, len(records))}/{len(records)}",
                flush=True,
            )


def extract_features(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    output = Path(args.output_dir)
    index = pd.read_csv(output / "storyboard_index.csv", dtype={"video_id": str})
    records = _frame_records(index, output / ".sb_cache", args.frames)
    if not records:
        raise SystemExit("[ERROR] 没有可用 storyboard 帧")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"frames={len(records)} device={device}", flush=True)
    person_started = time.perf_counter()
    _detect_people(records, args.person_batch_size, device)
    person_elapsed = time.perf_counter() - person_started
    siglip_started = time.perf_counter()
    _siglip_scores(records, args.siglip_batch_size, device, args.siglip_model)
    siglip_elapsed = time.perf_counter() - siglip_started

    frame_rows = []
    for row in records:
        serializable = {key: value for key, value in row.items() if key != "image"}
        frame_rows.append(serializable)
    frame_frame = pd.DataFrame(frame_rows)
    frame_frame.to_parquet(output / "frame_features.parquet", index=False)

    video_rows = []
    for video_id, group in frame_frame.groupby("video_id", sort=False):
        group = group.sort_values("frame_idx")
        summary = summarize_person_frames(
            group["person_boxes"].tolist(),
            frame_sizes=list(zip(group["width"], group["height"], strict=True)),
            min_person_area_ratio=args.min_person_area,
        )
        row = {
            "video_id": video_id,
            **{key: value for key, value in summary.items() if key != "person_area_ratios"},
            "person_area_ratios": json.dumps(summary["person_area_ratios"]),
        }
        for column in (
            "siglip_human_live",
            "siglip_talking",
            "siglip_irl",
            "siglip_game",
            "siglip_studio",
            "siglip_other_media",
        ):
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_max"] = float(group[column].max())
        row["game_dominant_frames"] = int(
            (group["siglip_game"] >= args.game_threshold).sum()
        )
        row["multiframe_rule"] = classify_multiframe_rule(
            summary,
            group["siglip_game"].tolist(),
            required_large_frames=args.required_large_frames,
            game_threshold=args.game_threshold,
            required_game_frames=args.required_game_frames,
        )
        video_rows.append(row)
    video_frame = pd.DataFrame(video_rows)
    video_frame.to_parquet(output / "video_features.parquet", index=False)
    summary = {
        "videos": len(video_frame),
        "frames": len(frame_frame),
        "device": str(device),
        "person_model": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "siglip_model": args.siglip_model,
        "rule": video_frame["multiframe_rule"].value_counts().to_dict(),
        "elapsed_sec": round(time.perf_counter() - started, 1),
        "person_elapsed_sec": round(person_elapsed, 1),
        "siglip_elapsed_sec": round(siglip_elapsed, 1),
    }
    (output / "feature_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_local_vlm(args: argparse.Namespace) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output = Path(args.output_dir)
    index = pd.read_csv(output / "storyboard_index.csv", dtype={"video_id": str})
    ids = index.loc[index["storyboard_status"].eq("ok"), "video_id"].tolist()
    if args.limit:
        ids = ids[: args.limit]
    score_path = output / "vlm_scores.csv"
    prior = pd.DataFrame(columns=["video_id", "vlm_result", "vlm_raw", "vlm_model"])
    if score_path.exists() and not args.overwrite:
        prior = pd.read_csv(score_path, dtype=str)
    done = set(
        prior.loc[prior["vlm_result"].isin(["T", "F", "U"]), "video_id"]
        .astype(str)
    )
    pending = [video_id for video_id in ids if str(video_id) not in done]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device).eval()
    rows = prior.to_dict("records")
    for offset, video_id in enumerate(pending, 1):
        frame_paths = [
            output / ".sb_cache" / str(video_id) / f"frame_{i:02d}.jpg"
            for i in range(args.frames)
        ]
        if not all(path.exists() for path in frame_paths):
            rows.append({
                "video_id": video_id,
                "vlm_result": "ERROR",
                "vlm_raw": "missing_frames",
                "vlm_model": args.model,
            })
            continue
        content = [
            {"type": "image", "path": str(path.resolve())}
            for path in frame_paths
        ]
        content.append({"type": "text", "text": VLM_PROMPT})
        messages = [{"role": "user", "content": content}]
        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=8,
                )
            prompt_len = inputs["input_ids"].shape[1]
            raw = processor.batch_decode(
                generated[:, prompt_len:],
                skip_special_tokens=True,
            )[0].strip()
            label = parse_vlm_label(raw)
        except Exception as exc:
            raw = f"{type(exc).__name__}:{str(exc)[:160]}"
            label = "ERROR"
        rows.append({
            "video_id": video_id,
            "vlm_result": label,
            "vlm_raw": raw,
            "vlm_model": args.model,
        })
        if offset % args.checkpoint_every == 0:
            _atomic_csv(pd.DataFrame(rows), score_path)
            print(f"vlm {offset}/{len(pending)}", flush=True)
    result = pd.DataFrame(rows).drop_duplicates("video_id", keep="last")
    _atomic_csv(result, score_path)
    summary = {
        "model": args.model,
        "device": str(device),
        "rows": len(result),
        "results": result["vlm_result"].value_counts().to_dict(),
    }
    (output / "vlm_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="真人直播多帧 storyboard 特征")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("input")
    fetch.add_argument("-o", "--output-dir", required=True)
    fetch.add_argument("--frames", type=int, default=6)
    fetch.add_argument("--workers", type=int, default=3)
    fetch.add_argument("--checkpoint-every", type=int, default=20)
    fetch.add_argument("--meta-sleep", type=float, default=0.5)
    fetch.add_argument("--sb-prefer", default="sb2")
    fetch.add_argument("--cookies")
    fetch.add_argument("--cookies-from-browser")
    fetch.add_argument("--no-prefetch-cookies", action="store_true")
    fetch.add_argument("--retry-errors", action="store_true")
    fetch.add_argument("--overwrite", action="store_true")
    fetch.add_argument("--limit", type=int)
    fetch.set_defaults(func=fetch_storyboards)

    extract = sub.add_parser("extract")
    extract.add_argument("-o", "--output-dir", required=True)
    extract.add_argument("--frames", type=int, default=6)
    extract.add_argument("--person-batch-size", type=int, default=4)
    extract.add_argument("--siglip-batch-size", type=int, default=32)
    extract.add_argument(
        "--siglip-model",
        default="google/siglip2-base-patch16-224",
    )
    extract.add_argument("--min-person-area", type=float, default=0.08)
    extract.add_argument("--required-large-frames", type=int, default=4)
    extract.add_argument("--game-threshold", type=float, default=0.60)
    extract.add_argument("--required-game-frames", type=int, default=4)
    extract.set_defaults(func=extract_features)

    vlm = sub.add_parser("vlm")
    vlm.add_argument("-o", "--output-dir", required=True)
    vlm.add_argument("--frames", type=int, default=6)
    vlm.add_argument(
        "--model",
        default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    )
    vlm.add_argument("--checkpoint-every", type=int, default=10)
    vlm.add_argument("--limit", type=int)
    vlm.add_argument("--overwrite", action="store_true")
    vlm.set_defaults(func=run_local_vlm)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

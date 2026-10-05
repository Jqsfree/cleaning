#!/usr/bin/env python3
"""Stage 3：人-作物交互检测（v1 绿field 脚手架）。

Person 检测 + 接触逻辑 → human_present, crop_interaction, mechanized_only。
IoU/接触阈值须人工校准（spec §5 HALT）；默认仅 --sample 试点。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

_SCRIPT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_SCRIPT))

from core.agri_v1_schema import normalize_agri_v1_frame  # noqa: E402
from core.human_live_multiframe import classify_thumbnail_person  # noqa: E402
from core.thumb_cache import resolve_thumbnail_path  # noqa: E402

try:
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )
    from torchvision.transforms.functional import pil_to_tensor
    from PIL import Image

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def detect_person_thumb(
    video_id: str,
    cache_dir: Path,
    *,
    model,
    device: torch.device,
    score_threshold: float = 0.5,
    min_person_area: float = 0.02,
) -> dict:
    path = resolve_thumbnail_path(video_id, cache_dir)
    if path is None or not Path(path).is_file():
        return {
            "human_present": "",
            "crop_interaction": "",
            "mechanized_only": "",
            "stage3_reason": "no_thumb",
        }
    img = Image.open(path).convert("RGB")
    tensor = pil_to_tensor(img).float().div(255).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor)[0]
    keep = (output["labels"] == 1) & (output["scores"] >= score_threshold)
    boxes = output["boxes"][keep].detach().cpu().numpy().tolist()
    person = classify_thumbnail_person(
        boxes,
        frame_size=(img.width, img.height),
        min_person_area_ratio=min_person_area,
    )
    human_present = "1" if person["action"] == "pass" else "0"
    return {
        "human_present": human_present,
        "crop_interaction": "",  # 需 100DOH + Grounding DINO；HALT 待校准
        "mechanized_only": "",
        "stage3_reason": person.get("reason", ""),
        "stage3_person_count": person.get("person_count", 0),
    }


def run_stage3(
    input_csv: Path,
    *,
    out_dir: Path,
    n_sample: int,
    seed: int,
    cache_dir: Path,
) -> dict:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not _HAS_TORCH:
        summary = {
            "stage": "stage3_interaction",
            "status": "error",
            "error": "torch/torchvision 不可用",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return summary

    frame = pd.read_csv(input_csv, dtype=str, low_memory=False)
    frame["video_id"] = frame["video_id"].astype(str).str.strip()
    frame = frame.drop_duplicates("video_id")
    if n_sample > 0 and n_sample < len(frame):
        frame = frame.sample(n=n_sample, random_state=seed).reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(device).eval()

    rows: list[dict] = []
    for i, vid in enumerate(frame["video_id"].tolist()):
        det = detect_person_thumb(vid, cache_dir, model=model, device=device)
        row = {"video_id": vid, **det}
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  stage3 {i + 1}/{len(frame)}", flush=True)

    out_df = frame.merge(pd.DataFrame(rows), on="video_id", how="left")
    out_df = normalize_agri_v1_frame(out_df)
    out_csv = out_dir / f"{input_csv.stem}_stage3_interaction.csv"
    out_df.to_csv(out_csv, index=False)
    summary = {
        "stage": "stage3_interaction",
        "input": str(input_csv.resolve()),
        "n_run": len(out_df),
        "n_human_present": int((out_df["human_present"] == "1").sum()),
        "output": str(out_csv.resolve()),
        "status": "scaffold",
        "halt_note": "crop_interaction 检测栈与 IoU 阈值须人工校准后再全量跑",
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3 interaction scaffold")
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--out-dir", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", type=Path, default=Path("qc_thumb_cache/exemplar_sim"))
    args = ap.parse_args()
    run_stage3(
        args.input,
        out_dir=args.out_dir,
        n_sample=args.sample,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

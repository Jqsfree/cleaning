#!/usr/bin/env python3
"""human268 上校准 exo_agriculture CLIP 阈值（T 误伤=0 优先）。"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT))

from categories.exo_agriculture.cascade_clip import (  # noqa: E402
    _RULES,
    decide,
    encode_prompt_pairs,
    load_cfg,
    margins_from_feats,
    q_keys,
    require_keys,
    score_frame,
)
from core.exemplar_sim import ClipEncoder  # noqa: E402

CALIB_JSON = Path(__file__).resolve().parents[2] / "models/exo_agriculture_clip_calibration.json"
DEFAULT_LABELS = (
    _SCRIPT.parent
    / "data/runs/exo_agriculture/machine_0814/03_qc/human268_thumb_v2_plus/exo农业_human_qc.csv"
)


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if "qc_result" not in df.columns:
        raise ValueError(f"{path} 需要 qc_result")
    lab = df["qc_result"].astype(str).str.strip().str.upper()
    df = df[lab.isin(["T", "F"])].copy()
    df["label"] = lab[lab.isin(["T", "F"])]
    df["y"] = (df["label"] == "T").astype(int)
    return df.drop_duplicates("video_id").reset_index(drop=True)


def pick_thresholds(
    scored: pd.DataFrame,
    *,
    require: list[str],
    min_f_precision: float = 0.95,
    max_t_hurt: int = 0,
) -> dict | None:
    """网格搜索 require 键阈值；T 误伤≤max_t_hurt，F precision≥min_f_precision。"""
    labels = scored["label"].to_numpy(dtype=str)
    ok = scored["clip_thumb_ok"].fillna(False).astype(bool).to_numpy()
    req_cols = [f"clip_{k}" for k in require]
    arrs = {k: scored[c].to_numpy(dtype=float) for k, c in zip(require, req_cols)}
    grid = np.round(np.arange(-0.05, 0.31, 0.01), 2)
    n_f = int((labels == "F").sum())
    n_t = int((labels == "T").sum())
    candidates = []
    for combo in product(grid, repeat=len(require)):
        thr = {k: float(v) for k, v in zip(require, combo)}
        scores = {k: arrs[k] for k in require}
        decision = decide(scores, thresholds=thr, require=require, ok_mask=ok)
        drop = decision == "clip_fail"
        dropped = labels[drop]
        n_drop = int(drop.sum())
        f_caught = int((dropped == "F").sum())
        t_hurt = int((dropped == "T").sum())
        if n_drop == 0 or t_hurt > max_t_hurt:
            continue
        prec = f_caught / max(n_drop, 1)
        if prec >= min_f_precision:
            candidates.append({
                "thresholds": thr,
                "n_drop": n_drop,
                "f_caught": f_caught,
                "f_recall": f_caught / max(n_f, 1),
                "drop_precision": prec,
                "t_hurt": t_hurt,
                "t_hurt_rate": t_hurt / max(n_t, 1),
            })
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r["f_caught"], r["drop_precision"], -r["t_hurt"]))


def write_config_thresholds(
    cfg_path: Path,
    thresholds: dict[str, float],
    *,
    calib_id: str,
) -> None:
    text = cfg_path.read_text(encoding="utf-8")
    lines = ["[thresholds]"]
    keys = load_cfg(cfg_path)
    for key in q_keys(keys):
        val = thresholds.get(key, float(keys["thresholds"][key]))
        lines.append(f"{key} = {val}")
    block = "\n".join(lines) + "\n"
    text = re.sub(r"\[thresholds\][^\[]*", block, text, count=1)
    if "calib = " in text:
        text = re.sub(r'calib = "[^"]*"', f'calib = "{calib_id}"', text)
    else:
        text = text.replace(
            'pretrained = "openai"',
            f'pretrained = "openai"\ncalib = "{calib_id}"',
            1,
        )
    cfg_path.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="校准 exo_agriculture CLIP 阈值")
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--config", type=Path, default=_RULES)
    ap.add_argument("--write-config", type=Path, default=None, help="写回 TOML thresholds")
    ap.add_argument("--calibration-json", type=Path, default=CALIB_JSON)
    ap.add_argument("--cache-dir", default="qc_thumb_cache/exemplar_sim")
    ap.add_argument("--thumb-workers", type=int, default=12)
    ap.add_argument(
        "--max-t-hurt", type=int, default=3,
        help="收紧档允许的 T 误伤上限（默认 3/268；0=只写 certain-noise 档）",
    )
    args = ap.parse_args()

    labels = load_labels(args.labels)
    cfg = load_cfg(args.config)
    req = require_keys(cfg)
    keys = q_keys(cfg)

    scored = score_frame(
        labels,
        cfg=cfg,
        cache_dir=args.cache_dir,
        thumb_workers=args.thumb_workers,
    )
    scored["label"] = labels.set_index("video_id").loc[
        scored["video_id"].astype(str), "label"
    ].to_numpy()

    strict = pick_thresholds(scored, require=req, max_t_hurt=0)
    tight = None
    if args.max_t_hurt > 0:
        tight = pick_thresholds(
            scored, require=req, min_f_precision=0.9, max_t_hurt=args.max_t_hurt,
        )
    chosen = tight if (tight and (strict is None or tight["f_caught"] > strict["f_caught"])) else strict
    thr_out = dict(cfg["thresholds"])
    note = None
    if chosen is None:
        note = "无可用阈值组合；保持 thresholds=0.0，应用前须目视"
        for k in req:
            thr_out[k] = 0.0
    else:
        thr_out.update(chosen["thresholds"])
        if chosen is tight and strict is not None:
            note = (
                f"收紧档 t_hurt≤{args.max_t_hurt}；"
                f"strict(t_hurt=0) 仅 f_caught={strict['f_caught']}"
            )

    calib_id = f"human268_{time.strftime('%Y%m%d')}"
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "labels": str(args.labels),
        "n_labels": len(labels),
        "n_t": int((labels["label"] == "T").sum()),
        "n_f": int((labels["label"] == "F").sum()),
        "require": req,
        "strict": strict,
        "tight": tight,
        "thresholds": thr_out,
        "note": note,
    }
    args.calibration_json.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.write_config:
        write_config_thresholds(args.write_config, thr_out, calib_id=calib_id)
        result["config_written"] = str(args.write_config)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

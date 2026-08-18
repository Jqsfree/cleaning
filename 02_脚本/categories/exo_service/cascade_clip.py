#!/usr/bin/env python3
"""exo_service L3：缩略图 CLIP 零样本三问（本地，非 API）。"""

from __future__ import annotations

import json
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from core.exemplar_sim import ClipEncoder, fetch_thumbnails_batch
from core.log import log

_RULES = Path(__file__).resolve().parent / "rules" / "cascade_l3_clip.toml"


def q_keys(cfg: dict[str, Any]) -> tuple[str, ...]:
    return tuple(cfg["decision"]["score_keys"])


def require_keys(cfg: dict[str, Any]) -> list[str]:
    return list(cfg["decision"]["require"])


def load_l3_cfg(path: Path | None = None) -> dict[str, Any]:
    return tomllib.loads((path or _RULES).read_text(encoding="utf-8"))


def _prompt_bank(cfg: dict[str, Any]) -> dict[str, tuple[list[str], list[str]]]:
    return {k: (list(cfg[k]["pos"]), list(cfg[k]["neg"])) for k in q_keys(cfg)}


def encode_prompt_pairs(
    encoder: ClipEncoder, bank: dict[str, tuple[list[str], list[str]]]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, (pos, neg) in bank.items():
        out[key] = (encoder.encode_text(pos), encoder.encode_text(neg))
    return out


def margins_from_feats(
    feats: np.ndarray,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    """feats (N, D) → 每问 margin (N,)。"""
    scores: dict[str, np.ndarray] = {}
    for key, (pos, neg) in pairs.items():
        scores[key] = feats @ pos.mean(axis=0) - feats @ neg.mean(axis=0)
    return scores


def decide(
    scores: dict[str, np.ndarray],
    *,
    thresholds: dict[str, float],
    require: list[str],
    ok_mask: np.ndarray,
) -> np.ndarray:
    """clip_pass / clip_fail / no_thumb。"""
    n = len(ok_mask)
    out = np.full(n, "no_thumb", dtype=object)
    passed = np.ones(n, dtype=bool)
    for key in require:
        passed &= scores[key] >= float(thresholds[key])
    out[ok_mask & passed] = "clip_pass"
    out[ok_mask & ~passed] = "clip_fail"
    return out


def sample_candidates(
    frame: pd.DataFrame, n: int, seed: int
) -> pd.DataFrame:
    if n <= 0 or n >= len(frame):
        return frame.copy()
    if "industry_primary" not in frame.columns:
        return frame.sample(n=n, random_state=seed)
    parts: list[pd.DataFrame] = []
    grouped = frame.groupby("industry_primary", dropna=False)
    remain = n
    keys = list(grouped.groups)
    for i, key in enumerate(keys):
        g = grouped.get_group(key)
        if i == len(keys) - 1:
            take = min(len(g), remain)
        else:
            take = min(len(g), max(1, round(n * len(g) / len(frame))))
            take = min(take, remain)
        if take > 0:
            parts.append(g.sample(n=take, random_state=seed))
            remain -= take
        if remain <= 0:
            break
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.reset_index(drop=True)


def run_l3(
    input_path: str,
    output_dir: str,
    *,
    n_sample: int = 2000,
    seed: int = 42,
    cache_dir: str = "qc_thumb_cache/exemplar_sim",
    batch_size: int = 64,
    thumb_workers: int = 24,
    cfg_path: Path | None = None,
    stem: str = "l3_clip",
    thumb_chunk: int = 5000,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_l3_cfg(cfg_path)
    thr = cfg["thresholds"]
    keys = q_keys(cfg)
    frame = pd.read_csv(input_path, dtype=str, low_memory=False)
    if "video_id" not in frame.columns:
        raise ValueError("需要 video_id")
    frame = frame.drop_duplicates("video_id").copy()
    sampled = sample_candidates(frame, n_sample, seed)
    ids = sampled["video_id"].astype(str).str.strip().tolist()
    log(f"L3 CLIP sample {len(ids):,} / pool {len(frame):,}  cfg={cfg_path or _RULES}")

    encoder = ClipEncoder(cfg["meta"]["model"], cfg["meta"]["pretrained"])
    log(f"  encoder device={encoder.device}")
    pairs = encode_prompt_pairs(encoder, _prompt_bank(cfg))

    feats = np.zeros((len(ids), 512), dtype=np.float32)
    ok = np.zeros(len(ids), dtype=bool)
    cache = Path(cache_dir)
    for t0i in range(0, len(ids), thumb_chunk):
        t1i = min(t0i + thumb_chunk, len(ids))
        chunk_ids = ids[t0i:t1i]
        paths = fetch_thumbnails_batch(chunk_ids, cache, workers=thumb_workers)
        ok_local = [j for j, p in enumerate(paths) if p is not None]
        for start in range(0, len(ok_local), batch_size):
            batch = ok_local[start : start + batch_size]
            imgs = [Image.open(paths[j]).convert("RGB") for j in batch]
            chunk_feats = encoder.encode_images(imgs)
            for k, j in enumerate(batch):
                ii = t0i + j
                feats[ii] = chunk_feats[k]
                ok[ii] = True
        log(f"  thumbs+encode {t1i}/{len(ids)}  ok={int(ok.sum()):,}")

    scores = margins_from_feats(feats, pairs)
    decision = decide(
        scores,
        thresholds={k: float(thr[k]) for k in keys},
        require=require_keys(cfg),
        ok_mask=ok,
    )
    result = sampled.copy()
    for key in keys:
        result[f"clip_{key}"] = scores[key]
    result["clip_decision"] = decision

    date_tag = time.strftime("%m%d")
    scored_csv = out / f"{stem}_scored_{date_tag}.csv"
    pass_csv = out / f"{stem}_pass_{date_tag}.csv"
    fail_csv = out / f"{stem}_fail_{date_tag}.csv"
    result.to_csv(scored_csv, index=False)
    result[result["clip_decision"] == "clip_pass"].to_csv(pass_csv, index=False)
    result[result["clip_decision"] == "clip_fail"].to_csv(fail_csv, index=False)

    n_ok = int(ok.sum())
    n_pass = int((decision == "clip_pass").sum())
    n_fail = int((decision == "clip_fail").sum())
    n_miss = int((decision == "no_thumb").sum())
    summary = {
        "layer": cfg["meta"].get("layer", "l3_clip"),
        "cfg": str(cfg_path or _RULES),
        "input": input_path,
        "n_pool": len(frame),
        "n_sample": len(ids),
        "seed": seed,
        "device": encoder.device,
        "n_thumb_ok": n_ok,
        "n_clip_pass": n_pass,
        "n_clip_fail": n_fail,
        "n_no_thumb": n_miss,
        "pass_rate_among_ok": round(n_pass / n_ok, 4) if n_ok else None,
        "thresholds": thr,
        "require": require_keys(cfg),
        "score_p50": {
            k: round(float(np.median(scores[k][ok])), 4) if n_ok else None
            for k in keys
        },
        "scored_csv": str(scored_csv),
        "pass_csv": str(pass_csv),
        "fail_csv": str(fail_csv),
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "notes": [
            "local open_clip, not DashScope",
            f"require={require_keys(cfg)}",
        ],
    }
    sum_path = out / f"{stem}_summary.json"
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  pass={n_pass} fail={n_fail} no_thumb={n_miss} → {sum_path}")
    return summary

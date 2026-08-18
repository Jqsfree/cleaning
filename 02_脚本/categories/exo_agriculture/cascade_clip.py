#!/usr/bin/env python3
"""exo_agriculture：缩略图 CLIP 零样本（本地，非 DashScope）。"""

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
from core.run_manifest import maybe_update_stage

_RULES = Path(__file__).resolve().parent / "rules" / "cascade_harvest_clip.toml"


def q_keys(cfg: dict[str, Any]) -> tuple[str, ...]:
    return tuple(cfg["decision"]["score_keys"])


def require_keys(cfg: dict[str, Any]) -> list[str]:
    return list(cfg["decision"]["require"])


def load_cfg(path: Path | None = None) -> dict[str, Any]:
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


def sample_candidates(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n >= len(frame):
        return frame.copy()
    return frame.sample(n=n, random_state=seed).reset_index(drop=True)


def _encode_ids(
    ids: list[str],
    encoder: ClipEncoder,
    *,
    cache_dir: Path,
    batch_size: int,
    thumb_workers: int,
    thumb_chunk: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    feats = np.zeros((len(ids), 512), dtype=np.float32)
    ok = np.zeros(len(ids), dtype=bool)
    for t0i in range(0, len(ids), thumb_chunk):
        t1i = min(t0i + thumb_chunk, len(ids))
        chunk_ids = ids[t0i:t1i]
        paths = fetch_thumbnails_batch(chunk_ids, cache_dir, workers=thumb_workers)
        ok_local = [j for j, p in enumerate(paths) if p is not None]
        for start in range(0, len(ok_local), batch_size):
            batch = ok_local[start : start + batch_size]
            imgs = [Image.open(paths[j]).convert("RGB") for j in batch]
            chunk_feats = encoder.encode_images(imgs)
            for k, j in enumerate(batch):
                ii = t0i + j
                feats[ii] = chunk_feats[k]
                ok[ii] = True
    return feats, ok


def score_frame(
    frame: pd.DataFrame,
    *,
    cfg: dict[str, Any],
    encoder: ClipEncoder | None = None,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    cache_dir: str = "qc_thumb_cache/exemplar_sim",
    batch_size: int = 64,
    thumb_workers: int = 16,
    thumb_chunk: int = 5000,
) -> pd.DataFrame:
    """对含 video_id 的表打分；返回 clip_* 列 + clip_decision。"""
    if "video_id" not in frame.columns:
        raise ValueError("需要 video_id")
    work = frame.drop_duplicates("video_id").copy()
    ids = work["video_id"].astype(str).str.strip().tolist()
    keys = q_keys(cfg)
    thr = cfg["thresholds"]
    enc = encoder or ClipEncoder(cfg["meta"]["model"], cfg["meta"]["pretrained"])
    prs = pairs or encode_prompt_pairs(enc, _prompt_bank(cfg))
    feats, ok = _encode_ids(
        ids,
        enc,
        cache_dir=Path(cache_dir),
        batch_size=batch_size,
        thumb_workers=thumb_workers,
        thumb_chunk=thumb_chunk,
    )
    scores = margins_from_feats(feats, prs)
    decision = decide(
        scores,
        thresholds={k: float(thr[k]) for k in keys},
        require=require_keys(cfg),
        ok_mask=ok,
    )
    out = work.copy()
    for key in keys:
        out[f"clip_{key}"] = scores[key]
    out["clip_decision"] = decision
    out["clip_thumb_ok"] = ok
    return out


def _scored_columns(keys: tuple[str, ...]) -> list[str]:
    return ["video_id", "clip_decision", "clip_thumb_ok", *[f"clip_{k}" for k in keys]]


def run_harvest_clip(
    input_path: str,
    output_dir: str,
    *,
    n_sample: int = 0,
    seed: int = 42,
    cache_dir: str = "qc_thumb_cache/exemplar_sim",
    batch_size: int = 64,
    thumb_workers: int = 16,
    cfg_path: Path | None = None,
    stem: str = "harvest_clip",
    batch_rows: int = 5000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """全量或抽样 CLIP 过滤；batch_rows>0 时分批 checkpoint。"""
    t0 = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(cfg_path)
    keys = q_keys(cfg)
    frame = pd.read_csv(input_path, dtype=str, low_memory=False)
    if "video_id" not in frame.columns:
        raise ValueError("需要 video_id")
    frame["video_id"] = frame["video_id"].astype(str).str.strip()
    frame = frame.drop_duplicates("video_id").copy()
    pool_n = len(frame)
    work = sample_candidates(frame, n_sample, seed) if n_sample > 0 else frame
    ids = work["video_id"].tolist()

    date_tag = time.strftime("%m%d")
    stem_base = Path(input_path).stem if stem == "harvest_clip" else stem
    ckpt_path = out / f"{stem_base}_clip.ckpt.csv"
    scored_csv = out / f"{stem_base}_clip_scored_{date_tag}.csv"
    pass_csv = out / f"{stem_base}_clip_pass_{date_tag}.csv"
    fail_csv = out / f"{stem_base}_clip_fail_{date_tag}.csv"
    remain_csv = out / f"{stem_base}_clip_remain_{date_tag}.csv"
    drop_csv = out / f"{stem_base}_clip_drop_{date_tag}.csv"
    progress_path = out / f"{stem_base}_clip_progress.json"
    sum_path = out / f"{stem_base}_clip_summary.json"

    done: set[str] = set()
    parts: list[pd.DataFrame] = []
    if ckpt_path.is_file() and not overwrite:
        prev = pd.read_csv(ckpt_path, dtype={"video_id": str})
        parts.append(prev)
        done = set(prev["video_id"].astype(str).str.strip())
        log(f"[续跑] checkpoint {len(done):,} 行")

    pending = [vid for vid in ids if vid not in done]
    log(
        f"harvest CLIP pool={pool_n:,} run={len(ids):,} pending={len(pending):,} "
        f"cfg={cfg_path or _RULES}"
    )

    encoder = ClipEncoder(cfg["meta"]["model"], cfg["meta"]["pretrained"])
    log(f"  encoder device={encoder.device}")
    pairs = encode_prompt_pairs(encoder, _prompt_bank(cfg))

    batch_i = 0
    for start in range(0, len(pending), max(batch_rows, 1)):
        batch_i += 1
        chunk_ids = pending[start : start + max(batch_rows, 1)]
        chunk_frame = work[work["video_id"].isin(chunk_ids)].copy()
        log(
            f"=== batch {batch_i} rows={len(chunk_ids)} "
            f"global {start + len(done) + 1}-{start + len(done) + len(chunk_ids)}/{len(ids)} ==="
        )
        scored = score_frame(
            chunk_frame,
            cfg=cfg,
            encoder=encoder,
            pairs=pairs,
            cache_dir=cache_dir,
            batch_size=batch_size,
            thumb_workers=thumb_workers,
        )
        parts.append(scored[_scored_columns(keys)])
        all_scored = pd.concat(parts, ignore_index=True)
        all_scored = all_scored.drop_duplicates("video_id", keep="last")
        tmp = ckpt_path.with_suffix(".tmp.csv")
        all_scored.to_csv(tmp, index=False)
        tmp.replace(ckpt_path)
        elapsed = time.perf_counter() - t0
        rate = len(all_scored) / max(elapsed, 1e-6)
        rem = (len(ids) - len(all_scored)) / max(rate, 1e-6)
        counts = all_scored["clip_decision"].value_counts().to_dict()
        progress = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "done": len(all_scored),
            "total": len(ids),
            "batch_rows": batch_rows,
            "decision_counts": counts,
            "rows_per_sec": round(rate, 2),
            "eta_sec": int(rem),
        }
        progress_path.write_text(
            json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        log(f"  ckpt {len(all_scored):,}/{len(ids):,}  {rate:.1f} r/s  eta≈{rem/3600:.1f}h")

    all_scored = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    all_scored = all_scored.drop_duplicates("video_id", keep="last")
    base = work.drop(columns=[c for c in work.columns if c.startswith("clip_")], errors="ignore")
    merged = base.merge(all_scored, on="video_id", how="left")
    merged.to_csv(scored_csv, index=False)

    passed = merged[merged["clip_decision"] == "clip_pass"]
    failed = merged[merged["clip_decision"] == "clip_fail"]
    passed.to_csv(pass_csv, index=False)
    failed.to_csv(fail_csv, index=False)

    # remain = 非 clip_fail（含 no_thumb / clip_pass）
    remain = merged[merged["clip_decision"] != "clip_fail"].copy()
    remain.to_csv(remain_csv, index=False)
    failed.to_csv(drop_csv, index=False)

    n_ok = int(merged["clip_thumb_ok"].fillna(False).astype(bool).sum()) if len(merged) else 0
    n_pass = int((merged["clip_decision"] == "clip_pass").sum())
    n_fail = int((merged["clip_decision"] == "clip_fail").sum())
    n_miss = int((merged["clip_decision"] == "no_thumb").sum())
    summary = {
        "layer": cfg["meta"].get("layer", "harvest_clip"),
        "cfg": str(cfg_path or _RULES),
        "input": input_path,
        "n_pool": pool_n,
        "n_run": len(ids),
        "seed": seed,
        "device": encoder.device,
        "n_thumb_ok": n_ok,
        "n_clip_pass": n_pass,
        "n_clip_fail": n_fail,
        "n_no_thumb": n_miss,
        "n_clip_remain": len(remain),
        "pass_rate_among_ok": round(n_pass / n_ok, 4) if n_ok else None,
        "thresholds": cfg["thresholds"],
        "require": require_keys(cfg),
        "scored_csv": str(scored_csv),
        "pass_csv": str(pass_csv),
        "fail_csv": str(fail_csv),
        "remain_csv": str(remain_csv),
        "drop_csv": str(drop_csv),
        "checkpoint": str(ckpt_path),
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "notes": [
            "local open_clip; clip_decision≠交付 KPI",
            "clip_remain=非 clip_fail（no_thumb 保留交 VL）",
            f"require={require_keys(cfg)}",
        ],
    }
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  pass={n_pass} fail={n_fail} remain={len(remain)} no_thumb={n_miss} → {sum_path}")

    maybe_update_stage(
        out,
        "harvest_clip",
        paths={
            "scored": str(scored_csv),
            "remain": str(remain_csv),
            "drop": str(drop_csv),
            "summary": str(sum_path),
            "checkpoint": str(ckpt_path),
        },
        stats={
            "n": len(merged),
            "n_remain": len(remain),
            "n_fail": n_fail,
            **{f"clip_{k}": int(v) for k, v in merged["clip_decision"].value_counts().to_dict().items()},
        },
    )
    return summary

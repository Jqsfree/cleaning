#!/usr/bin/env python3
"""对 0814 旧 clip_pass 套 v1.1（从 embedding store 重打，免重编码）。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from categories.exo_agriculture.cascade_clip import (  # noqa: E402
    _prompt_bank,
    decide,
    encode_prompt_pairs,
    load_cfg,
    margins_from_feats,
    q_keys,
    require_keys,
)
from core.exemplar_sim import ClipEncoder  # noqa: E402


def hours(frame: pd.DataFrame) -> float | None:
    if "duration_seconds" not in frame.columns:
        return None
    return round(pd.to_numeric(frame["duration_seconds"], errors="coerce").sum() / 3600.0, 1)


def main() -> int:
    t0 = time.perf_counter()
    out = Path("data/runs/exo_agriculture/machine_0814/06_tools")
    store = Path("data/assets/embeddings/exo_agriculture_0814_semantic_remain")
    print("load store...", flush=True)
    index = pd.read_csv(store / "index.csv", dtype={"video_id": str})
    index["video_id"] = index["video_id"].astype(str).str.strip()
    emb = np.load(store / "embeddings.npy", mmap_mode="r")
    ok = np.load(store / "thumb_ok.npy")
    n = len(index)
    print(f"rows={n}", flush=True)

    cfg = load_cfg()
    keys = q_keys(cfg)
    enc = ClipEncoder(cfg["meta"]["model"], cfg["meta"]["pretrained"])
    print("device", enc.device, flush=True)
    pairs = encode_prompt_pairs(enc, _prompt_bank(cfg))

    decisions = np.empty(n, dtype=object)
    scores_hold = {k: np.empty(n, dtype=np.float32) for k in keys}
    batch = 100_000
    for start in range(0, n, batch):
        end = min(start + batch, n)
        feats = np.zeros((end - start, 512), dtype=np.float32)
        ok_chunk = ok[start:end]
        idx = np.flatnonzero(ok_chunk)
        if len(idx):
            feats[idx] = np.asarray(emb[start:end][idx], dtype=np.float32)
            norms = np.linalg.norm(feats[idx], axis=1, keepdims=True)
            feats[idx] = feats[idx] / np.maximum(norms, 1e-8)
        sc = margins_from_feats(feats, pairs)
        for k in keys:
            scores_hold[k][start:end] = sc[k]
        decisions[start:end] = decide(
            sc,
            thresholds={k: float(cfg["thresholds"][k]) for k in keys},
            require=require_keys(cfg),
            ok_mask=ok_chunk,
        )
        print(f"scored {end}/{n}", flush=True)

    v11 = pd.DataFrame({"video_id": index["video_id"].tolist(), "clip_decision_v11": decisions})
    for k in keys:
        v11[f"clip_{k}"] = scores_hold[k]
    v11["clip_thumb_ok"] = ok
    v11_path = out / "农业采集_0814_semantic_remain_clip_v11_from_emb_0821.csv"
    v11.to_csv(v11_path, index=False)
    print("wrote", v11_path, flush=True)

    print("load old clip_pass...", flush=True)
    old_pass = pd.read_csv(
        out / "农业采集_0814_semantic_remain_clip_pass_0818.csv",
        dtype=str,
        low_memory=False,
    )
    old_pass["video_id"] = old_pass["video_id"].astype(str).str.strip()
    drop_cols = [c for c in old_pass.columns if c.startswith("clip_")]
    base = old_pass.drop(columns=drop_cols, errors="ignore")
    merged = base.merge(v11, on="video_id", how="left")
    merged["clip_decision_v11"] = merged["clip_decision_v11"].fillna("no_thumb")
    merged["clip_decision"] = merged["clip_decision_v11"]

    new_pass = merged[merged["clip_decision"] == "clip_pass"].copy()
    new_fail = merged[merged["clip_decision"] == "clip_fail"].copy()
    new_remain = merged[merged["clip_decision"] != "clip_fail"].copy()

    pass_p = out / "农业采集_0814_clip_pass_v11_on_oldpass_0821.csv"
    fail_p = out / "农业采集_0814_clip_fail_v11_on_oldpass_0821.csv"
    remain_p = out / "农业采集_0814_clip_remain_v11_on_oldpass_0821.csv"
    new_pass.to_csv(pass_p, index=False)
    new_fail.to_csv(fail_p, index=False)
    new_remain.to_csv(remain_p, index=False)

    summary = {
        "input_old_pass": str(out / "农业采集_0814_semantic_remain_clip_pass_0818.csv"),
        "n_old_pass": len(old_pass),
        "n_v11_pass": len(new_pass),
        "n_v11_fail": len(new_fail),
        "n_v11_remain": len(new_remain),
        "n_no_thumb": int((merged["clip_decision"] == "no_thumb").sum()),
        "hours_old_pass": hours(old_pass),
        "hours_v11_pass": hours(new_pass),
        "hours_v11_fail": hours(new_fail),
        "thresholds": cfg["thresholds"],
        "require": require_keys(cfg),
        "pass_csv": str(pass_p),
        "fail_csv": str(fail_p),
        "remain_csv": str(remain_p),
        "v11_full_scored": str(v11_path),
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    (out / "农业采集_0814_clip_v11_on_oldpass_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

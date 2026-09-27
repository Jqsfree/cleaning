#!/usr/bin/env python3
"""从已落盘的 CLIP embedding store 重打 exo_agriculture margin（免重编码缩略图）。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT))

from categories.exo_agriculture.cascade_clip import (  # noqa: E402
    _RULES,
    _prompt_bank,
    decide,
    encode_prompt_pairs,
    load_cfg,
    margins_from_feats,
    q_keys,
    require_keys,
)
from core.exemplar_sim import ClipEncoder  # noqa: E402


def load_store(store_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    index = pd.read_csv(store_dir / "index.csv", dtype={"video_id": str})
    embeddings = np.load(store_dir / "embeddings.npy", mmap_mode="r")
    ok_path = store_dir / "thumb_ok.npy"
    if ok_path.is_file():
        thumb_ok = np.load(ok_path, mmap_mode="r")
    else:
        thumb_ok = ~np.isnan(embeddings[:, 0])
    if len(index) != len(embeddings):
        raise SystemExit(
            f"[ERROR] index 行数 {len(index)} != embeddings {len(embeddings)}"
        )
    return index, embeddings, np.asarray(thumb_ok, dtype=bool)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="exo_agriculture：embedding store + TOML → clip_q* / clip_decision",
    )
    ap.add_argument(
        "--embeddings",
        required=True,
        help="store 目录（含 embeddings.npy / index.csv / thumb_ok.npy）",
    )
    ap.add_argument("-o", "--output", required=True, help="输出 CSV 路径")
    ap.add_argument(
        "--config",
        default=None,
        help="cascade_harvest_clip.toml；默认品类 rules",
    )
    ap.add_argument(
        "--merge-csv",
        default=None,
        help="可选：与原 scored/元数据 CSV 按 video_id join",
    )
    ap.add_argument("--model", default=None, help="覆盖 TOML meta.model")
    ap.add_argument("--pretrained", default=None, help="覆盖 TOML meta.pretrained")
    args = ap.parse_args()

    store = Path(args.embeddings)
    cfg_path = Path(args.config) if args.config else _RULES
    cfg = load_cfg(cfg_path)
    keys = q_keys(cfg)
    model = args.model or cfg["meta"]["model"]
    pretrained = args.pretrained or cfg["meta"]["pretrained"]

    index, embeddings, thumb_ok = load_store(store)
    n = len(index)
    log_t0 = time.perf_counter()
    print(f"store={store} rows={n:,} thumb_ok={int(thumb_ok.sum()):,}")

    encoder = ClipEncoder(model, pretrained)
    pairs = encode_prompt_pairs(encoder, _prompt_bank(cfg))

    # 只对 thumb_ok 行算 margin；其余保持 0 并由 decide → no_thumb
    feats = np.zeros((n, 512), dtype=np.float32)
    ok_idx = np.flatnonzero(thumb_ok)
    if len(ok_idx):
        feats[ok_idx] = np.asarray(embeddings[ok_idx], dtype=np.float32)
        # 已 L2 归一化的 float16 再读回；重新归一化防数值漂移
        norms = np.linalg.norm(feats[ok_idx], axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        feats[ok_idx] = feats[ok_idx] / norms

    scores = margins_from_feats(feats, pairs)
    decision = decide(
        scores,
        thresholds={k: float(cfg["thresholds"][k]) for k in keys},
        require=require_keys(cfg),
        ok_mask=thumb_ok,
    )

    out = pd.DataFrame({"video_id": index["video_id"].astype(str).str.strip()})
    for key in keys:
        out[f"clip_{key}"] = scores[key]
    out["clip_decision"] = decision
    out["clip_thumb_ok"] = thumb_ok

    if args.merge_csv:
        base = pd.read_csv(args.merge_csv, dtype=str, low_memory=False)
        base["video_id"] = base["video_id"].astype(str).str.strip()
        drop_cols = [c for c in base.columns if c.startswith("clip_")]
        base = base.drop(columns=drop_cols, errors="ignore")
        out = base.merge(out, on="video_id", how="left")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    counts = out["clip_decision"].value_counts().to_dict()
    summary = {
        "embeddings": str(store),
        "config": str(cfg_path),
        "output": str(out_path),
        "n": len(out),
        "decision_counts": counts,
        "thresholds": cfg["thresholds"],
        "require": require_keys(cfg),
        "elapsed_sec": round(time.perf_counter() - log_t0, 2),
    }
    sum_path = out_path.with_suffix(".summary.json")
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

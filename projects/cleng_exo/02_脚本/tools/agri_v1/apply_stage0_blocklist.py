#!/usr/bin/env python3
"""Stage 0：对 metadata_pass 再滤 blocklist_keywords（双保险，spec §4）。

Stage -1 已含 blocklist；本脚本对 pass 池做独立 keyword 硬拒并产出 stage0_pass/reject。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).resolve().parent.parent.parent
_REPO = _SCRIPT.parent
sys.path.insert(0, str(_SCRIPT))

from core.metadata_filter import load_blocklist_keywords  # noqa: E402

DEFAULT_BLOCKLIST = _REPO / "models/exo_agriculture_metadata_filter/blocklist_keywords.csv"


def apply_stage0(
    metadata_pass_csv: Path,
    *,
    out_dir: Path,
    blocklist_path: Path,
    chunksize: int = 100_000,
) -> dict:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    blocklist = load_blocklist_keywords(blocklist_path)

    stem = metadata_pass_csv.stem.replace("_metadata_pass", "")
    pass_path = out_dir / "stage0_blocklist_pass.csv"
    reject_path = out_dir / "stage0_blocklist_reject.csv"

    n_in = n_pass = n_reject = 0
    wp = wr = True
    reject_kw: dict[str, int] = {}

    for chunk in pd.read_csv(metadata_pass_csv, chunksize=chunksize, engine="python"):
        n_in += len(chunk)
        kw = chunk["keyword"].fillna("").astype(str).str.strip().str.lower()
        block_mask = kw.isin(blocklist)
        n_reject += int(block_mask.sum())
        n_pass += int((~block_mask).sum())
        for k in kw[block_mask].value_counts().index:
            reject_kw[k] = reject_kw.get(k, 0) + int((kw == k).sum())

        chunk.loc[~block_mask].to_csv(pass_path, mode="w" if wp else "a", header=wp, index=False)
        wp = False
        blocked = chunk.loc[block_mask].copy()
        if len(blocked):
            blocked["stage0_reject_reason"] = "blocklist_keyword"
            blocked.to_csv(reject_path, mode="w" if wr else "a", header=wr, index=False)
            wr = False
        print(f"  cumulative in={n_in:,} pass={n_pass:,} reject={n_reject:,}", flush=True)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "stage0_blocklist",
        "input": str(metadata_pass_csv.resolve()),
        "blocklist_path": str(blocklist_path.resolve()),
        "n_blocklist_keywords": len(blocklist),
        "n_in": n_in,
        "n_pass": n_pass,
        "n_reject": n_reject,
        "reject_rate": round(n_reject / max(n_in, 1), 4),
        "top_reject_keywords": dict(
            sorted(reject_kw.items(), key=lambda x: -x[1])[:30]
        ),
        "pass_path": str(pass_path.resolve()),
        "reject_path": str(reject_path.resolve()) if n_reject else None,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 0 blocklist on metadata_pass")
    ap.add_argument("metadata_pass", type=Path)
    ap.add_argument("-o", "--out-dir", type=Path, required=True)
    ap.add_argument("--blocklist", type=Path, default=DEFAULT_BLOCKLIST)
    ap.add_argument("--chunksize", type=int, default=100_000)
    args = ap.parse_args()
    apply_stage0(
        args.metadata_pass,
        out_dir=args.out_dir,
        blocklist_path=args.blocklist,
        chunksize=args.chunksize,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

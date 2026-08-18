#!/usr/bin/env python3
"""CLI: exo_service L3 CLIP 三问（本地）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT))

from categories.exo_service.cascade_clip import run_l3  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="exo_service L3 CLIP zero-shot (local)")
    ap.add_argument("input", help="L2 candidates CSV")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-n", "--sample", type=int, default=2000, help="抽样条数；0=全量")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default="qc_thumb_cache/exemplar_sim")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--thumb-workers", type=int, default=24)
    ap.add_argument(
        "--config",
        default=None,
        help="cascade_l3_*.toml；默认 categories/exo_service/rules/cascade_l3_clip.toml",
    )
    ap.add_argument("--stem", default="l3_clip", help="输出文件名前缀")
    args = ap.parse_args()
    cfg = Path(args.config) if args.config else None
    s = run_l3(
        args.input,
        args.output,
        n_sample=args.sample,
        seed=args.seed,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        thumb_workers=args.thumb_workers,
        cfg_path=cfg,
        stem=args.stem,
    )
    print(
        f"done  sample={s['n_sample']:,}  pass={s['n_clip_pass']:,}  "
        f"fail={s['n_clip_fail']:,}  no_thumb={s['n_no_thumb']:,}  "
        f"pass_rate={s['pass_rate_among_ok']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

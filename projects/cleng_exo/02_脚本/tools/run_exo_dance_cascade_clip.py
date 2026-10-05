#!/usr/bin/env python3
"""CLI: exo_dance CLIP 单人闸（本地缩略图零样本）。

用法:
  PYTHONPATH=02_脚本 .venv/bin/python3 02_脚本/tools/run_exo_dance_cascade_clip.py \\
    data/runs/exo_dance/machine_0923/06_tools/text_gov_v01/…_ml_keep.csv \\
    -o data/runs/exo_dance/machine_0923/06_tools/clip_solo_v01/ -n 2000

  # 全量
  … -n 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT))

from categories.exo_dance.cascade_clip import run_solo_clip  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="exo_dance CLIP solo dancer gate (local)")
    ap.add_argument("input", help="ml_keep / clean keep CSV")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-n", "--sample", type=int, default=2000, help="抽样条数；0=全量")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default="qc_thumb_cache/exemplar_sim")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--thumb-workers", type=int, default=24)
    ap.add_argument(
        "--config",
        default=None,
        help="cascade_clip.toml；默认 categories/exo_dance/rules/cascade_clip.toml",
    )
    ap.add_argument("--stem", default="clip_solo", help="输出文件名前缀")
    args = ap.parse_args()
    cfg = Path(args.config) if args.config else None
    s = run_solo_clip(
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
        f"pass_rate={s['pass_rate_among_ok']}  "
        f"pass_h={s.get('pass_hours')}  elapsed={s['elapsed_sec']}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""CLI: exo_fitness CLIP 健身动作零样本（本地）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT))

from categories.exo_agriculture.cascade_clip import run_harvest_clip  # noqa: E402

_DEFAULT_CFG = (
    _SCRIPT / "categories" / "exo_fitness" / "rules" / "cascade_fitness_clip.toml"
)


def main() -> int:
    ap = argparse.ArgumentParser(description="exo_fitness CLIP gym workout filter")
    ap.add_argument("input", help="候选 CSV（需 video_id；常用 clean keep）")
    ap.add_argument("-o", "--output", required=True, help="输出目录")
    ap.add_argument("--sample", type=int, default=0, help="抽样；0=全量")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default="qc_thumb_cache/exemplar_sim")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--thumb-workers", type=int, default=16)
    ap.add_argument("--batch-rows", type=int, default=5000, help="分批 checkpoint")
    ap.add_argument("--config", default=str(_DEFAULT_CFG))
    ap.add_argument("--stem", default="harvest_clip", help="harvest_clip=用输入文件名前缀")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--save-embeddings",
        default=None,
        help="可选：写出 float16 embedding store（与 export_clip_embeddings 同结构）",
    )
    args = ap.parse_args()
    s = run_harvest_clip(
        args.input,
        args.output,
        n_sample=args.sample,
        seed=args.seed,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        thumb_workers=args.thumb_workers,
        cfg_path=Path(args.config),
        stem=args.stem,
        batch_rows=args.batch_rows,
        overwrite=args.overwrite,
        save_embeddings=args.save_embeddings,
    )
    print(
        f"done  run={s['n_run']:,}  pass={s['n_clip_pass']:,}  "
        f"fail={s['n_clip_fail']:,}  remain={s['n_clip_remain']:,}  "
        f"no_thumb={s['n_no_thumb']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

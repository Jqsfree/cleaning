#!/usr/bin/env python3
"""CLI: exo_agriculture CLIP 农业场景零样本（本地）。

默认在 CLIP 前跑 Stage -1 metadata 过滤器（阈值 0.3，召回优先省算力）。
可用 --skip-metadata-filter 跳过（输入已是 metadata_pass）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent
_REPO = _SCRIPT.parent
sys.path.insert(0, str(_SCRIPT))

from categories.exo_agriculture.cascade_clip import run_harvest_clip  # noqa: E402
from tools.apply_exo_agriculture_metadata_filter import apply_filter  # noqa: E402

DEFAULT_MODEL_DIR = _REPO / "models/exo_agriculture_metadata_filter"
DEFAULT_EMBEDDING_STORE = _REPO / "data/assets/embeddings/exo_agriculture_0814_semantic_remain"


def main() -> int:
    ap = argparse.ArgumentParser(description="exo_agriculture metadata filter + CLIP")
    ap.add_argument("input", help="候选 CSV（需 video_id；常用 semantic_remain 或 quality keep）")
    ap.add_argument("-o", "--output", required=True, help="输出目录（06_tools/）")
    ap.add_argument("--sample", type=int, default=0, help="CLIP 抽样条数；0=全量")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", default="qc_thumb_cache/exemplar_sim")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--thumb-workers", type=int, default=16)
    ap.add_argument("--batch-rows", type=int, default=5000)
    ap.add_argument("--config", default=None)
    ap.add_argument("--stem", default="harvest_clip")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save-embeddings", default=None)
    ap.add_argument(
        "--embedding-store",
        type=Path,
        default=DEFAULT_EMBEDDING_STORE,
        help="只读 embedding store；命中则跳过 encode_images（默认 machine_0814 store）",
    )
    ap.add_argument(
        "--no-embedding-store",
        action="store_true",
        help="禁用 embedding store，全部现场 encode（调试用）",
    )
    ap.add_argument(
        "--skip-metadata-filter", action="store_true",
        help="跳过 Stage -1（输入已是 metadata_pass）",
    )
    ap.add_argument("--metadata-model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--metadata-threshold", type=float, default=0.3)
    ap.add_argument("--metadata-sample-reject", type=int, default=100)
    ap.add_argument("--metadata-eval-labels", type=Path, default=None)
    ap.add_argument(
        "--metadata-halt-on-high-t-hurt", action="store_true",
        help="有人工标且 metadata T误杀超限时中止（不跑 CLIP）",
    )
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    clip_input = Path(args.input)
    meta_summary = None

    if not args.skip_metadata_filter:
        meta_dir = out / "metadata_filter"
        print(f"[metadata] Stage -1 → {meta_dir}", flush=True)
        meta_summary = apply_filter(
            Path(args.input),
            out_dir=meta_dir,
            model_dir=args.metadata_model_dir,
            blocklist_path=args.metadata_model_dir / "blocklist_keywords.csv",
            threshold=args.metadata_threshold,
            chunksize=50_000,
            sample_reject=args.metadata_sample_reject,
            seed=args.seed,
            eval_labels=args.metadata_eval_labels,
            halt_on_high_t_hurt=args.metadata_halt_on_high_t_hurt,
        )
        pass_files = list(meta_dir.glob("*_metadata_pass.csv"))
        if not pass_files:
            print("[ERROR] metadata_filter 未产出 pass 文件", file=sys.stderr)
            return 2
        clip_input = pass_files[0]

    t0 = time.perf_counter()
    cfg = Path(args.config) if args.config else None
    s = run_harvest_clip(
        str(clip_input),
        args.output,
        n_sample=args.sample,
        seed=args.seed,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        thumb_workers=args.thumb_workers,
        cfg_path=cfg,
        stem=args.stem,
        batch_rows=args.batch_rows,
        overwrite=args.overwrite,
        save_embeddings=args.save_embeddings,
        embedding_store=None if args.no_embedding_store else args.embedding_store,
    )
    pipeline = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(Path(args.input).resolve()),
        "metadata_filter": meta_summary,
        "clip_input": str(clip_input.resolve()),
        "clip_summary": s,
        "n_candidates": meta_summary["n_candidates"] if meta_summary else s.get("n_pool"),
        "n_metadata_reject": meta_summary["n_metadata_reject"] if meta_summary else 0,
        "n_clip_processed": s.get("n_run"),
        "clip_saved_by_metadata": meta_summary["clip_candidates_saved"] if meta_summary else 0,
        "elapsed_clip_sec": round(time.perf_counter() - t0, 1),
    }
    pipe_path = out / "pipeline_metadata_clip_summary.json"
    pipe_path.write_text(json.dumps(pipeline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"done  metadata_reject={pipeline['n_metadata_reject']:,}  "
        f"clip_run={s['n_run']:,}  pass={s['n_clip_pass']:,}  "
        f"fail={s['n_clip_fail']:,}  remain={s['n_clip_remain']:,}  "
        f"no_thumb={s['n_no_thumb']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

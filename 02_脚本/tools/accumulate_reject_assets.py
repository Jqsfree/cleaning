#!/usr/bin/env python3
"""
tools/accumulate_reject_assets.py — 批次后一条命令累计 text+thumb 排除资产

不进默认 pipeline。顺序：text 提案 → thumb 提案 → export。

用法:
  02_脚本/tools/accumulate_reject_assets.py -o $BATCH/ --category film_tv
  02_脚本/tools/accumulate_reject_assets.py -o $BATCH/ --category film_tv \\
    --text-input drop.csv --thumb-input thumb_qc.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.run_manifest import load_manifest, update_stage  # noqa: E402
from tools.export_reject_assets import DEFAULT_ASSETS, export_assets  # noqa: E402
from tools.propose_reject_tags import (  # noqa: E402
    _read_table,
    propose_from_frame,
    propose_from_thumb,
    write_proposed,
)


def _find_drop_csvs(batch_root: Path) -> list[Path]:
    clean = batch_root / "05_clean"
    if not clean.exists():
        return []
    found = sorted(clean.glob("**/drop*.csv"))
    return found


def _find_thumb_csvs(batch_root: Path) -> list[Path]:
    qc = batch_root / "03_qc"
    cands: list[Path] = []
    for pat in ("*thumb_qc*.csv", "*vision_thumb*.csv", "*_thumb*.csv"):
        cands.extend(qc.glob(pat))
    # 也扫 06_tools
    tools = batch_root / "06_tools"
    if tools.exists():
        for pat in ("*thumb_qc*.csv", "*vision_thumb*.csv"):
            cands.extend(tools.glob(pat))
    # 去重
    return sorted({p.resolve() for p in cands})


def main() -> None:
    p = argparse.ArgumentParser(description="累计 text+thumb 排除提案并 export 资产")
    p.add_argument("-o", "--batch-root", required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--text-input", default=None, help="显式 drop/scored CSV")
    p.add_argument("--thumb-input", default=None, help="显式 vision_thumb 结果 CSV")
    p.add_argument("--from-ml", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--assets-root", default=str(DEFAULT_ASSETS))
    p.add_argument("--cascade", action="store_true",
                   help="额外跑级联抽样队列（cascade_reject_propose）")
    args = p.parse_args()

    batch_root = Path(args.batch_root)
    batch_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── text ──
    text_paths: list[Path] = []
    if args.text_input:
        text_paths = [Path(args.text_input)]
    else:
        text_paths = _find_drop_csvs(batch_root)

    n_text = 0
    for tp in text_paths:
        if not tp.exists():
            print(f"[WARN] 跳过不存在: {tp}")
            continue
        df = _read_table(str(tp))
        out = propose_from_frame(
            df, category=args.category, from_ml=args.from_ml, modality="text",
        )
        write_proposed(batch_root, out, merge=True)
        n_text += len(out)
        print(f"[{time.strftime('%H:%M:%S')}] text 提案 +{len(out):,} ← {tp.name}")

    if not text_paths:
        print(f"[{time.strftime('%H:%M:%S')}] 无 text 输入（无 --text-input / 05_clean/drop*.csv）")

    # ── thumb ──
    thumb_paths: list[Path] = []
    if args.thumb_input:
        thumb_paths = [Path(args.thumb_input)]
    else:
        thumb_paths = _find_thumb_csvs(batch_root)

    n_thumb = 0
    for tp in thumb_paths:
        if not tp.exists():
            print(f"[WARN] 跳过不存在: {tp}")
            continue
        df = _read_table(str(tp))
        out = propose_from_thumb(df, category=args.category)
        write_proposed(batch_root, out, merge=True)
        n_thumb += len(out)
        print(f"[{time.strftime('%H:%M:%S')}] thumb 提案 +{len(out):,} ← {tp.name}")

    if not thumb_paths:
        print(f"[{time.strftime('%H:%M:%S')}] 无 thumb 输入（无 --thumb-input / *thumb_qc*）")

    # ── cascade sample（可选）──
    if args.cascade:
        from tools.cascade_reject_propose import run_cascade
        n_sample = run_cascade(batch_root, category=args.category)
        print(f"[{time.strftime('%H:%M:%S')}] cascade 抽样队列: {n_sample:,}")

    # ── export ──
    stats = {}
    if not args.skip_export:
        stats = export_assets(
            batch_roots=[batch_root],
            assets_root=Path(args.assets_root),
        )

    prop_path = batch_root / "03_qc" / "reject_proposed.csv"
    n_all = len(pd.read_csv(prop_path)) if prop_path.exists() else 0

    print()
    print("=" * 56)
    print("  排除资产累计完成")
    print("=" * 56)
    print(f"  text 新增提案:  {n_text:,}")
    print(f"  thumb 新增提案: {n_thumb:,}")
    print(f"  proposed 合计:  {n_all:,}")
    print(f"  export tags:    {len(stats)}")
    print(f"  耗时:           {time.time() - t0:.1f}s")
    print("=" * 56)

    if load_manifest(batch_root):
        try:
            update_stage(
                batch_root,
                "reject_accumulate",
                paths={"reject_proposed": str(prop_path)},
                stats={
                    "n_text": n_text,
                    "n_thumb": n_thumb,
                    "n_proposed": n_all,
                    "n_export_tags": len(stats),
                },
            )
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()

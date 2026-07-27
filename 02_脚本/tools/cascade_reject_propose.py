#!/usr/bin/env python3
"""
tools/cascade_reject_propose.py — text → thumb → 人工抽样 级联

不自动全量 drop；只写 proposed 与 03_qc/reject_sample_for_validate.csv。

用法:
  02_脚本/tools/cascade_reject_propose.py -o $BATCH/ --category film_tv
  02_脚本/tools/cascade_reject_propose.py -o $BATCH/ --category film_tv \\
    --universe keep.csv --thumb-input thumb_qc.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.reject_modality import load_cascade_cfg  # noqa: E402
from core.reject_taxonomy import normalize_tags  # noqa: E402
from core.run_manifest import load_manifest, update_stage  # noqa: E402
from tools.propose_reject_tags import (  # noqa: E402
    _read_table,
    propose_from_frame,
    propose_from_thumb,
    write_proposed,
)


def _first_tag(s: str) -> str:
    tags = normalize_tags(s)
    return tags[0] if tags else ""


def run_cascade(
    batch_root: Path,
    *,
    category: str,
    universe: Path | None = None,
    thumb_input: Path | None = None,
    text_input: Path | None = None,
    from_ml: bool = False,
    n_validate: int | None = None,
) -> int:
    """
    L1: 对 universe/text_input 做 text 提案（high 写入 proposed）
    L2: mid / 未命中 且有 thumb → thumb 提案
    L3: mid、冲突、或仅单模态 mid → 抽样队列
    返回抽样行数。
    """
    cfg = load_cascade_cfg()
    n_val = int(n_validate if n_validate is not None else cfg.get("sampling", {}).get("n_validate", 100))
    seed = int(cfg.get("sampling", {}).get("seed", 42))

    qc = batch_root / "03_qc"
    qc.mkdir(parents=True, exist_ok=True)

    # 解析输入
    if text_input and text_input.exists():
        text_df = _read_table(str(text_input))
    elif universe and universe.exists():
        text_df = _read_table(str(universe))
    else:
        # 尝试 keep / quality
        cands = list((batch_root / "05_clean").glob("**/keep*.csv")) if (batch_root / "05_clean").exists() else []
        cands += list((batch_root / "01_quality").glob("**/*quality*.csv")) if (batch_root / "01_quality").exists() else []
        if not cands:
            # 仅基于已有 proposed 做冲突/抽样
            text_df = pd.DataFrame()
        else:
            text_df = _read_table(str(cands[0]))

    thumb_df = pd.DataFrame()
    if thumb_input and thumb_input.exists():
        thumb_df = _read_table(str(thumb_input))
    else:
        for pat in ("*thumb_qc*.csv", "*vision_thumb*.csv"):
            hits = list(qc.glob(pat)) + list((batch_root / "06_tools").glob(pat)) if (batch_root / "06_tools").exists() else list(qc.glob(pat))
            if hits:
                thumb_df = _read_table(str(hits[0]))
                break

    text_prop = pd.DataFrame()
    if not text_df.empty:
        text_prop = propose_from_frame(
            text_df, category=category, from_ml=from_ml, modality="text",
        )
        # high 写入；mid 进队列逻辑
        high = text_prop[text_prop["confidence_band"] == "high"] if len(text_prop) else text_prop
        write_proposed(batch_root, high, merge=True)

    thumb_prop = pd.DataFrame()
    if not thumb_df.empty:
        thumb_prop = propose_from_thumb(thumb_df, category=category)
        write_proposed(batch_root, thumb_prop, merge=True)

    # 按 video_id 对齐
    t_map: dict[str, str] = {}
    t_band: dict[str, str] = {}
    if len(text_prop):
        for _, r in text_prop.iterrows():
            vid = str(r["video_id"])
            t_map[vid] = _first_tag(r["reject_tags"])
            t_band[vid] = str(r.get("confidence_band", "high"))

    h_map: dict[str, str] = {}
    if len(thumb_prop):
        for _, r in thumb_prop.iterrows():
            h_map[str(r["video_id"])] = _first_tag(r["reject_tags"])

    # fusion：一致则回写 modality=fusion 标记行
    fusion_rows = []
    sample_rows = []
    all_ids = set(t_map) | set(h_map)

    # mid text 无 thumb
    for vid, band in t_band.items():
        if band == "mid":
            sample_rows.append({
                "video_id": vid,
                "reject_tags": t_map.get(vid, ""),
                "cascade_reason": "text_mid_band",
                "text_tag": t_map.get(vid, ""),
                "thumb_tag": h_map.get(vid, ""),
                "modality_hint": "text",
            })

    for vid in all_ids:
        tt, ht = t_map.get(vid, ""), h_map.get(vid, "")
        if tt and ht:
            if tt == ht:
                fusion_rows.append({
                    "video_id": vid,
                    "reject_tags": tt,
                    "propose_source": "fusion:text+thumb",
                    "confidence": "agree",
                    "confidence_band": "high",
                    "modality": "fusion",
                    "label_source": "proposed",
                    "registry_version": "",
                    "pipeline_category": category,
                })
            else:
                sample_rows.append({
                    "video_id": vid,
                    "reject_tags": tt,
                    "cascade_reason": "modality_conflict",
                    "text_tag": tt,
                    "thumb_tag": ht,
                    "modality_hint": "conflict",
                })

    if fusion_rows:
        write_proposed(batch_root, pd.DataFrame(fusion_rows), merge=True)

    sample_df = pd.DataFrame(sample_rows)
    if sample_df.empty:
        out_sample = qc / "reject_sample_for_validate.csv"
        pd.DataFrame(columns=[
            "video_id", "reject_tags", "cascade_reason", "text_tag", "thumb_tag", "modality_hint",
        ]).to_csv(out_sample, index=False)
        return 0

    # 去重后抽样
    sample_df = sample_df.drop_duplicates(subset=["video_id"], keep="first")
    if len(sample_df) > n_val:
        sample_df = sample_df.sample(n=n_val, random_state=seed)

    out_sample = qc / "reject_sample_for_validate.csv"
    sample_df.to_csv(out_sample, index=False)
    return len(sample_df)


def main() -> None:
    p = argparse.ArgumentParser(description="排除类级联提案 + 抽样验证队列")
    p.add_argument("-o", "--batch-root", required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--universe", default=None, help="候选全集 CSV（keep/quality）")
    p.add_argument("--text-input", default=None)
    p.add_argument("--thumb-input", default=None)
    p.add_argument("--from-ml", action="store_true")
    p.add_argument("-n", "--n-validate", type=int, default=None)
    args = p.parse_args()

    batch_root = Path(args.batch_root)
    n = run_cascade(
        batch_root,
        category=args.category,
        universe=Path(args.universe) if args.universe else None,
        thumb_input=Path(args.thumb_input) if args.thumb_input else None,
        text_input=Path(args.text_input) if args.text_input else None,
        from_ml=args.from_ml,
        n_validate=args.n_validate,
    )
    print()
    print("=" * 56)
    print("  排除类级联完成")
    print("=" * 56)
    print(f"  抽样验证队列: {n:,}")
    print(f"  → {batch_root / '03_qc' / 'reject_sample_for_validate.csv'}")
    print("  说明: 不自动 drop；人工 confirm/correct 后 ingest")
    print("=" * 56)

    if load_manifest(batch_root):
        try:
            update_stage(
                batch_root,
                "reject_cascade",
                paths={
                    "reject_proposed": str(batch_root / "03_qc" / "reject_proposed.csv"),
                    "reject_sample_for_validate": str(
                        batch_root / "03_qc" / "reject_sample_for_validate.csv"
                    ),
                },
                stats={"n_sample": n},
            )
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()

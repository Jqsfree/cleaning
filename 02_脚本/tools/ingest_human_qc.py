#!/usr/bin/env python3
"""
tools/ingest_human_qc.py — 人工质检结果入库

将外部标注 CSV/Parquet 规范写入批次 03_qc/，并更新 manifest。

用法:
  02_脚本/tools/ingest_human_qc.py labels.csv -o $BATCH/ \\
    --category film_tv --source human --batch 0724
  02_脚本/tools/ingest_human_qc.py labels.csv -o $BATCH/ \\
    --category film_tv --source machine --batch 0727 --dimension text
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.human_qc import (  # noqa: E402
    human_validated_rejects,
    normalize_frame,
    pass_rate,
    split_pass_fail,
    train_export_columns,
)
from core.run_manifest import load_manifest, update_stage  # noqa: E402


def _read_table(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def main() -> None:
    p = argparse.ArgumentParser(
        description="人工质检结果入库 → 03_qc/{pass,fail,labeled,train_export}.csv",
    )
    p.add_argument("input", help="人工标注 CSV/Parquet")
    p.add_argument(
        "-o", "--batch-root", required=True,
        help="批次根目录（写入 03_qc/）",
    )
    p.add_argument("--category", required=True)
    p.add_argument("--source", required=True, choices=("human", "machine"))
    p.add_argument("--batch", required=True)
    p.add_argument(
        "--dimension", default="overall",
        help="质检维度（默认 overall；如 text/thumb/storyboard）",
    )
    p.add_argument("--label-col", default=None, help="标签列名（默认自动检测）")
    p.add_argument("--id-col", default=None, help="video_id 列名（默认自动检测）")
    p.add_argument(
        "--reject-col", default=None,
        help="排除类列（可选；缺省不要求——人工不做全量细类标注）",
    )
    p.add_argument(
        "--reject-action-col", default=None,
        help="排除验证动作列 confirm|correct（可选）",
    )
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"[ERROR] 文件不存在: {inp}")
        sys.exit(1)

    batch_root = Path(args.batch_root)
    qc_dir = batch_root / "03_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    labeled_at = time.strftime("%Y-%m-%d %H:%M:%S")
    raw = _read_table(str(inp))
    print(f"[{time.strftime('%H:%M:%S')}] 读取 {inp}  ({len(raw):,} 行)")

    try:
        labeled = normalize_frame(
            raw,
            category=args.category,
            source=args.source,
            batch=args.batch,
            dimension=args.dimension,
            label_col=args.label_col,
            id_col=args.id_col,
            reject_col=args.reject_col,
            reject_action_col=args.reject_action_col,
            labeled_at=labeled_at,
        )
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(2)

    dropped = int(labeled.attrs.get("dropped_unmapped", 0) or 0)
    if dropped:
        print(f"  跳过无法映射标签: {dropped:,} 行")

    pass_df, fail_df = split_pass_fail(labeled)
    train_df = train_export_columns(labeled)
    validated = human_validated_rejects(labeled)

    paths = {
        "pass": str(qc_dir / "pass.csv"),
        "fail": str(qc_dir / "fail.csv"),
        "labeled": str(qc_dir / "labeled.csv"),
        "train_export": str(qc_dir / "train_export.csv"),
    }
    pass_df.to_csv(paths["pass"], index=False)
    fail_df.to_csv(paths["fail"], index=False)
    labeled.to_csv(paths["labeled"], index=False)
    train_df.to_csv(paths["train_export"], index=False)

    if len(validated):
        vpath = qc_dir / "reject_validated.csv"
        validated.to_csv(vpath, index=False)
        paths["reject_validated"] = str(vpath)

    rate = pass_rate(labeled)
    n = len(labeled)
    n_pass = len(pass_df)
    n_fail = len(fail_df)
    n_rej = len(validated)

    print()
    print("=" * 56)
    print("  人工质检入库完成")
    print("=" * 56)
    print(f"  有效标注:   {n:>10,}")
    print(f"  pass:       {n_pass:>10,}  ({n_pass / max(n, 1) * 100:5.1f}%)")
    print(f"  fail:       {n_fail:>10,}  ({n_fail / max(n, 1) * 100:5.1f}%)")
    print(f"  ★ 人工合格率 (唯一 KPI): {rate * 100:.1f}%")
    print(f"  排除类验证金标: {n_rej:,}  （可选；非全量打标）")
    print(f"  输出:       {qc_dir}/")
    for k, v in paths.items():
        print(f"              {k:12s} → {Path(v).name}")
    print("=" * 56)

    if load_manifest(batch_root):
        try:
            update_stage(
                batch_root,
                "human_qc",
                paths=paths,
                stats={
                    "n_labeled": n,
                    "n_pass": n_pass,
                    "n_fail": n_fail,
                    "n_reject_validated": n_rej,
                    "pass_rate": round(rate, 4),
                    "dimension": args.dimension,
                    "input": str(inp.resolve()),
                },
            )
            print(f"  manifest 已更新 stage=human_qc")
        except FileNotFoundError:
            pass
    else:
        print("  (无 manifest，跳过更新；可用 run_manifest.py init)")


if __name__ == "__main__":
    main()

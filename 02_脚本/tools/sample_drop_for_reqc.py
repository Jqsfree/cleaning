#!/usr/bin/env python3
"""
tools/sample_drop_for_reqc.py — 从 clean drop 池抽样回流人工复检

闭环：自动过滤只丢「确定噪声」；drop 中可能含误杀，需抽样回流人工。
人工标注完成后用 ingest_human_qc.py 入库。

用法:
  02_脚本/tools/sample_drop_for_reqc.py $BATCH/05_clean/run01/drop.csv \\
    -o $BATCH/ --n 200
  02_脚本/tools/sample_drop_for_reqc.py drop.csv --batch-root $BATCH -n 200
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.run_manifest import load_manifest, update_stage  # noqa: E402

# 与 03_sample 一致的样本量公式
Z_TABLE = {90: 1.645, 95: 1.96, 99: 2.576}


def calc_sample_size(
    n_total: int,
    confidence: int = 95,
    margin: float = 0.05,
    p: float = 0.5,
) -> int:
    z = Z_TABLE.get(confidence, 1.96)
    n_inf = (z ** 2) * p * (1 - p) / (margin ** 2)
    n_adj = n_inf / (1 + (n_inf - 1) / n_total) if n_total > 0 else n_inf
    return max(1, round(n_adj))


def _read_table(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def resolve_out_dir(batch_root: str | None, output: str | None) -> Path:
    """优先写到 {batch}/03_qc/drop_reflux/。"""
    if batch_root:
        return Path(batch_root) / "03_qc" / "drop_reflux"
    if output:
        out = Path(output)
        # 若给的是批次根（含 01_quality 或 manifest），落到 drop_reflux
        if (out / "manifest.json").exists() or (out / "01_quality").is_dir():
            return out / "03_qc" / "drop_reflux"
        return out
    raise ValueError("必须指定 --batch-root 或 -o")


def sample_drop(
    df: pd.DataFrame,
    *,
    n: int | None = None,
    confidence: int = 95,
    margin: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    n_total = len(df)
    if n_total == 0:
        return df.copy()
    if n is not None:
        sample_n = min(n, n_total)
    else:
        sample_n = min(calc_sample_size(n_total, confidence, margin), n_total)
    return df.sample(n=sample_n, random_state=seed).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="从 drop 池抽样回流人工复检",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            产出写入 03_qc/drop_reflux/sample.csv（需人工标注）。
            标注完成后:
              02_脚本/tools/ingest_human_qc.py 03_qc/drop_reflux/labeled.csv \\
                -o $BATCH/ --category … --source … --batch … --dimension overall
        """),
    )
    p.add_argument("input", help="clean drop CSV/Parquet（或任意 drop 池）")
    p.add_argument(
        "-o", "--output", default=None,
        help="输出目录或批次根；默认与 --batch-root 联用",
    )
    p.add_argument(
        "--batch-root", default=None,
        help="批次根目录（优先；写出 03_qc/drop_reflux/）",
    )
    p.add_argument("-n", "--sample-size", type=int, default=None)
    p.add_argument("--confidence", type=int, default=95, choices=[90, 95, 99])
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"[ERROR] 文件不存在: {inp}")
        sys.exit(1)

    try:
        out_dir = resolve_out_dir(args.batch_root, args.output)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)
    note_path = out_dir / "README.txt"
    sample_path = out_dir / "sample.csv"

    df = _read_table(str(inp))
    n_total = len(df)
    print(f"[{time.strftime('%H:%M:%S')}] 输入 drop 池: {inp}  ({n_total:,} 行)")

    sampled = sample_drop(
        df,
        n=args.sample_size,
        confidence=args.confidence,
        margin=args.margin,
        seed=args.seed,
    )
    sampled.to_csv(sample_path, index=False)

    note = (
        "drop 回流抽样 — 需人工复检\n"
        f"来源: {inp.resolve()}\n"
        f"总体: {n_total}\n"
        f"样本: {len(sampled)}\n"
        f"seed: {args.seed}\n"
        f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "\n"
        "标注后请写入 human_label (pass|fail) 或等价列，再执行:\n"
        "  02_脚本/tools/ingest_human_qc.py <labeled.csv> -o <batch>/ \\\n"
        "    --category … --source … --batch …\n"
    )
    note_path.write_text(note, encoding="utf-8")

    print()
    print("=" * 56)
    print("  drop 回流抽样完成")
    print("=" * 56)
    print(f"  总体:       {n_total:>10,}")
    print(f"  样本:       {len(sampled):>10,}")
    print(f"  输出:       {sample_path}")
    print(f"  说明:       {note_path}")
    print("  → 人工标注后用 ingest_human_qc.py 入库")
    print("=" * 56)

    batch_root = Path(args.batch_root) if args.batch_root else None
    if batch_root is None and args.output:
        cand = Path(args.output)
        if (cand / "manifest.json").exists():
            batch_root = cand
        elif (cand.parent.parent / "manifest.json").exists():
            # …/03_qc/drop_reflux → 批次根
            batch_root = cand.parent.parent

    if batch_root and load_manifest(batch_root):
        try:
            update_stage(
                batch_root,
                "drop_reflux",
                paths={"sample": str(sample_path), "note": str(note_path)},
                stats={
                    "n_pool": n_total,
                    "n_sample": len(sampled),
                    "seed": args.seed,
                    "input": str(inp.resolve()),
                },
            )
            print(f"  manifest 已更新 stage=drop_reflux")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()

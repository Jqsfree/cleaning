#!/usr/bin/env python3
"""
tools/lot_accept.py — 交批登记 + 验证已有结果（默认不加第二轮人工）

用法:
  # 1) 交 remain → 07_deliver
  02_脚本/tools/lot_accept.py prepare -o $BATCH/ \\
    --lot-csv $BATCH/06_tools/xxx_after_highconf_drop.csv \\
    --frame remain --deliver-name yb01_deliver_remain.csv

  # 2) 默认：验证已有 human_qc + overturn（不加新人工）
  02_脚本/tools/lot_accept.py verify -o $BATCH/ --min-pass-rate 0.85

  # 可选：客户要求 remain 独立 CI 时才 sample + decide --labeled
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.lot_accept import (  # noqa: E402
    decide,
    prepare_deliver,
    record_sample,
    verify_existing,
)
from core.run_manifest import load_manifest  # noqa: E402

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_SAMPLE_PY = _SCRIPT_DIR / "pipeline" / "03_sample.py"


def _cmd_prepare(args: argparse.Namespace) -> None:
    meta = prepare_deliver(
        args.batch_root,
        lot_csv=args.lot_csv,
        sample_frame=args.frame,
        deliver_name=args.deliver_name,
        method=args.method,
        notes=args.notes,
    )
    print(
        f"deliver ok  lot_size={meta['lot_size']:,}  frame={meta['sample_frame']}  "
        f"method={meta['method']}  decision={meta['decision']}"
    )
    print(f"  → {load_manifest(args.batch_root).get('deliver_path')}")
    print("下一步: lot_accept verify（复用已有抽检，不加新人工）")


def _cmd_verify(args: argparse.Namespace) -> None:
    meta = verify_existing(
        args.batch_root,
        min_pass_rate=args.min_pass_rate,
        max_overturn=args.max_overturn,
    )
    print(
        f"decision={meta['decision']}  method={meta['method']}  "
        f"n={meta.get('n')}  pass_rate={meta.get('pass_rate')}  "
        f"overturn={meta.get('overturn_rate')}"
    )
    print(f"  evidence_frame={meta.get('evidence_frame')}  "
          f"deliver_frame={meta.get('deliver_frame')}")


def _cmd_sample(args: argparse.Namespace) -> None:
    data = load_manifest(args.batch_root)
    if not data or not data.get("lot"):
        print("[ERROR] 请先 lot_accept prepare")
        sys.exit(2)
    prev_sample = dict((data.get("stages") or {}).get("sample") or {})

    lot_path = (
        (data.get("stages") or {}).get("lot_accept", {}).get("paths", {}).get("lot")
        or data.get("deliver_path")
    )
    if args.lot_csv:
        lot_path = args.lot_csv
    if not lot_path or not Path(lot_path).is_file():
        print(f"[ERROR] lot 文件不存在: {lot_path}")
        sys.exit(1)

    out_dir = Path(args.batch_root) / "02_sample" / "lot_accept"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_SAMPLE_PY),
        str(lot_path),
        "-o",
        str(out_dir),
        "--confidence",
        str(args.confidence),
        "--margin",
        str(args.margin),
        "--seed",
        str(args.seed),
    ]
    if args.sample_size:
        cmd.extend(["-n", str(args.sample_size)])
    print(f"[lot_accept] 可选 remain 独立抽样 frame={data['lot'].get('deliver_frame')} …")
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        sys.exit(r.returncode)

    csvs = sorted(
        (p for p in out_dir.glob("*sample*.csv") if "audit" not in p.name.lower()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not csvs:
        print(f"[ERROR] 未找到抽样 CSV: {out_dir}")
        sys.exit(1)
    sample_csv = csvs[0]
    meta = record_sample(args.batch_root, sample_csv=sample_csv, lot_csv=lot_path)

    if prev_sample:
        from core.run_manifest import load_manifest as _load, save_manifest

        cur = _load(args.batch_root)
        stages = cur.setdefault("stages", {})
        stages["sample"] = prev_sample
        save_manifest(args.batch_root, cur)

    print(f"sample ok  n={meta.get('n')}  → {sample_csv}")
    print("（可选路径）标完后: lot_accept decide --labeled …；默认请用 verify")


def _cmd_decide(args: argparse.Namespace) -> None:
    data = load_manifest(args.batch_root) or {}
    meta = decide(
        args.batch_root,
        labeled_csv=args.labeled,
        method=args.method,
        min_pass_rate=args.min_pass_rate,
        aql=args.aql,
        ac=args.ac,
        re=args.re,
        category=args.category or str(data.get("category") or "live_sell"),
        source=args.source or str(data.get("source") or "human"),
        batch=args.batch or str(data.get("batch") or ""),
        id_col=args.id_col,
        label_col=args.label_col,
    )
    print(
        f"decision={meta['decision']}  method={meta['method']}  "
        f"n={meta.get('n')}  pass_rate={meta.get('pass_rate')}"
    )


def _cmd_show(args: argparse.Namespace) -> None:
    data = load_manifest(args.batch_root)
    if not data:
        print(f"[ERROR] 无 manifest: {args.batch_root}")
        sys.exit(1)
    lot = data.get("lot")
    if not lot:
        print("(无 lot 元数据；请先 prepare)")
        return
    for k in (
        "lot_id", "lot_size", "sample_frame", "evidence_frame", "deliver_frame",
        "method", "n", "pass_rate", "min_pass_rate", "overturn_rate",
        "max_overturn", "ac", "re", "aql", "decision", "notes",
    ):
        if k in lot and lot[k] is not None:
            print(f"{k}={lot[k]}")
    print(f"deliver_path={data.get('deliver_path')}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="lot：交 remain + verify 已有结果；默认不加第二轮人工",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="拷 lot → 07_deliver 并登记 manifest.lot")
    p_prep.add_argument("-o", "--batch-root", required=True)
    p_prep.add_argument("--lot-csv", required=True)
    p_prep.add_argument(
        "--frame", required=True,
        choices=("quality", "remain", "clean_keep", "human_pass"),
    )
    p_prep.add_argument("--deliver-name", required=True)
    p_prep.add_argument(
        "--method", default="prescreen_plus_screening",
        choices=("ci_estimate", "aql", "prescreen_plus_screening"),
    )
    p_prep.add_argument("--notes", default="")

    p_v = sub.add_parser(
        "verify",
        help="默认：复用 human_qc + overturn 写判决（不加新人工）",
    )
    p_v.add_argument("-o", "--batch-root", required=True)
    p_v.add_argument("--min-pass-rate", type=float, default=0.85)
    p_v.add_argument("--max-overturn", type=float, default=0.08)

    p_s = sub.add_parser(
        "sample",
        help="可选：remain 独立 CI 抽样（默认流程不需要）",
    )
    p_s.add_argument("-o", "--batch-root", required=True)
    p_s.add_argument("--lot-csv", default=None)
    p_s.add_argument("--confidence", type=int, default=90, choices=(90, 95, 99))
    p_s.add_argument("--margin", type=float, default=0.05)
    p_s.add_argument("-n", "--sample-size", type=int, default=None)
    p_s.add_argument("--seed", type=int, default=42)

    p_d = sub.add_parser(
        "decide",
        help="可选：独立标注表 → accept/reject（默认用 verify）",
    )
    p_d.add_argument("-o", "--batch-root", required=True)
    p_d.add_argument("--labeled", required=True)
    p_d.add_argument("--method", default="ci_estimate", choices=("ci_estimate", "aql"))
    p_d.add_argument("--min-pass-rate", type=float, default=0.85)
    p_d.add_argument("--aql", type=float, default=None)
    p_d.add_argument("--ac", type=int, default=None)
    p_d.add_argument("--re", type=int, default=None)
    p_d.add_argument("--category", default=None)
    p_d.add_argument("--source", default=None, choices=("human", "machine"))
    p_d.add_argument("--batch", default=None)
    p_d.add_argument("--id-col", default=None)
    p_d.add_argument("--label-col", default=None)

    p_show = sub.add_parser("show", help="打印 manifest.lot")
    p_show.add_argument("-o", "--batch-root", required=True)

    args = p.parse_args()
    if args.cmd == "prepare":
        _cmd_prepare(args)
    elif args.cmd == "verify":
        _cmd_verify(args)
    elif args.cmd == "sample":
        _cmd_sample(args)
    elif args.cmd == "decide":
        _cmd_decide(args)
    elif args.cmd == "show":
        _cmd_show(args)


if __name__ == "__main__":
    main()

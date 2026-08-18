#!/usr/bin/env python3
"""评估机采文本过滤是否到达可写规则边界。

用法:
  02_脚本/tools/eval_text_boundary.py propose \\
      --qc 03_qc/run01/*textqc*.csv --rules-dir 02_脚本/categories/parent_child/rules \\
      -o 04_rules/

  02_脚本/tools/eval_text_boundary.py check \\
      --qc-fresh 03_qc/run02/*textqc*.csv --qc-fresh 03_qc/run02b/*textqc*.csv \\
      --qc-all 03_qc/run01/*textqc*.csv --qc-all 03_qc/run02/*textqc*.csv \\
      --rules-dir 02_脚本/categories/parent_child/rules \\
      -o 04_rules/ --category parent_child
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.text_boundary import (  # noqa: E402
    CONSECUTIVE_DEFAULT,
    MIN_F_DEFAULT,
    declare_boundary,
    evaluate_sample,
    load_qc_rows,
    proposed_to_toml,
    write_report,
)

# 品类政策：这些 GENERIC 过闸也不写入，边界判定时跳过（否则永远到不了 0）
POLICY_SKIP: dict[str, set[str]] = {
    "exo_factory": {"how_to_tutorial"},
    "exo_fitness": {"how_to_tutorial"},
    "exo_agriculture": {"how_to_tutorial"},
    "unbox": {"how_to_tutorial"},
    "exo_livestock": set(),
    "parent_child": set(),
    "exo_service": set(),
}


def _eval_kwargs(args: argparse.Namespace) -> dict:
    skip = set(args.skip_name or [])
    skip |= POLICY_SKIP.get(args.category or "", set())
    return {
        "min_f": args.min_f,
        "mine": not args.no_mine,
        "skip_names": skip,
    }


def _paths(values: list[str]) -> list[Path]:
    out: list[Path] = []
    for v in values:
        p = Path(v)
        if not p.exists():
            raise SystemExit(f"[ERROR] 不存在: {p}")
        out.append(p)
    return out


def cmd_propose(args: argparse.Namespace) -> int:
    rows = load_qc_rows(_paths(args.qc))
    rules_dir = Path(args.rules_dir) if args.rules_dir else None
    ev = evaluate_sample(rows, rules_dir, **_eval_kwargs(args))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(out_dir / "proposed_rules.json", ev)
    (out_dir / "proposed_pass2.toml").write_text(
        proposed_to_toml(ev["addable"]), encoding="utf-8"
    )
    print(json.dumps({
        "labels": ev["labels"],
        "n_addable": ev["n_addable"],
        "residual_f": ev["residual_f"],
        "top": [
            {
                "name": a["name"],
                "incremental_f": a["incremental_f"],
                "n_f": a["n_f"],
                "example": (a["examples"] or [""])[0],
            }
            for a in ev["addable"][:15]
        ],
    }, ensure_ascii=False, indent=2))
    print(f"\n写盘: {out_dir / 'proposed_rules.json'}")
    print(f"     {out_dir / 'proposed_pass2.toml'}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    rules_dir = Path(args.rules_dir) if args.rules_dir else None
    fresh_evals = []
    for group in args.qc_fresh:
        rows = load_qc_rows(_paths(group.split(",")))
        fresh_evals.append(evaluate_sample(rows, rules_dir, **_eval_kwargs(args)))
    cumulative = None
    if args.qc_all:
        cumulative = evaluate_sample(
            load_qc_rows(_paths(args.qc_all)), rules_dir, **_eval_kwargs(args)
        )
    verdict = declare_boundary(fresh_evals, consecutive=args.consecutive)
    payload = {
        "category": args.category,
        "declared_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status": verdict["status"],
        "reason": verdict["reason"],
        "consecutive_zero": verdict["consecutive_zero"],
        "required_consecutive": verdict["required_consecutive"],
        "min_f": args.min_f,
        "mine": not args.no_mine,
        "skip_names": sorted(_eval_kwargs(args)["skip_names"]),
        "fresh": [
            {"labels": e["labels"], "n_addable": e["n_addable"], "residual_f": e["residual_f"]}
            for e in fresh_evals
        ],
        "cumulative": None if cumulative is None else {
            "labels": cumulative["labels"],
            "n_addable": cumulative["n_addable"],
            "residual_f": cumulative["residual_f"],
        },
    }
    if args.keep_path:
        payload["keep_path"] = args.keep_path
    out_dir = Path(args.output_dir)
    write_report(out_dir / "text_boundary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n写盘: {out_dir / 'text_boundary.json'}")
    print(f"判定: {payload['status']}")
    return 0 if payload["status"] == "text_boundary" else 2


def main() -> int:
    p = argparse.ArgumentParser(description="文本过滤边界评估")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prop = sub.add_parser("propose", help="从 QC 标注提出可写 certain-noise 规则")
    p_prop.add_argument("--qc", action="append", required=True, help="text QC CSV，可多次")
    p_prop.add_argument("--rules-dir", default="", help="现有 blacklist 目录")
    p_prop.add_argument("-o", "--output-dir", required=True)
    p_prop.add_argument("--min-f", type=int, default=MIN_F_DEFAULT)
    p_prop.add_argument("--category", default="")
    p_prop.add_argument("--no-mine", action="store_true", help="不计标题 ngram 矿机（只评 GENERIC 库）")
    p_prop.add_argument("--skip-name", action="append", default=[], help="政策跳过的候选名，可多次")
    p_prop.set_defaults(func=cmd_propose)

    p_chk = sub.add_parser("check", help="用连续新鲜样本判定是否到达文本边界")
    p_chk.add_argument(
        "--qc-fresh", action="append", required=True,
        help="一份独立新鲜样本（逗号分隔多个文件视为同一份）",
    )
    p_chk.add_argument("--qc-all", action="append", default=[], help="累计标注 CSV")
    p_chk.add_argument("--rules-dir", default="")
    p_chk.add_argument("-o", "--output-dir", required=True)
    p_chk.add_argument("--category", default="")
    p_chk.add_argument("--keep-path", default="")
    p_chk.add_argument("--min-f", type=int, default=MIN_F_DEFAULT)
    p_chk.add_argument("--consecutive", type=int, default=CONSECUTIVE_DEFAULT)
    p_chk.add_argument("--no-mine", action="store_true", help="不计标题 ngram 矿机（只评 GENERIC 库）")
    p_chk.add_argument("--skip-name", action="append", default=[], help="政策跳过的候选名，可多次")
    p_chk.set_defaults(func=cmd_check)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
tools/run_manifest.py — 批次 manifest 增删查

用法:
  02_脚本/tools/run_manifest.py init -o data/runs/film_tv/human_0724/ \\
    --category film_tv --source human --batch 0724 --input raw/.../a.csv
  02_脚本/tools/run_manifest.py update -o … --stage quality --path keep=…/q.csv
  02_脚本/tools/run_manifest.py show -o data/runs/film_tv/human_0724/
  02_脚本/tools/run_manifest.py paths -o data/runs/film_tv/human_0724/
  02_脚本/tools/run_manifest.py list --runs-root data/runs
  02_脚本/tools/run_manifest.py find-deliver --category film_tv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.run_manifest import (  # noqa: E402
    find_deliver_paths,
    format_list_table,
    format_paths_only,
    format_summary,
    init_manifest,
    iter_manifests,
    load_manifest,
    update_stage,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RUNS = _REPO_ROOT / "data" / "runs"


def main() -> None:
    p = argparse.ArgumentParser(description="批次 manifest 索引")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="创建 manifest.json")
    p_init.add_argument("-o", "--batch-root", required=True)
    p_init.add_argument("--category", required=True)
    p_init.add_argument("--source", required=True, choices=("human", "machine"))
    p_init.add_argument("--batch", required=True)
    p_init.add_argument("--input", default="")
    p_init.add_argument("--notes", default="")

    p_up = sub.add_parser("update", help="更新某一 stage")
    p_up.add_argument("-o", "--batch-root", required=True)
    p_up.add_argument("--stage", required=True)
    p_up.add_argument(
        "--path", action="append", default=[],
        help="key=value，可多次，如 keep=a.csv drop=b.csv",
    )
    p_up.add_argument("--deliver", default=None, help="登记最终交付路径")

    p_show = sub.add_parser("show", help="打印 manifest")
    p_show.add_argument("-o", "--batch-root", required=True)

    p_paths = sub.add_parser("paths", help="仅打印各 stage 文件路径（show 精简版）")
    p_paths.add_argument("-o", "--batch-root", required=True)

    p_list = sub.add_parser("list", help="扫描 runs 下列出各批次 manifest")
    p_list.add_argument(
        "--runs-root", default=str(_DEFAULT_RUNS),
        help="runs 根目录（默认 data/runs）",
    )

    p_fd = sub.add_parser("find-deliver", help="按品类查找交付路径")
    p_fd.add_argument("--category", required=True)
    p_fd.add_argument("--batch", default=None)
    p_fd.add_argument("--source", default=None, choices=("human", "machine"))
    p_fd.add_argument(
        "--runs-root", default=str(_DEFAULT_RUNS),
        help="runs 根目录（默认 data/runs）",
    )

    args = p.parse_args()

    if args.cmd == "init":
        path = init_manifest(
            args.batch_root,
            category=args.category,
            source=args.source,
            batch=args.batch,
            input_path=args.input,
            notes=args.notes,
        )
        print(f"已写 {path}")
        return

    if args.cmd == "update":
        paths: dict[str, str] = {}
        for item in args.path:
            if "=" not in item:
                print(f"[ERROR] --path 应为 key=value: {item}")
                sys.exit(2)
            k, v = item.split("=", 1)
            paths[k.strip()] = v.strip()
        path = update_stage(
            args.batch_root,
            args.stage,
            paths=paths or None,
            deliver_path=args.deliver,
        )
        print(f"已更新 {path}")
        return

    if args.cmd == "show":
        data = load_manifest(args.batch_root)
        if not data:
            print(f"[ERROR] 无 manifest: {args.batch_root}")
            sys.exit(1)
        print(format_summary(data))
        return

    if args.cmd == "paths":
        data = load_manifest(args.batch_root)
        if not data:
            print(f"[ERROR] 无 manifest: {args.batch_root}")
            sys.exit(1)
        print(format_paths_only(data))
        return

    if args.cmd == "list":
        rows = iter_manifests(args.runs_root)
        print(format_list_table(rows))
        print(f"\n共 {len(rows)} 个批次  (root={args.runs_root})")
        return

    if args.cmd == "find-deliver":
        hits = find_deliver_paths(
            args.runs_root,
            category=args.category,
            batch=args.batch,
            source=args.source,
        )
        if not hits:
            print("(no deliver paths)")
            sys.exit(0)
        for h in hits:
            print(
                f"{h['category']}\t{h['source']}\t{h['batch']}\t{h['deliver_path']}"
            )
        return


if __name__ == "__main__":
    main()

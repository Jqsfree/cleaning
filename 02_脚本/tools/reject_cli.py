#!/usr/bin/env python3
"""
tools/reject_cli.py — 排除类资产工具统一入口（薄包装，逻辑仍在各脚本）

用法:
  02_脚本/tools/reject_cli.py accumulate -o $BATCH/ ...
  02_脚本/tools/reject_cli.py cascade -o $BATCH/
  02_脚本/tools/reject_cli.py propose ...
  02_脚本/tools/reject_cli.py export --batch-root $BATCH/
  02_脚本/tools/reject_cli.py metrics --assets-root data/assets/rejects
  02_脚本/tools/reject_cli.py suggest --assets-root data/assets/rejects
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent

_COMMANDS: dict[str, str] = {
    "accumulate": "accumulate_reject_assets.py",
    "cascade": "cascade_reject_propose.py",
    "propose": "propose_reject_tags.py",
    "export": "export_reject_assets.py",
    "metrics": "reject_source_metrics.py",
    "suggest": "suggest_reject_opt.py",
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        description="排除类资产 CLI（转发到既有 tools 脚本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "子命令:\n"
            "  accumulate  text drop + thumb → proposed → export\n"
            "  cascade     高把握提案 + 验证抽样\n"
            "  propose     仅提案标签\n"
            "  export      导出到 data/assets/rejects/\n"
            "  metrics     来源 overturn 账本\n"
            "  suggest     优化建议（默认不改配置）\n"
        ),
    )
    p.add_argument(
        "command",
        choices=sorted(_COMMANDS.keys()),
        help="子命令",
    )
    p.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="原脚本参数（可带 --）",
    )
    args = p.parse_args(argv)

    script = _TOOLS / _COMMANDS[args.command]
    if not script.is_file():
        print(f"[ERROR] 脚本不存在: {script}", flush=True)
        sys.exit(1)

    rest = list(args.rest)
    if rest and rest[0] == "--":
        rest = rest[1:]

    cmd = [sys.executable, str(script), *rest]
    print(f"[reject_cli] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()

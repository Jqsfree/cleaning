#!/usr/bin/env python3
"""CLI: exo_service 文本级联 L1 DROP + L2 行业路由。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT))

from categories.exo_service.cascade_text import run_l1_l2  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="exo_service L1 certain-noise DROP + L2 industry route")
    ap.add_argument("input", help="quality CSV")
    ap.add_argument("-o", "--output", required=True, help="输出目录")
    ap.add_argument("--stem", default="商业服务_quality")
    args = ap.parse_args()
    summary = run_l1_l2(args.input, args.output, stem=args.stem)
    print(
        f"done  input={summary['n_input']:,}  "
        f"l1_drop={summary['n_l1_drop']:,}  "
        f"candidates={summary['n_candidates']:,}  "
        f"unrouted={summary['n_unrouted']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

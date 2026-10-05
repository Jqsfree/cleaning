#!/usr/bin/env python3
"""Audit cross-category human labels or shadow-scan metadata for non-live forms.

Examples:
  python 02_脚本/pipeline/non_live_text.py audit --root /path/to/Downloads \
    --root data/runs --out work/non_live_audit
  python 02_脚本/pipeline/non_live_text.py scan --input quality.csv \
    --out work/non_live_scan

Neither command automatically drops source rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.non_live_text import DEFAULT_POLICY, audit_annotations, scan_metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Replay candidate signals over annotation files")
    audit.add_argument("--root", type=Path, action="append", default=[],
                       help="Annotation search root; may be repeated")
    audit.add_argument("--file", type=Path, action="append", default=[],
                       help="Additional explicit human annotation file; may be repeated")
    audit.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    audit.add_argument("--out", type=Path, required=True)
    scan = sub.add_parser("scan", help="Tag candidate titles; keep source unchanged")
    scan.add_argument("--input", type=Path, required=True)
    scan.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    scan.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "audit":
        if not args.root and not args.file:
            parser.error("audit requires at least one --root or --file")
        result = audit_annotations(roots=args.root, files=args.file,
                                   policy_path=args.policy, output=args.out)
    else:
        result = scan_metadata(input_path=args.input, policy_path=args.policy, output=args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

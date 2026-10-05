#!/usr/bin/env python3
"""Unified metadata pipeline; run --help for batch and diagnostic commands."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.metadata_cli import main
if __name__ == "__main__":
    raise SystemExit(main())

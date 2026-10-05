#!/usr/bin/env python3
"""Compatibility entrypoint. Prefer 02_脚本/pipeline/metadata.py."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from core.metadata_cli import main
if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""从仓库根目录 ``.env`` 加载环境变量（不覆盖已存在的 os.environ）。

无第三方 python-dotenv 依赖。在 QC / 实验脚本入口调用 ``load_project_env()``。
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../cleng_exo (repo)
_DEFAULT_ENV = _REPO_ROOT / ".env"


def load_project_env(env_path: Path | None = None, *, override: bool = False) -> Path | None:
    """Parse KEY=VALUE lines from ``.env``. Returns path loaded, or None if missing."""
    path = Path(env_path) if env_path else _DEFAULT_ENV
    if not path.is_file():
        return None
    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if not key:
            continue
        if override or key not in os.environ or os.environ.get(key, "") == "":
            os.environ[key] = val
    return path

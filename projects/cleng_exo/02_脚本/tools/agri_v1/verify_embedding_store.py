#!/usr/bin/env python3
"""校验 / 记录 CLIP embedding store 同步状态（spec Phase 1.2）。

用法:
  bash 02_脚本/tools/agri_v1/sync_embedding_store.sh [REMOTE:/path/to/store/]

未提供 REMOTE 时仅写 store_manifest.json（status=missing）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
STORE = _REPO / "data/assets/embeddings/exo_agriculture_0814_semantic_remain"
MANIFEST = STORE / "store_manifest.json"


def write_manifest(*, status: str, extra: dict | None = None) -> dict:
    STORE.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "store_dir": str(STORE.resolve()),
        "status": status,
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_files": ["index.csv", "embeddings.npy", "thumb_ok.npy", "meta.json"],
    }
    index = STORE / "index.csv"
    if index.is_file():
        import pandas as pd

        df = pd.read_csv(index, usecols=["video_id"], dtype=str)
        manifest["n_rows"] = len(df)
        manifest["n_unique_video_id"] = int(df["video_id"].nunique())
        meta = STORE / "meta.json"
        if meta.is_file():
            manifest["meta"] = json.loads(meta.read_text(encoding="utf-8"))
    if extra:
        manifest.update(extra)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _store_complete() -> bool:
    return all((STORE / name).is_file() for name in ("index.csv", "embeddings.npy", "thumb_ok.npy", "meta.json"))


def main() -> int:
    remote = sys.argv[1] if len(sys.argv) > 1 else ""
    if not remote:
        if _store_complete():
            m = write_manifest(status="ready", extra={"sync_required": False, "source": "local"})
            print(json.dumps(m, ensure_ascii=False, indent=2))
            return 0
        m = write_manifest(
            status="missing",
            extra={
                "sync_required": True,
                "instructions": (
                    "提供 rsync 源后执行: "
                    "bash 02_脚本/tools/agri_v1/sync_embedding_store.sh user@host:/path/exo_agriculture_0814_semantic_remain/"
                ),
            },
        )
        print(json.dumps(m, ensure_ascii=False, indent=2))
        return 1

    import subprocess

    STORE.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-avz", "--progress", f"{remote.rstrip('/')}/", f"{STORE}/"]
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        write_manifest(status="sync_failed", extra={"remote": remote, "returncode": proc.returncode})
        return proc.returncode

    m = write_manifest(status="ready", extra={"remote": remote})
    print(json.dumps(m, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

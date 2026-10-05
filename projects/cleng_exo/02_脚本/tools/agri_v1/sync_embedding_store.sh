#!/usr/bin/env bash
# 同步 CLIP embedding store → data/assets/embeddings/exo_agriculture_0814_semantic_remain/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
REMOTE="${1:-}"
PYTHON="${ROOT}/.venv/bin/python3"
exec "$PYTHON" 02_脚本/tools/agri_v1/verify_embedding_store.py ${REMOTE:+"$REMOTE"}

#!/usr/bin/env python3
"""
v2 清洗管道 — 多 chunk 合并版本（薄包装）。

直接复用 clean_sports_chunk_v2.py 的全部规则与逻辑，
仅通过 sys.argv 透传参数。

用法同 clean_sports_chunk_v2.py:
  python3 clean_sports_chunk_v2_merged.py <input.csv>
"""

import sys
from pathlib import Path

# 将自身文件名替换为 clean_sports_chunk_v2.py 再执行
sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_sports_chunk_v2

if __name__ == "__main__":
    clean_sports_chunk_v2.main()

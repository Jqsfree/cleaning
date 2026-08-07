#!/usr/bin/env python3
"""
tools/build_exemplar_bank.py — 本地样例视频 → 抽帧 + CLIP 原型

用法:
  02_脚本/tools/build_exemplar_bank.py \\
    --video-dir "/home/jqs/Downloads/直播和直播带货场景视频样品/YouTube/直播场景" \\
    -o data/assets/exemplars/yt_live_scene/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.exemplar_sim import DEFAULT_N_FRAMES, build_bank  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="样例视频建库（抽帧+原型）")
    p.add_argument("--video-dir", required=True, help="样例视频目录（mp4/mkv/webm）")
    p.add_argument("-o", "--out-dir", required=True, help="输出 bank 目录")
    p.add_argument("--n-frames", type=int, default=DEFAULT_N_FRAMES)
    p.add_argument("--copy", action="store_true", help="拷贝视频而非 symlink")
    args = p.parse_args()

    print(f"video_dir={args.video_dir}")
    print(f"out={args.out_dir}  n_frames={args.n_frames}")
    meta = build_bank(
        args.video_dir,
        args.out_dir,
        n_frames=args.n_frames,
        symlink=not args.copy,
    )
    print(f"done  n_exemplars={meta['n_exemplars']}  model={meta['model']}")


if __name__ == "__main__":
    main()

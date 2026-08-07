#!/usr/bin/env python3
"""生成 sb0/sb1/sb2 storyboard 三档对比页并在浏览器打开。

目录结构（由 export_pass19_storyboard_frames.py 等导出）::

    <frames_dir>/
        manifest.csv
        sb0/<video_id>/frame_00.jpg ...
        sb1/<video_id>/frame_00.jpg ...
        sb2/<video_id>/frame_00.jpg ...

用法::

    python3 experiments/welding_sb/view_sb_compare.py data/runs/welding/vision_qc_pass/pass19_sb_frames
    python3 experiments/welding_sb/view_sb_compare.py <dir> --video 7Q3MDgSZWyc --frame 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE = _SCRIPT_DIR / "templates" / "sb_compare.html"
TIERS = ("sb0", "sb1", "sb2")
OUT_NAME = "compare.html"


def load_manifest(frames_dir: Path) -> dict:
    manifest_path = frames_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"未找到 manifest.csv: {manifest_path}")

    df = pd.read_csv(manifest_path)
    required = {"video_id", "tier", "w", "h", "n"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest.csv 缺少列: {sorted(missing)}")

    data: dict[str, dict[str, dict]] = {}
    for row in df.itertuples(index=False):
        vid = str(row.video_id)
        tier = str(row.tier)
        if tier not in TIERS:
            continue
        data.setdefault(vid, {})[tier] = {
            "w": int(row.w),
            "h": int(row.h),
            "n": int(row.n),
        }
    if not data:
        raise ValueError(f"{manifest_path} 无有效 sb0/sb1/sb2 记录")
    return data


def build_html(frames_dir: Path, data: dict) -> str:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"模板缺失: {TEMPLATE}")

    title = f"Storyboard 对比 · {frames_dir.name}"
    tpl = TEMPLATE.read_text(encoding="utf-8")
    return (
        tpl.replace("__TITLE__", title)
        .replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False))
    )


def write_compare_html(frames_dir: Path, out_path: Path | None = None) -> Path:
    frames_dir = frames_dir.resolve()
    data = load_manifest(frames_dir)
    html = build_html(frames_dir, data)
    dest = out_path or (frames_dir / OUT_NAME)
    dest.write_text(html, encoding="utf-8")
    return dest


def open_browser(url: str) -> None:
    if shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    webbrowser.open(url)


def serve_and_open(frames_dir: Path, port: int, video: str | None, frame: int | None) -> None:
    frames_dir = frames_dir.resolve()
    handler = type("Handler", (SimpleHTTPRequestHandler,), {"directory": str(frames_dir)})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/{OUT_NAME}"
    if video:
        frag = f"v={video}"
        if frame is not None:
            frag += f"&f={frame}"
        url += f"#{frag}"

    print(f"对比页: {url}")
    print("Ctrl+C 结束本地服务")
    open_browser(url)
    try:
        thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="sb0/sb1/sb2 storyboard 浏览器三档对比")
    parser.add_argument("frames_dir", type=Path, help="含 manifest.csv 与 sb0/sb1/sb2 的目录")
    parser.add_argument("-o", "--output", type=Path, default=None, help=f"HTML 输出路径（默认 <dir>/{OUT_NAME}）")
    parser.add_argument("--video", default=None, help="打开时定位到 video_id")
    parser.add_argument("--frame", type=int, default=None, help="打开时定位到帧序号（0 起）")
    parser.add_argument("--no-open", action="store_true", help="只生成 HTML，不打开浏览器")
    parser.add_argument("--serve", type=int, nargs="?", const=8765, metavar="PORT",
                        help="用内置 HTTP 服务打开（默认端口 8765）")
    args = parser.parse_args()

    dest = write_compare_html(args.frames_dir, args.output)
    print(f"已生成: {dest}")

    if args.no_open:
        return

    if args.serve is not None:
        serve_and_open(args.frames_dir, args.serve, args.video, args.frame)
        return

    url = dest.as_uri()
    if args.video:
        frag = f"v={args.video}"
        if args.frame is not None:
            frag += f"&f={args.frame}"
        url += f"#{frag}"
    print(f"打开: {url}")
    open_browser(url)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

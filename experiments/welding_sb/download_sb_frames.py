#!/usr/bin/env python3
"""
下载视觉QC通过视频的 sb0/sb1/sb2 storyboard 帧，用于质量对比。

使用 yt-dlp 提取 storyboard URL（自动读 Chrome cookies），
下载 sb0/sb1/sb2 各档位第一张拼图，取中间帧保存。

输出结构:
  sb_compare/{video_id}/
    sb0.jpg   — 320x180 每帧
    sb1.jpg   — ~160x90 每帧
    sb2.jpg   — 80x45 每帧
"""

import os
import csv
from io import BytesIO
from urllib.request import urlopen, Request

from PIL import Image
import yt_dlp

OUT_DIR = "/home/jqs/projects/clean_DATASET/data/runs/welding/sb_compare"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"

# 视觉QC run01 通过的 19 条
VIDEO_IDS = [
    "AvGXWT2ipq4", "7Q3MDgSZWyc", "FFyNhc547GA", "ML1TMVbO_kA",
    "P0MyPziDNVY", "Z0oTOEBZ7wI", "IXvYXFWGGf8", "y9dgwI_VZTk",
    "3kTGaf-Fo8o", "sbUmr8BZLDI", "gXAqkDRB7-k", "-EEVs_-1Tkc",
    "f75qLVYgG5w", "NP893SsKOCc", "QD9lSl3p6BM", "Tpy9neFZtj8",
    "0fOcJLiJQ04", "RAYvQMCosms", "vayOz54wMqo",
]


def fetch_url(url: str, timeout: int = 15) -> bytes:
    req = Request(url, headers={"User-Agent": UA, "Referer": "https://www.youtube.com/"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def get_storyboard_urls(video_id: str) -> dict:
    """用 yt-dlp 提取各 sb 档位第一张拼图 URL（自动读 Chrome cookies）。"""
    ydl_opts = {
        "quiet": True, "skip_download": True, "no_warnings": True,
        "cookiesfrombrowser": ("chrome",),
        "extractor_args": {"youtube": {"player_client": ["android", "web"], "skip": ["hls", "dash"]}},
    }
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        print(f"  [{video_id}] yt-dlp 失败: {e}")
        return {}

    result = {}
    for f in info.get("formats", []):
        fid = f.get("format_id", "")
        if f.get("format_note") != "storyboard" or fid not in ("sb0", "sb1", "sb2"):
            continue
        fragments = f.get("fragments", [])
        sheet_url = fragments[0]["url"] if fragments else f.get("url", "")
        if sheet_url:
            result[fid] = {
                "url": sheet_url,
                "cols": f.get("columns", 10),
                "rows": f.get("rows", 10),
                "w": f.get("width", 160),
                "h": f.get("height", 90),
            }
    return result


def extract_sample_frame(sheet_bytes: bytes, sb_info: dict) -> Image.Image:
    """从拼图中取中间帧。"""
    cols = sb_info["cols"]
    rows = sb_info["rows"]
    fw = sb_info["w"]
    fh = sb_info["h"]
    sheet = Image.open(BytesIO(sheet_bytes)).convert("RGB")
    mid_frame = (rows * cols) // 2
    r, c = mid_frame // cols, mid_frame % cols
    x, y = c * fw, r * fh
    if x + fw <= sheet.width and y + fh <= sheet.height:
        return sheet.crop((x, y, x + fw, y + fh))
    return sheet.crop((0, 0, fw, fh))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for i, vid in enumerate(VIDEO_IDS):
        print(f"\n[{i+1}/19] {vid}")
        urls = get_storyboard_urls(vid)
        if not urls:
            print(f"  无 storyboard")
            continue

        vid_dir = os.path.join(OUT_DIR, vid)
        os.makedirs(vid_dir, exist_ok=True)

        for fmt in ("sb0", "sb1", "sb2"):
            if fmt not in urls:
                print(f"  {fmt}: 无")
                continue
            try:
                data = fetch_url(urls[fmt]["url"])
                frame = extract_sample_frame(data, urls[fmt])
                path = os.path.join(vid_dir, f"{fmt}.jpg")
                frame.save(path, quality=90)
                print(f"  {fmt}: {frame.width}x{frame.height} ✓")
            except Exception as e:
                print(f"  {fmt}: 下载失败 {e}")

    # 汇总
    summary = []
    for vid in VIDEO_IDS:
        for fmt in ("sb0", "sb1", "sb2"):
            p = os.path.join(OUT_DIR, vid, f"{fmt}.jpg")
            if os.path.exists(p):
                img = Image.open(p)
                summary.append([vid, fmt, img.width, img.height, p])
    with open(os.path.join(OUT_DIR, "_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "format", "width", "height", "path"])
        w.writerows(summary)

    print(f"\n完成！{len(summary)} 张 → {OUT_DIR}")


if __name__ == "__main__":
    main()

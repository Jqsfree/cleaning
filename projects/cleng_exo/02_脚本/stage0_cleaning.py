#!/usr/bin/env python3
"""
视频数据采购 —— 清洗入口脚本（Stage 0，规则/传统CV，不接语义模型）
v2：Phase B 全面改为"不下载整段视频"的流式局部抓帧方案，适配数据源为
YouTube 等视频网站链接的场景。若某行的值是本地已存在的文件路径，
会自动退回本地 ffprobe/直接读帧的方式（同一脚本兼容两种来源）。

依据《客户视频数据采购需求说明》第三、四节提炼，只做与类目无关、
可脚本化、不需要深度学习模型的技术性清洗：

Phase A —— 元数据级（YouTube来源用 yt-dlp --skip-download 探测，不下载任何视频数据）
    1. 必填字段空值检查
    2. 元数据是否可探测（探测失败即标记 corrupt）
    3. 时长门槛（默认 >=30 秒，可按类目覆盖）
    4. 分辨率门槛（默认 >=1080p）
    5. 时长的 IQR 统计离群值（仅标记，不自动淘汰）
    6. 重复 video_id / 文件路径检测

Phase B —— 画面级技术判据（文档标注"全批统一，不分档"的三项），
           全部通过局部小片段/散点抓帧完成，不下载整段视频：
    1. 假横屏：抓 N 个散点单帧（复用 Phase B 里为"有效时长"抓的帧，不重复下载）
    2. 抖动：额外抓 K 段约 1.5~2 秒的连续小片段做光流，其余时间不碰
    3. 有效时长（近似）：稀疏采样 N 个时间点各抓 1 帧，用相邻采样帧的
       直方图相关性判断区间内"像是同一镜头"，按此估算连续镜头占比。
       ⚠️ 这是近似值：采样间隔内发生的切镜头会被漏检，用于换取带宽节省。

依赖：
    pip install opencv-python-headless numpy yt-dlp --break-system-packages
    系统需要 ffmpeg / ffprobe

用法：
    python stage0_cleaning.py input.csv output.csv \
        [--id-col video_id] [--category-col category] \
        [--samples 10] [--shake-windows 4] [--skip-frame-checks]

输入 CSV 至少要有一列（默认列名 video_id）存放 YouTube 视频ID/URL，
或本地已存在的文件路径（脚本会自动判断来源类型）。
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np

# ---------------- 全局默认阈值（来自 PDF 第三节） ----------------
GLOBAL_DURATION_MIN_SEC = 30.0
GLOBAL_RESOLUTION_MIN_H = 1080

SHAKE_PATH_SPEED_MIN = 0.05
SHAKE_NET_OVER_PATH_MAX = 0.35
SHAKE_DURATION_RATIO_MAX = 0.08

EFFECTIVE_SHOT_MIN_SEC = 3.0
EFFECTIVE_DURATION_RATIO_MIN = 0.85
HIST_CONTINUITY_THRESHOLD = 0.6  # 相邻采样帧直方图相关性，高于此值视为"同一镜头延续"

CATEGORY_OVERRIDES = {
    "八.通用场景": {"duration_min": 30.0, "resolution_min_h": 1080},
    "十一.专业领域现场纪实": {"resolution_min_h": 1080},
    "vlog": {"duration_min": None, "resolution_min_h": None},
}

ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})")


# ==================== 来源判定 ====================
def resolve_source(value: str):
    """判断这行数据是本地文件还是 YouTube 视频，返回 ("local"|"youtube"|"unknown", 标识)"""
    value = value.strip()
    if not value:
        return ("unknown", "")
    if os.path.exists(value):
        return ("local", value)
    m = ID_RE.search(value)
    if m:
        return ("youtube", m.group(1))
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return ("youtube", value)
    return ("unknown", value)


# ==================== Phase A: 元数据探测 ====================
def probe_local(path: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=width,height,avg_frame_rate",
         "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"corrupt": True, "duration": None, "width": None, "height": None, "fps": None}
    try:
        lines = [l for l in result.stdout.strip().splitlines() if l]
        # 第一行通常是 video stream: width,height,fps ; 最后一行是 format duration
        w, h, rate = lines[0].split(",")
        num, den = (rate.split("/") + ["1"])[:2]
        fps = float(num) / float(den) if float(den or 1) != 0 else 0.0
        duration = float(lines[-1]) if lines[-1].replace(".", "", 1).isdigit() else None
        corrupt = duration is None or duration <= 0 or int(w) <= 0 or int(h) <= 0
        return {"corrupt": corrupt, "duration": duration, "width": int(w), "height": int(h), "fps": fps}
    except (ValueError, IndexError):
        return {"corrupt": True, "duration": None, "width": None, "height": None, "fps": None}


def probe_youtube(video_id: str, auth_args: list[str] | None = None) -> dict:
    """只探测元数据，不下载任何视频/音频数据"""
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = ["yt-dlp", "--skip-download", "--print",
           "%(duration)s;;%(width)s;;%(height)s;;%(fps)s"]
    if auth_args:
        cmd[1:1] = auth_args
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return {"corrupt": True, "duration": None, "width": None, "height": None, "fps": None}
    try:
        line = result.stdout.strip().splitlines()[-1]
        dur_s, w_s, h_s, fps_s = line.split(";;")
        duration = float(dur_s) if dur_s not in ("", "NA", "None") else None
        width = int(float(w_s)) if w_s not in ("", "NA", "None") else None
        height = int(float(h_s)) if h_s not in ("", "NA", "None") else None
        fps = float(fps_s) if fps_s not in ("", "NA", "None") else None
        corrupt = duration is None or duration <= 0 or not width or not height
        return {"corrupt": corrupt, "duration": duration, "width": width, "height": height, "fps": fps}
    except (ValueError, IndexError):
        return {"corrupt": True, "duration": None, "width": None, "height": None, "fps": None}


def iqr_outlier_bounds(values: list) -> tuple:
    if len(values) < 4:
        return (float("-inf"), float("inf"))
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return (q1 - 1.5 * iqr, q3 + 1.5 * iqr)


# ==================== Phase B: 局部抓帧（不下载整段视频） ====================
def grab_youtube_frame(video_id: str, timestamp: float, tmp_dir: str, idx: int, window: float = 0.6,
                       auth_args: list[str] | None = None) -> str:
    """只下载 timestamp 附近约 window 秒的小片段，截 1 帧后删除片段"""
    start = max(0, timestamp - window / 2)
    end = timestamp + window / 2
    clip_path = os.path.join(tmp_dir, f"clip_{idx}.mp4")
    frame_path = os.path.join(tmp_dir, f"frame_{idx:03d}.jpg")

    cmd = ["yt-dlp", "-f", "worst[height>=240]/best",
           "--download-sections", f"*{start:.2f}-{end:.2f}",
           "-o", clip_path]
    if auth_args:
        cmd[1:1] = auth_args
    cmd.append(f"https://www.youtube.com/watch?v={video_id}")
    dl = subprocess.run(cmd, capture_output=True, text=True)
    if dl.returncode != 0 or not os.path.exists(clip_path):
        return ""
    subprocess.run(["ffmpeg", "-y", "-i", clip_path, "-frames:v", "1", "-q:v", "2", frame_path],
                    capture_output=True)
    if os.path.exists(clip_path):
        os.remove(clip_path)
    return frame_path if os.path.exists(frame_path) else ""


def grab_youtube_window_frames(video_id: str, timestamp: float, tmp_dir: str, idx: int,
                                 window_sec: float = 1.6, extract_fps: int = 5,
                                 auth_args: list[str] | None = None) -> list:
    """下载约 window_sec 秒的连续小片段，按 extract_fps 抽出多帧用于光流计算"""
    start = max(0, timestamp)
    end = timestamp + window_sec
    clip_path = os.path.join(tmp_dir, f"shakeclip_{idx}.mp4")
    frame_pattern = os.path.join(tmp_dir, f"shake_{idx:03d}_%02d.jpg")

    cmd = ["yt-dlp", "-f", "worst[height>=240]/best",
           "--download-sections", f"*{start:.2f}-{end:.2f}",
           "-o", clip_path]
    if auth_args:
        cmd[1:1] = auth_args
    cmd.append(f"https://www.youtube.com/watch?v={video_id}")
    dl = subprocess.run(cmd, capture_output=True, text=True)
    if dl.returncode != 0 or not os.path.exists(clip_path):
        return []
    subprocess.run(["ffmpeg", "-y", "-i", clip_path, "-vf", f"fps={extract_fps}",
                     "-q:v", "3", frame_pattern], capture_output=True)
    if os.path.exists(clip_path):
        os.remove(clip_path)

    frames = sorted(
        os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
        if f.startswith(f"shake_{idx:03d}_")
    )
    return frames


def grab_local_frame(path: str, timestamp: float, tmp_dir: str, idx: int) -> str:
    frame_path = os.path.join(tmp_dir, f"frame_{idx:03d}.jpg")
    subprocess.run(["ffmpeg", "-y", "-ss", str(timestamp), "-i", path,
                     "-frames:v", "1", "-q:v", "2", frame_path], capture_output=True)
    return frame_path if os.path.exists(frame_path) else ""


def grab_local_window_frames(path: str, timestamp: float, tmp_dir: str, idx: int,
                               window_sec: float = 1.6, extract_fps: int = 5) -> list:
    frame_pattern = os.path.join(tmp_dir, f"shake_{idx:03d}_%02d.jpg")
    subprocess.run(["ffmpeg", "-y", "-ss", str(timestamp), "-i", path, "-t", str(window_sec),
                     "-vf", f"fps={extract_fps}", "-q:v", "3", frame_pattern], capture_output=True)
    return sorted(
        os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
        if f.startswith(f"shake_{idx:03d}_")
    )


# ---------------- 检测算法（传统CV，与来源无关） ----------------
def compute_shake_ratio(window_frame_lists: list) -> float:
    if not window_frame_lists:
        return 0.0
    shaky = 0
    valid = 0
    for frames in window_frame_lists:
        imgs = [cv2.imread(p) for p in frames if p]
        imgs = [im for im in imgs if im is not None]
        if len(imgs) < 3:
            continue
        h = imgs[0].shape[0]
        grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
        disp = []
        for i in range(1, len(grays)):
            flow = cv2.calcOpticalFlowFarneback(grays[i - 1], grays[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            disp.append((float(np.mean(flow[..., 0])), float(np.mean(flow[..., 1]))))
        path_len = sum((dx**2 + dy**2) ** 0.5 for dx, dy in disp)
        net_dx, net_dy = sum(d[0] for d in disp), sum(d[1] for d in disp)
        net_len = (net_dx**2 + net_dy**2) ** 0.5
        path_speed_norm = path_len / h
        valid += 1
        if path_speed_norm >= SHAKE_PATH_SPEED_MIN and (net_len / path_len if path_len > 0 else 1) < SHAKE_NET_OVER_PATH_MAX:
            shaky += 1
    return shaky / valid if valid else 0.0


def compute_effective_duration_approx(frame_paths: list, timestamps: list, total_duration: float) -> float:
    """
    近似有效时长占比：相邻采样点之间若直方图高度相关，视为该区间是
    同一连续镜头；若区间长度 >=3秒则计入有效时长。
    ⚠️ 采样间隔内部真实发生的切镜头会被漏检，属于用带宽换来的近似值。
    """
    valid_pairs = [(p, t) for p, t in zip(frame_paths, timestamps) if p]
    if len(valid_pairs) < 2 or total_duration <= 0:
        return 1.0

    hists = []
    for p, t in valid_pairs:
        img = cv2.imread(p)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hists.append((hist, t))

    effective = 0.0
    for i in range(1, len(hists)):
        interval = hists[i][1] - hists[i - 1][1]
        corr = cv2.compareHist(hists[i - 1][0], hists[i][0], cv2.HISTCMP_CORREL)
        if corr >= HIST_CONTINUITY_THRESHOLD and interval >= EFFECTIVE_SHOT_MIN_SEC:
            effective += interval

    return min(1.0, effective / total_duration)


def compute_fake_landscape(frame_paths: list, n_strips: int = 20) -> bool:
    valid = [p for p in frame_paths if p]
    if not valid:
        return False
    votes = 0
    checked = 0
    for p in valid[:8]:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        strip_w = w // n_strips
        if strip_w == 0:
            continue
        sharpness = [float(cv2.Laplacian(gray[:, i * strip_w:(i + 1) * strip_w], cv2.CV_64F).var())
                     for i in range(n_strips)]
        mid = n_strips // 2
        center_avg = np.mean(sharpness[mid - 3: mid + 3]) if n_strips >= 6 else np.mean(sharpness)
        if center_avg <= 0:
            continue
        left_ratio = np.mean(sharpness[:3]) / center_avg
        right_ratio = np.mean(sharpness[-3:]) / center_avg
        checked += 1
        if left_ratio < 0.25 and right_ratio < 0.25 and abs(left_ratio - right_ratio) < 0.15:
            votes += 1
    return checked > 0 and votes / checked >= 0.5


# ==================== 逐视频处理 ====================
def evaluate_frame_metrics(source_type: str, ident: str, duration: float,
                             n_samples: int, n_shake_windows: int, tmp_dir: str,
                             auth_args: list[str] | None = None) -> dict:
    timestamps = [duration * (i + 0.5) / n_samples for i in range(n_samples)]

    if source_type == "youtube":
        sparse_frames = [grab_youtube_frame(ident, t, tmp_dir, i, auth_args=auth_args) for i, t in enumerate(timestamps)]
    else:
        sparse_frames = [grab_local_frame(ident, t, tmp_dir, i) for i, t in enumerate(timestamps)]

    effective_ratio = compute_effective_duration_approx(sparse_frames, timestamps, duration)
    fake_landscape = compute_fake_landscape(sparse_frames)

    shake_timestamps = [duration * (i + 0.5) / n_shake_windows for i in range(n_shake_windows)]
    window_frame_lists = []
    for i, t in enumerate(shake_timestamps):
        if source_type == "youtube":
            frames = grab_youtube_window_frames(ident, t, tmp_dir, 1000 + i, auth_args=auth_args)
        else:
            frames = grab_local_window_frames(ident, t, tmp_dir, 1000 + i)
        if frames:
            window_frame_lists.append(frames)

    shake_ratio = compute_shake_ratio(window_frame_lists)

    return {
        "effective_duration_ratio": round(effective_ratio, 4),
        "effective_duration_pass": effective_ratio >= EFFECTIVE_DURATION_RATIO_MIN,
        "is_fake_landscape": fake_landscape,
        "shake_ratio": round(shake_ratio, 4),
        "shake_pass": shake_ratio <= SHAKE_DURATION_RATIO_MAX,
    }


# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--id-col", default="video_id", help="视频ID/URL/本地路径所在列名")
    parser.add_argument("--category-col", default="category")
    parser.add_argument("--samples", type=int, default=10, help="有效时长近似检测的采样点数")
    parser.add_argument("--shake-windows", type=int, default=4, help="抖动检测抓取的连续小片段数量")
    parser.add_argument("--skip-frame-checks", action="store_true")
    parser.add_argument("--cookies", default=os.getenv("YT_DLP_COOKIES_FILE"),
                        help="yt-dlp cookies 文件（默认读环境变量 YT_DLP_COOKIES_FILE）")
    parser.add_argument("--cookies-from-browser", default=os.getenv("YT_DLP_COOKIES_FROM_BROWSER"),
                        help="从浏览器提取 cookies（如 chrome；比文件更新鲜、更抗反爬）")
    args = parser.parse_args()

    # 认证：浏览器优先（更新鲜），其次 cookies 文件
    auth_args: list[str] = []
    if args.cookies_from_browser:
        auth_args = ["--cookies-from-browser", args.cookies_from_browser]
    elif args.cookies:
        auth_args = ["--cookies", args.cookies]

    with open(args.input_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows or args.id_col not in rows[0]:
        print(f"输入 CSV 必须包含列: {args.id_col}（用 --id-col 指定实际列名）")
        sys.exit(1)

    seen = set()
    for row in rows:
        raw = row.get(args.id_col, "").strip()
        row["file_missing"] = not bool(raw)
        source_type, ident = resolve_source(raw) if raw else ("unknown", "")
        row["_source_type"] = source_type
        row["_ident"] = ident
        row["is_duplicate"] = ident in seen if ident else False
        if ident:
            seen.add(ident)

        if source_type == "local":
            info = probe_local(ident)
        elif source_type == "youtube":
            info = probe_youtube(ident, auth_args=auth_args)
        else:
            info = {"corrupt": False, "duration": None, "width": None, "height": None, "fps": None}

        row["file_corrupt"] = info["corrupt"]
        row["duration_sec"] = info["duration"]
        row["width"] = info["width"]
        row["height"] = info["height"]
        row["fps"] = info["fps"]

    def thresholds_for(row):
        cat = row.get(args.category_col, "") if args.category_col in row else ""
        override = CATEGORY_OVERRIDES.get(cat, {})
        return (override.get("duration_min", GLOBAL_DURATION_MIN_SEC),
                override.get("resolution_min_h", GLOBAL_RESOLUTION_MIN_H))

    for row in rows:
        dmin, hmin = thresholds_for(row)
        row["duration_pass"] = (dmin is None) or (row["duration_sec"] is not None and row["duration_sec"] >= dmin)
        row["resolution_pass"] = (hmin is None) or (row["height"] is not None and row["height"] >= hmin)

    durations = [r["duration_sec"] for r in rows if r["duration_sec"] not in (None, 0)]
    lo, hi = iqr_outlier_bounds(durations)
    for row in rows:
        d = row["duration_sec"]
        row["duration_outlier"] = bool(d is not None and (d < lo or d > hi))

    for row in rows:
        if (args.skip_frame_checks or row["file_missing"] or row["file_corrupt"]
                or row["_source_type"] == "unknown" or not row["duration_sec"]):
            row.update({"effective_duration_ratio": None, "effective_duration_pass": None,
                        "is_fake_landscape": None, "shake_ratio": None, "shake_pass": None})
            continue
        print(f"抓帧检测: {row['_ident']} ({row['_source_type']})")
        with tempfile.TemporaryDirectory() as tmp:
            metrics = evaluate_frame_metrics(
                row["_source_type"], row["_ident"], row["duration_sec"],
                args.samples, args.shake_windows, tmp,
                auth_args=auth_args,
            )
        row.update(metrics)

    for row in rows:
        reasons = []
        if row["file_missing"]:
            reasons.append("file_missing")
        if row["file_corrupt"]:
            reasons.append("file_corrupt")
        if row["is_duplicate"]:
            reasons.append("duplicate")
        if row["duration_pass"] is False:
            reasons.append("duration_below_threshold")
        if row["resolution_pass"] is False:
            reasons.append("resolution_below_threshold")
        if row.get("shake_pass") is False:
            reasons.append("excessive_shake")
        if row.get("effective_duration_pass") is False:
            reasons.append("insufficient_effective_duration")
        if row.get("is_fake_landscape") is True:
            reasons.append("fake_landscape")

        row["reject_reasons"] = ";".join(reasons)
        row["keep"] = len(reasons) == 0
        row.pop("_source_type", None)
        row.pop("_ident", None)

    fieldnames = list(rows[0].keys())
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    kept = sum(1 for r in rows if r["keep"])
    outliers = sum(1 for r in rows if r["duration_outlier"])
    print(f"完成。{kept}/{len(rows)} 通过 Stage 0 清洗，{outliers} 条时长为统计离群值（未自动淘汰，建议人工复核）。")
    print(f"结果写入 {args.output_csv}")


if __name__ == "__main__":
    main()

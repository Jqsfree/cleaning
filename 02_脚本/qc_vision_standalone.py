#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立视觉质检脚本 —— 从 CSV 读取 YouTube 视频，下载片段 → 抽帧 → 本地视觉模型 QC → 写回 CSV。

不依赖 app/ 模块，可直接执行。

用法:
  python qc_vision_standalone.py \
    --csv 语言教学课_ea79a3b9_records_run04_keep.csv \
    --mode video_frames \
    --backend local \
    --threads 1 \
    --max-rows 10

  # storyboard 模式（不下载视频，从 YouTube 元数据取预览图，更快）:
  python qc_vision_standalone.py \
    --csv 语言教学课_ea79a3b9_records_run04_keep.csv \
    --mode storyboard \
    --backend local

前置条件:
  - ffmpeg + ffprobe（brew install ffmpeg）
  - yt-dlp（pip install yt-dlp）
  - Ollama 运行中 + 已拉取视觉模型（默认 qwen3-vl:8b）
    ollama pull qwen3-vl:8b
  - pip install httpx openai
"""

from __future__ import annotations

import argparse
import base64
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# 1. 轻量工具函数
# ---------------------------------------------------------------------------

BEIJING_TZ = ZoneInfo("Asia/Shanghai")


def now_beijing_iso() -> str:
    return datetime.now(BEIJING_TZ).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 2. 配置（可从 .env 加载，未设置则用默认值）
# ---------------------------------------------------------------------------

# 尝试加载项目根 .env（不影响独立运行）
try:
    from dotenv import load_dotenv

    _SCRIPT_DIR = Path(__file__).resolve().parent
    load_dotenv(_SCRIPT_DIR / ".env")
except ImportError:
    pass

# --- 工作目录 ---
WORK_DIR = Path(os.getenv("QC_WORK_DIR", str(Path(__file__).resolve().parent / "storage" / "work")))
WORK_DIR.mkdir(parents=True, exist_ok=True)

# --- yt-dlp ---
YT_EXTRACTOR_ARGS = ["--extractor-args", "youtube:player_client=android,web"]


def yt_dlp_cookie_args() -> list[str]:
    """Cookie 参数：优先 cookies.txt，否则浏览器。"""
    file_env = os.getenv("YT_DLP_COOKIES_FILE", "").strip()
    if file_env and Path(file_env).expanduser().is_file():
        return ["--cookies", str(Path(file_env).expanduser())]
    cookies_txt = Path(__file__).resolve().parent / "cookies.txt"
    if cookies_txt.is_file():
        return ["--cookies", str(cookies_txt)]
    browser = os.getenv("YT_DLP_COOKIES_FROM_BROWSER", "").strip()
    if browser:
        return ["--cookies-from-browser", browser]
    return []


# --- 视觉模型 ---
QC_VISION_BACKEND = os.getenv("QC_VISION_BACKEND", "local").strip().lower()
QC_VISION_LOCAL_BASE = os.getenv("QC_VISION_LOCAL_BASE_URL", "http://127.0.0.1:11434/v1").strip()
QC_VISION_LOCAL_MODEL = os.getenv("QC_VISION_LOCAL_MODEL", "qwen3-vl:8b").strip()
QC_VISION_THINK = os.getenv("QC_VISION_THINK", "false").strip().lower() in {"1", "true", "yes", "on"}
QC_VISION_API_BASE = os.getenv(
    "QC_VISION_API_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
).strip()
QC_VISION_API_MODEL = os.getenv("QC_VISION_API_MODEL", "qwen3-vl-flash").strip()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama").strip() or "ollama"
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "90").strip() or 90)

# --- 下载 / 抽帧 ---
DEFAULT_VIDEO_SECONDS = 30
DEFAULT_NUM_FRAMES = 5
MAX_VIDEO_HEIGHT = 480
YT_DLP_META_TIMEOUT = int(os.getenv("YT_DLP_META_TIMEOUT_SEC", "55").strip() or 55)

# --- 质检 Prompt ---
DEFAULT_PROMPT = os.getenv(
    "QC_VISION_PROMPT",
    """你是一位视频内容审核专家，专门评估化妆美妆教程类视频。下面是从一段视频中抽取的若干帧画面。

请根据这些画面，逐条判断以下三项（先给「是」「否」判断，再简要说明）。最后单独一行输出最终判定。

1. 视频中是否属于单人在给自己化妆美妆？（不是给别人化，不是多人）
2. 视频中人物是否在讲解化妆美妆的步骤？（口述或字幕均可，纯音乐无声不算）
3. 视频是否一直有完整的人存在（必须有完整头部，不能遮挡或缺失）？

---- 请严格按照以下格式回复（不要添加额外寒暄） ----
1. 是/否 | 一句依据
2. 是/否 | 一句依据
3. 是/否 | 一句依据
最终判定: T 或 F
简述: 一句话总结判定理由
----
判定规则: 三项全部为「是」则输出 T，任一项为「否」或「不确定」则输出 F。""",
).strip()


# ---------------------------------------------------------------------------
# 3. 视频下载（yt-dlp）
# ---------------------------------------------------------------------------

def download_youtube_segment(
    url: str,
    seconds: int = DEFAULT_VIDEO_SECONDS,
    max_height: int = MAX_VIDEO_HEIGHT,
    out_dir: Path | None = None,
) -> Path:
    out_dir = Path(out_dir or WORK_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-.]", "_", url.strip())[:80]
    output_template = str(out_dir / f"segment_{safe}.%(ext)s")
    cookies = yt_dlp_cookie_args()

    fmt = (
        f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/"
        f"bv*[height<={max_height}]+ba/"
        f"b[height<={max_height}]/best"
    )
    cmd = [
        "yt-dlp",
        *cookies,
        *YT_EXTRACTOR_ARGS,
        "--force-ipv4",
        "--retries", "6",
        "--fragment-retries", "6",
        "--socket-timeout", "20",
        "--no-check-certificates",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--external-downloader", "ffmpeg",
        "--external-downloader-args", f"ffmpeg:-t {seconds}",
        "-o", output_template,
        "--no-playlist",
        "--quiet", "--no-warnings",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode == 0:
        for f in out_dir.glob(f"segment_{safe}.*"):
            return f

    # 备用方案：下载完整视频再裁切
    cmd2 = [
        "yt-dlp",
        *cookies,
        *YT_EXTRACTOR_ARGS,
        "--force-ipv4",
        "--retries", "6",
        "--fragment-retries", "6",
        "--socket-timeout", "20",
        "--no-check-certificates",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--no-playlist",
        "-o", output_template,
        "--quiet", "--no-warnings",
        url,
    ]
    result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=180)
    if result2.returncode != 0:
        raise RuntimeError(
            f"yt-dlp 下载失败: {result.stderr or result.stdout}\n"
            f"备用方案也失败: {result2.stderr or result2.stdout}"
        )
    full_path = next((f for f in out_dir.glob(f"segment_{safe}.*")), None)
    if not full_path:
        raise FileNotFoundError(f"未找到下载文件: segment_{safe}")
    out_path = out_dir / f"segment_{safe}_trim.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(full_path), "-t", str(seconds), "-c", "copy", str(out_path)],
        capture_output=True,
        timeout=60,
    )
    try:
        full_path.unlink()
    except OSError:
        pass
    if not out_path.is_file():
        raise RuntimeError("ffmpeg 裁切前 N 秒失败")
    return out_path


# ---------------------------------------------------------------------------
# 4. 抽帧（ffmpeg）
# ---------------------------------------------------------------------------

def _check_ffmpeg() -> None:
    missing = [c for c in ("ffprobe", "ffmpeg") if shutil.which(c) is None]
    if missing:
        raise RuntimeError(
            f"未找到 {'、'.join(missing)}。请安装 FFmpeg: brew install ffmpeg"
        )


def extract_frames_as_base64(
    video_path: Path,
    num_frames: int = DEFAULT_NUM_FRAMES,
    work_dir: Path | None = None,
) -> list[str]:
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    _check_ffmpeg()

    work_dir = Path(work_dir or WORK_DIR)
    work_dir.mkdir(parents=True, exist_ok=True)
    prefix = work_dir / f"frame_{video_path.stem}_"

    # ffprobe 获取总帧数
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0", str(video_path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    total = 0
    if probe.returncode == 0 and probe.stdout.strip().isdigit():
        total = int(probe.stdout.strip())
    if total < num_frames:
        total = max(total, num_frames)

    indices = [
        int(i * (total - 1) / (num_frames - 1)) if num_frames > 1 else 0
        for i in range(num_frames)
    ]
    select_expr = "+".join(f"eq(n,{i})" for i in indices)
    out_pattern = str(prefix) + "%04d.jpg"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
        "-vsync", "vfr", "-q:v", "2", out_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 抽帧失败: {result.stderr or result.stdout}")

    b64_list = []
    for i in range(num_frames):
        path = Path(f"{prefix}{i+1:04d}.jpg")
        if not path.is_file():
            candidates = list(work_dir.glob(f"frame_{video_path.stem}_*.jpg"))
            if not candidates:
                raise FileNotFoundError(f"未找到抽帧输出: {prefix}")
            path = sorted(candidates)[min(i, len(candidates) - 1)]
        b64_list.append(base64.b64encode(path.read_bytes()).decode("ascii"))
        try:
            path.unlink()
        except OSError:
            pass
    return b64_list


# ---------------------------------------------------------------------------
# 5. YouTube 元数据 / storyboard
# ---------------------------------------------------------------------------

import json
import ssl
from urllib.request import Request, urlopen


def _yt_dlp_dump_json(url: str, timeout_sec: int = 60) -> dict:
    cookies = yt_dlp_cookie_args()
    cmd = [
        "yt-dlp", *cookies, *YT_EXTRACTOR_ARGS,
        "--dump-single-json", "--skip-download",
        "--no-warnings", "--quiet", url,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    if p.returncode != 0:
        raise RuntimeError(f"yt-dlp 读取元数据失败: {p.stderr or p.stdout}")
    try:
        return json.loads(p.stdout or "{}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"元数据 JSON 解析失败: {e}") from e


def _pick_storyboard_urls(meta: dict, max_images: int, prefer_medium: bool = False) -> list[str]:
    cands: list[tuple[int, str]] = []
    seen: set[str] = set()

    def add(u: str, score: int) -> None:
        u = str(u or "").strip()
        if not u or u in seen:
            return
        ul = u.lower()
        if "/sb/" not in ul and "storyboard" not in ul:
            return
        # 过滤 yt-dlp 模板 URL（路径含 $M/$N/$L 等占位符，直接下载会 404）；
        # sqp 查询参数里也可能含 $，只检查路径部分。
        path = u.split("?")[0]
        if "$" in path:
            return
        seen.add(u)
        cands.append((score, u))

    for t in (meta.get("thumbnails") or []):
        if isinstance(t, dict):
            add(str(t.get("url") or ""), int(t.get("width") or 0) * int(t.get("height") or 0))
    for f in (meta.get("formats") or []):
        if not isinstance(f, dict):
            continue
        fid = str(f.get("format_id") or "").lower()
        if not (fid.startswith("sb") or "storyboard" in str(f.get("format_note") or "").lower()):
            continue
        w, h = int(f.get("width") or 0), int(f.get("height") or 0)
        rows, cols = int(f.get("rows") or 1), int(f.get("columns") or 1)
        score = w * h * max(1, rows * cols)
        for fr in (f.get("fragments") or []):
            if isinstance(fr, dict):
                add(str(fr.get("url") or ""), score)
        add(str(f.get("url") or ""), score)

    if not cands:
        return []
    cands.sort(key=lambda x: x[0], reverse=True)
    urls = [u for _, u in cands]
    k = max(1, int(max_images))
    if k == 1 or prefer_medium:
        return [urls[len(urls) // 2]]
    if len(urls) <= k:
        return urls
    return [urls[int(i * (len(urls) - 1) / (k - 1))] for i in range(k)]


def extract_storyboard_as_base64(
    url: str,
    max_images: int = 1,
    prefer_medium: bool = True,
    timeout_sec: int = 60,
) -> list[str]:
    meta = _yt_dlp_dump_json(url, timeout_sec=timeout_sec)
    urls = _pick_storyboard_urls(meta, max_images=max_images, prefer_medium=prefer_medium)
    if not urls:
        raise RuntimeError("未找到可用 storyboard 图片")
    out: list[str] = []
    ssl_ctx = ssl._create_unverified_context()
    for u in urls:
        req = Request(u, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout_sec, context=ssl_ctx) as resp:
            out.append(base64.b64encode(resp.read()).decode("ascii"))
    return out


# ---------------------------------------------------------------------------
# 6. 视觉模型调用（支持本地 Ollama 和云端 API）
# ---------------------------------------------------------------------------

def _host_from_base_url(base_url: str) -> str:
    u = (base_url or "").strip().rstrip("/")
    return u[:-3] if u.endswith("/v1") else u


def evaluate_frames_local_openai(image_base64_list: list[str], prompt: str) -> str:
    """通过 Ollama OpenAI 兼容 /v1/chat/completions 调用本地视觉模型。"""
    import httpx
    from openai import OpenAI

    base_url = QC_VISION_LOCAL_BASE
    model = QC_VISION_LOCAL_MODEL
    api_key = OLLAMA_API_KEY

    content: list[dict] = []
    for b64 in image_base64_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})

    to = float(LLM_TIMEOUT_SEC)
    http_client = httpx.Client(
        timeout=httpx.Timeout(to, connect=10.0, pool=5.0),
        trust_env=False,  # 本机 Ollama 不走代理
    )
    client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
    extra = {"think": QC_VISION_THINK} if QC_VISION_THINK else {}
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        stream=False,
        **(extra if extra else {}),
    )
    return (resp.choices[0].message.content or "").strip()


def evaluate_frames_api(image_base64_list: list[str], prompt: str) -> str:
    """通过云端 API（DashScope 百炼）调用视觉模型。"""
    import httpx
    from openai import OpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("未设置 DASHSCOPE_API_KEY")

    base_url = QC_VISION_API_BASE
    model = QC_VISION_API_MODEL

    content: list[dict] = []
    for b64 in image_base64_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})

    to = float(LLM_TIMEOUT_SEC)
    http_client = httpx.Client(
        timeout=httpx.Timeout(to, connect=10.0, pool=5.0),
        trust_env=True,
    )
    client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
    extra = {"think": QC_VISION_THINK} if QC_VISION_THINK else {}
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        stream=False,
        **({"extra_body": extra} if extra else {}),
    )
    return (resp.choices[0].message.content or "").strip()


def evaluate_frames(image_base64_list: list[str], prompt: str, backend: str = "local") -> str:
    if backend == "api":
        return evaluate_frames_api(image_base64_list, prompt)
    return evaluate_frames_local_openai(image_base64_list, prompt)


def parse_qc_response(text: str) -> tuple[str, str]:
    """从模型回复中提取 T/F 判定和简述。"""
    verdict = "F"
    summary = ""

    # 提取最终判定: T 或 F
    m = re.search(r"最终判定[：:]\s*([TFtf])", text)
    if m:
        verdict = m.group(1).upper()

    # 提取简述
    m = re.search(r"简述[：:]\s*(.+?)(?:\n|$)", text)
    if m:
        summary = m.group(1).strip()
    else:
        # 兜底：取第一条判断的前 80 字
        m = re.search(r"1\.\s*(.+?)(?:\n|$)", text)
        if m:
            summary = m.group(1).strip()[:80]

    return verdict, summary


# ---------------------------------------------------------------------------
# 7. 单条视频质检流水线
# ---------------------------------------------------------------------------

def run_single_qc(
    url: str,
    *,
    mode: str = "video_frames",
    backend: str = "local",
    seconds: int = DEFAULT_VIDEO_SECONDS,
    num_frames: int = DEFAULT_NUM_FRAMES,
    prompt: str | None = None,
    quiet: bool = False,
) -> str:
    """对单条视频执行质检，返回文本结果。"""
    prompt = (prompt or DEFAULT_PROMPT).strip()
    if not prompt:
        raise ValueError("质检指令为空")
    mode = (mode or "video_frames").strip().lower()
    backend = (backend or "local").strip().lower()

    if mode == "storyboard":
        if not quiet:
            print("  [storyboard] 提取中...")
        frames_b64 = extract_storyboard_as_base64(url, max_images=1, prefer_medium=True)
    else:
        if not quiet:
            print(f"  [download] 下载前{seconds}秒...")
        video_path = download_youtube_segment(url, seconds=seconds)
        if not quiet:
            print(f"  [frames] 抽{num_frames}帧...")
        frames_b64 = extract_frames_as_base64(video_path, num_frames=num_frames)
        try:
            video_path.unlink()
        except OSError:
            pass

    if not quiet:
        print(f"  [vision] 调用{backend}视觉模型...")
    return evaluate_frames(frames_b64, prompt, backend=backend)


# ---------------------------------------------------------------------------
# 8. 批量 CSV 处理
# ---------------------------------------------------------------------------

def read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """读取 CSV，返回 (字段名列表, 行数据列表)。"""
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def write_csv_atomic(csv_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """原子写入 CSV（先写临时文件，再替换）。"""
    tmp = csv_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(csv_path)


def run_batch_qc(
    csv_path: Path,
    *,
    mode: str = "video_frames",
    backend: str = "local",
    seconds: int = DEFAULT_VIDEO_SECONDS,
    num_frames: int = DEFAULT_NUM_FRAMES,
    prompt: str | None = None,
    max_rows: int = 0,
    threads: int = 1,
    only_missing: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    """
    批量质检主函数。

    参数:
      csv_path: CSV 文件路径
      mode: video_frames / storyboard
      backend: local / api
      max_rows: 最多处理行数（0=不限制）
      threads: 并发线程数（本地 Ollama 建议 1）
      only_missing: 仅处理 qc_result 为空的行
      resume: 同上，别名兼容
    """
    print(f"\n{'='*60}")
    print(f"批量视觉质检")
    print(f"CSV: {csv_path}")
    print(f"模式: {mode} | 后端: {backend} | 线程: {threads}")
    if max_rows > 0:
        print(f"最多处理: {max_rows} 行")
    print(f"{'='*60}\n")

    fieldnames, rows = read_csv_rows(csv_path)
    total_csv = len(rows)
    print(f"CSV 总行数: {total_csv}")

    # 确保 QC 字段存在
    for fn in ("qc_status", "qc_result", "qc_summary", "qc_updated_at"):
        if fn not in fieldnames:
            fieldnames.append(fn)

    # 筛选待处理行
    to_process: list[tuple[int, dict[str, str]]] = []
    for i, row in enumerate(rows):
        url = (row.get("url") or "").strip()
        if not url or "youtube" not in url.lower():
            continue
        if (only_missing or resume) and (row.get("qc_result") or "").strip():
            continue
        to_process.append((i, row))
        if max_rows > 0 and len(to_process) >= max_rows:
            break

    total = len(to_process)
    if total == 0:
        print("没有需要质检的行（所有行已有 qc_result，或没有有效 YouTube URL）。")
        return {"done": 0, "total": 0}

    print(f"待质检: {total} 行（跳过已有结果的行）\n")

    done_count = 0
    error_count = 0
    start_time = time.time()
    lock = threading.Lock()

    def process_one(idx: int, row: dict[str, str]) -> tuple[int, str, str, str]:
        url = (row.get("url") or "").strip()
        title = (row.get("title") or "")[:60]
        req_dir = Path(tempfile.mkdtemp(prefix="qc_", dir=str(WORK_DIR)))
        try:
            print(f"[{done_count+1}/{total}] {title}...")
            raw = run_single_qc(
                url,
                mode=mode,
                backend=backend,
                seconds=seconds,
                num_frames=num_frames,
                prompt=prompt,
                quiet=False,
            )
            verdict, summary = parse_qc_response(raw)
            return idx, "ok", verdict, summary
        except Exception:
            err = traceback.format_exc()
            err_short = "\n".join(err.strip().split("\n")[-5:])
            print(f"  [ERROR] {err_short[:200]}")
            return idx, "error", "F", err_short[:200]
        finally:
            shutil.rmtree(req_dir, ignore_errors=True)

    if threads > 1:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futures = {
                ex.submit(process_one, idx, row): idx
                for idx, row in to_process
            }
            for fut in as_completed(futures):
                i, status, verdict, summary = fut.result()
                with lock:
                    rows[i]["qc_status"] = status
                    rows[i]["qc_result"] = verdict
                    rows[i]["qc_summary"] = summary
                    rows[i]["qc_updated_at"] = now_beijing_iso()
                    done_count += 1
                    if status == "error":
                        error_count += 1
                    if done_count % 10 == 0:
                        print(f"  --- 进度: {done_count}/{total} (错误: {error_count}) ---")
                        write_csv_atomic(csv_path, fieldnames, rows)
    else:
        for idx, row in to_process:
            i, status, verdict, summary = process_one(idx, row)
            with lock:
                rows[i]["qc_status"] = status
                rows[i]["qc_result"] = verdict
                rows[i]["qc_summary"] = summary
                rows[i]["qc_updated_at"] = now_beijing_iso()
                done_count += 1
                if status == "error":
                    error_count += 1
            # 每 10 条保存一次
            if done_count % 10 == 0:
                write_csv_atomic(csv_path, fieldnames, rows)
                elapsed = time.time() - start_time
                print(f"  --- 进度: {done_count}/{total} (错误: {error_count}) | {elapsed:.0f}s ---")

    # 最终写入
    write_csv_atomic(csv_path, fieldnames, rows)
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"质检完成！共 {done_count} 条，成功 {done_count-error_count}，失败 {error_count}")
    print(f"总耗时: {elapsed:.0f}s | 结果已写回 {csv_path}")
    print(f"{'='*60}")
    return {"done": done_count, "total": total, "errors": error_count, "elapsed_sec": elapsed}


# ---------------------------------------------------------------------------
# 9. CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="独立视觉质检脚本 —— YouTube 视频抽帧 + 本地视觉模型 QC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 用 storyboard 模式快速质检前 10 条（不下载视频）
  python qc_vision_standalone.py --csv 语言教学课_ea79a3b9_records_run04_keep.csv --mode storyboard --max-rows 10

  # 用 video_frames 模式完整质检（下载30秒→抽5帧）
  python qc_vision_standalone.py --csv 语言教学课_ea79a3b9_records_run04_keep.csv --mode video_frames

  # 指定云端 API
  python qc_vision_standalone.py --csv xxx.csv --backend api --threads 3

前置条件:
  - ffmpeg + ffprobe: brew install ffmpeg
  - yt-dlp: pip install yt-dlp
  - Ollama: ollama pull qwen3-vl:8b（本地视觉模型）
  - pip install httpx openai
        """,
    )
    parser.add_argument("--csv", required=True, help="CSV 文件路径")
    parser.add_argument(
        "--mode", default="video_frames",
        choices=["video_frames", "storyboard"],
        help="质检模式: video_frames=下载片段抽帧, storyboard=元数据预览图（更快）",
    )
    parser.add_argument(
        "--backend", default="local",
        choices=["local", "api"],
        help="视觉模型后端: local=本机 Ollama, api=云端 DashScope",
    )
    parser.add_argument("--seconds", type=int, default=DEFAULT_VIDEO_SECONDS,
                        help=f"下载视频前 N 秒（默认 {DEFAULT_VIDEO_SECONDS}）")
    parser.add_argument("--frames", type=int, default=DEFAULT_NUM_FRAMES,
                        help=f"抽帧数（默认 {DEFAULT_NUM_FRAMES}）")
    parser.add_argument("--prompt", default=None, help="自定义质检 Prompt（默认内置中文 Prompt）")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="最多处理行数（0=不限制，适合测试）")
    parser.add_argument("--threads", type=int, default=1,
                        help="并发线程数（本地 Ollama 建议 1）")
    parser.add_argument("--no-resume", action="store_true",
                        help="即使已有 qc_result 也重新质检")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"错误: CSV 文件不存在: {csv_path}")
        sys.exit(1)

    run_batch_qc(
        csv_path=csv_path,
        mode=args.mode,
        backend=args.backend,
        seconds=args.seconds,
        num_frames=args.frames,
        prompt=args.prompt,
        max_rows=args.max_rows,
        threads=args.threads,
        only_missing=not args.no_resume,
    )


if __name__ == "__main__":
    main()

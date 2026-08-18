#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
# -*- coding: utf-8 -*-
"""vlog 片头层 0：低清流 + ffmpeg 抽帧 + 动静代理。不调用 VL。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from core.yt_dlp_auth import (  # noqa: E402
    YtDlpAuth,
    detect_js_runtimes,
    prefetch_browser_cookies,
    resolve_yt_dlp_auth,
)

FFMPEG_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


@dataclass
class Sample:
    index: int
    window: int
    timestamp: float
    path: str
    read_ok: bool


@dataclass
class Result:
    video_id: str
    url: str
    status: str
    reason: str = ""
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    sample_count: int = 0
    valid_sample_count: int = 0
    visual_change_rate: Optional[float] = None
    mean_change: Optional[float] = None
    median_change: Optional[float] = None
    static_window_ratio: Optional[float] = None
    dynamic_window_ratio: Optional[float] = None
    frame_paths: Optional[list[str]] = None


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found in PATH: {name}")


def yt_dlp_base_args(auth: YtDlpAuth | None) -> list[str]:
    args = ["yt-dlp", "--no-warnings", "--no-playlist"]
    if auth and auth.cookies_file:
        args.extend(["--cookies", auth.cookies_file])
    elif auth and auth.cookies_from_browser:
        spec = ":".join(x for x in auth.cookies_from_browser if x)
        args.extend(["--cookies-from-browser", spec])
    js = detect_js_runtimes()
    if js:
        runtime = "deno" if "deno" in js else next(iter(js))
        args.extend(["--js-runtimes", runtime])
    return args


def run_cmd(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def get_video_info(url: str, auth: YtDlpAuth | None) -> dict:
    cmd = yt_dlp_base_args(auth) + [
        "--skip-download",
        "--dump-single-json",
        url,
    ]
    p = run_cmd(cmd, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip()[-2000:] or "yt-dlp metadata failed")
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("yt-dlp returned invalid JSON") from exc


def get_lowres_stream_url(
    url: str, auth: YtDlpAuth | None, max_height: int = 360,
) -> str:
    format_selector = (
        f"bv*[height<={max_height}]/"
        f"bv*[height<=?{max_height}]/"
        "bv*"
    )
    cmd = yt_dlp_base_args(auth) + ["-f", format_selector, "-g", url]
    p = run_cmd(cmd, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip()[-2000:] or "yt-dlp -g failed")
    urls = [x.strip() for x in p.stdout.splitlines() if x.strip()]
    if not urls:
        raise RuntimeError("No playable video stream URL returned by yt-dlp")
    return urls[0]


def sample_plan(
    duration: float,
    total_seconds: float = 60.0,
    windows: int = 5,
    frames_per_window: int = 5,
    margin: float = 0.5,
) -> list[tuple[int, float]]:
    """返回 (window, timestamp)。每窗约 3s 连续观察。"""
    usable = min(total_seconds, max(0.0, duration))
    if usable <= 0:
        return []

    actual_windows = min(windows, max(1, int(math.ceil(usable / 12.0))))
    window_len = usable / actual_windows
    result: list[tuple[int, float]] = []
    for w in range(actual_windows):
        start = w * window_len
        end = (w + 1) * window_len
        span = min(3.0, max(0.2, end - start - 2 * margin))
        center = start + margin + span / 2
        if frames_per_window == 1:
            ts = [min(center, usable - 0.05)]
        else:
            ts = np.linspace(
                start + margin,
                min(start + margin + span, usable - 0.05),
                frames_per_window,
            ).tolist()
        result.extend((w, float(x)) for x in ts)
    return result


def ffmpeg_proxy_args() -> list[str]:
    """ffmpeg 默认不读 HTTP(S)_PROXY；流 URL 绑代理出口 IP，直连会 4xx/5xx。"""
    proxy = (
        os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
        or ""
    ).strip()
    if proxy:
        return ["-http_proxy", proxy]
    return []


def extract_frames(
    stream_url: str,
    output_dir: Path,
    plan: list[tuple[int, float]],
    width: int = 640,
) -> list[Sample]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for leftover in output_dir.glob("frame_*.jpg"):
        leftover.unlink(missing_ok=True)
    timestamps = [t for _, t in plan]
    select_expr = "+".join(
        f"between(t\\,{t - 0.04:.3f}\\,{t + 0.04:.3f})"
        for t in timestamps
    )
    pattern = str(output_dir / "frame_%03d.jpg")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-user_agent", FFMPEG_UA,
        *ffmpeg_proxy_args(),
        "-i", stream_url,
        "-an",
        "-t", str(max(timestamps) + 1.0),
        "-vf", f"scale={width}:-2,select='{select_expr}'",
        "-vsync", "vfr",
        "-q:v", "3",
        pattern,
    ]
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=240,
            check=False,
        )
        err = (p.stderr or "").strip()[-2000:]
    except subprocess.TimeoutExpired as exc:
        err = f"TIMEOUT after 240s; {(exc.stderr or b'')[-500:]!r}"

    files = sorted(output_dir.glob("frame_*.jpg"))
    if not files:
        raise RuntimeError(f"ffmpeg produced no frames. {err}")

    n = min(len(files), len(plan))
    samples: list[Sample] = []
    for i in range(n):
        window, ts = plan[i]
        samples.append(
            Sample(
                index=i,
                window=window,
                timestamp=float(ts),
                path=str(files[i]),
                read_ok=True,
            )
        )
    return samples


def frame_change_score(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ga = cv2.resize(ga, (160, 90))
    gb = cv2.resize(gb, (160, 90))
    diff = cv2.absdiff(ga, gb)
    return float(np.mean(diff) / 255.0)


def analyze_samples(samples: list[Sample], windows: int = 5) -> dict:
    valid = []
    for s in samples:
        img = cv2.imread(s.path)
        if img is not None:
            valid.append((s, img))

    if len(valid) < 2:
        return {
            "valid_sample_count": len(valid),
            "visual_change_rate": None,
            "mean_change": None,
            "median_change": None,
            "static_window_ratio": None,
            "dynamic_window_ratio": None,
            "changes": [],
            "window_stats": [],
        }

    changes = []
    for (s1, i1), (s2, i2) in zip(valid[:-1], valid[1:]):
        changes.append({
            "t1": s1.timestamp,
            "t2": s2.timestamp,
            "score": frame_change_score(i1, i2),
            "window": s1.window,
        })

    change_threshold = 0.08
    dynamic = sum(x["score"] >= change_threshold for x in changes)
    change_rate = dynamic / len(changes)

    grouped: dict[int, list[float]] = {}
    for x in changes:
        grouped.setdefault(int(x["window"]), []).append(float(x["score"]))

    window_stats = []
    for w in range(windows):
        vals = grouped.get(w, [])
        if not vals:
            continue
        window_stats.append({
            "window": w,
            "mean_change": float(np.mean(vals)),
            "max_change": float(np.max(vals)),
            "dynamic": bool(np.mean(vals) >= change_threshold),
        })

    dynamic_window_count = sum(x["dynamic"] for x in window_stats)
    static_window_count = len(window_stats) - dynamic_window_count
    return {
        "valid_sample_count": len(valid),
        "visual_change_rate": float(change_rate),
        "mean_change": float(np.mean([x["score"] for x in changes])),
        "median_change": float(np.median([x["score"] for x in changes])),
        "static_window_ratio": (
            static_window_count / len(window_stats) if window_stats else None
        ),
        "dynamic_window_ratio": (
            dynamic_window_count / len(window_stats) if window_stats else None
        ),
        "changes": changes,
        "window_stats": window_stats,
    }


def _slim_info(info: dict) -> dict:
    return {
        "id": info.get("id"),
        "duration": info.get("duration"),
        "width": info.get("width"),
        "height": info.get("height"),
        "fps": info.get("fps"),
        "title": (info.get("title") or "")[:200],
    }


def result_from_debug(debug_path: Path, url: str) -> Result | None:
    try:
        data = json.loads(debug_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    samples = data.get("samples") or []
    paths = []
    for s in samples:
        p = Path(s.get("path") or "")
        if s.get("read_ok") and p.exists():
            paths.append(str(p))
    if len(paths) < 2:
        return None
    analysis = data.get("analysis") or {}
    info = data.get("info") or {}
    return Result(
        video_id=str(data.get("video_id") or debug_path.parent.name),
        url=url,
        status="READY_FOR_VL",
        reason="RESUMED",
        duration=info.get("duration"),
        width=info.get("width"),
        height=info.get("height"),
        sample_count=len(samples),
        valid_sample_count=int(analysis.get("valid_sample_count") or len(paths)),
        visual_change_rate=analysis.get("visual_change_rate"),
        mean_change=analysis.get("mean_change"),
        median_change=analysis.get("median_change"),
        static_window_ratio=analysis.get("static_window_ratio"),
        dynamic_window_ratio=analysis.get("dynamic_window_ratio"),
        frame_paths=paths,
    )


def process_one(
    video_id: str,
    url: str,
    output_dir: Path,
    duration_limit: float,
    auth: YtDlpAuth | None,
    windows: int = 5,
    frames_per_window: int = 5,
    duration_hint: float | None = None,
) -> Result:
    try:
        frame_dir = output_dir / video_id
        debug_path = frame_dir / "debug.json"
        resumed = result_from_debug(debug_path, url)
        if resumed is not None:
            return resumed

        # 超时中断后可能已有部分 JPEG，够 VL 就不再拉流
        existing = sorted(frame_dir.glob("w*_t*.jpg"))
        if len(existing) < 5:
            existing = sorted(frame_dir.glob("frame_*.jpg"))
        if len(existing) >= 5:
            samples = []
            for i, fp in enumerate(existing):
                m = re.search(r"w(\d+)_t([0-9.]+)", fp.name)
                window = int(m.group(1)) if m else i // 5
                ts = float(m.group(2)) if m else float(i)
                samples.append(Sample(i, window, ts, str(fp), True))
            analysis = analyze_samples(samples, windows=windows)
            return Result(
                video_id=video_id,
                url=url,
                status="READY_FOR_VL",
                reason="PARTIAL_FRAMES",
                duration=duration_hint,
                sample_count=len(samples),
                valid_sample_count=int(analysis["valid_sample_count"]),
                visual_change_rate=analysis["visual_change_rate"],
                mean_change=analysis["mean_change"],
                median_change=analysis["median_change"],
                static_window_ratio=analysis["static_window_ratio"],
                dynamic_window_ratio=analysis["dynamic_window_ratio"],
                frame_paths=[s.path for s in samples],
            )

        info: dict = {}
        duration = float(duration_hint or 0.0)
        if duration <= 0:
            info = get_video_info(url, auth)
            duration = float(info.get("duration") or 0.0)
        if duration <= 0:
            return Result(video_id, url, "REVIEW", "NO_DURATION")
        info.setdefault("duration", duration)
        info.setdefault("id", video_id)

        actual_end = min(duration_limit, duration)
        if actual_end < 4:
            return Result(
                video_id, url, "REVIEW",
                "TOO_SHORT_FOR_FIRST_MINUTE_SAMPLE",
                duration=duration,
            )

        plan = sample_plan(
            duration=duration,
            total_seconds=actual_end,
            windows=windows,
            frames_per_window=frames_per_window,
        )
        frame_dir.mkdir(parents=True, exist_ok=True)
        samples = None
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                stream_url = get_lowres_stream_url(url, auth, max_height=360)
                samples = extract_frames(stream_url, frame_dir, plan)
                break
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    continue
                raise
        if samples is None:
            raise last_exc or RuntimeError("extract_frames failed")

        for sample in samples:
            src = Path(sample.path)
            if not src.exists():
                continue
            target = frame_dir / (
                f"w{sample.window:02d}_t{sample.timestamp:07.3f}.jpg"
            )
            try:
                src.rename(target)
                sample.path = str(target)
            except OSError:
                pass

        analysis = analyze_samples(samples, windows=windows)
        result = Result(
            video_id=video_id,
            url=url,
            status="READY_FOR_VL",
            reason="SAMPLED",
            duration=duration,
            width=int(info["width"]) if info.get("width") else None,
            height=int(info["height"]) if info.get("height") else None,
            sample_count=len(samples),
            valid_sample_count=int(analysis["valid_sample_count"]),
            visual_change_rate=analysis["visual_change_rate"],
            mean_change=analysis["mean_change"],
            median_change=analysis["median_change"],
            static_window_ratio=analysis["static_window_ratio"],
            dynamic_window_ratio=analysis["dynamic_window_ratio"],
            frame_paths=[s.path for s in samples if s.read_ok],
        )
        debug = {
            "video_id": video_id,
            "url": url,
            "info": _slim_info(info),
            "timestamps": [t for _, t in plan],
            "samples": [asdict(s) for s in samples],
            "analysis": analysis,
        }
        debug_path.write_text(
            json.dumps(debug, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result
    except subprocess.TimeoutExpired:
        return Result(video_id, url, "REVIEW", "TIMEOUT")
    except Exception as exc:
        return Result(video_id, url, "REVIEW", f"{type(exc).__name__}: {exc}")


def read_input(path: Path, url_column: str) -> Iterable[tuple[str, str, float | None]]:
    if path.suffix.lower() in {".txt", ".list"}:
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                url = line.strip()
                if url:
                    yield str(i), url, None
        return

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if url_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Column '{url_column}' not found. Columns: {reader.fieldnames}"
            )
        for i, row in enumerate(reader):
            url = (row.get(url_column) or "").strip()
            if not url:
                continue
            video_id = (row.get("video_id") or row.get("id") or str(i)).strip()
            dur_raw = (row.get("duration_seconds") or row.get("duration") or "").strip()
            try:
                duration_hint = float(dur_raw) if dur_raw else None
            except ValueError:
                duration_hint = None
            yield video_id, url, duration_hint


def load_done_ids(jsonl_path: Path, retry_review: bool) -> set[str]:
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = str(rec.get("video_id") or "").strip()
            if not vid:
                continue
            status = str(rec.get("status") or "")
            if retry_review and status == "REVIEW":
                continue
            done.add(vid)
    return done


def main() -> int:
    parser = argparse.ArgumentParser(
        description="vlog 片头层 0：低清流 + ffmpeg 抽帧（不调 VL）",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--url-column", default="url")
    parser.add_argument("--output-dir", default="vlog_qc_frames")
    parser.add_argument("--result-jsonl", default="vlog_qc_results.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条；0=全部")
    parser.add_argument("-w", "--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-review",
        action="store_true",
        help="resume 时重跑 REVIEW",
    )
    parser.add_argument("--cookies", default=None)
    parser.add_argument("--cookies-from-browser", default=None)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--frames-per-window", type=int, default=5)
    parser.add_argument("--duration-limit", type=float, default=60.0)
    args = parser.parse_args()

    require_binary("yt-dlp")
    require_binary("ffmpeg")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(args.result_jsonl)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    auth = resolve_yt_dlp_auth(args.cookies, args.cookies_from_browser)
    if auth.cookies_from_browser and args.workers > 1:
        cache = str(output_dir / ".cookies_cache.txt")
        print(f"[cookies] 预取浏览器 cookies → {cache}", flush=True)
        auth = prefetch_browser_cookies(auth, cache)
    if auth.cookies_file:
        print(f"[cookies] file={auth.cookies_file}", flush=True)
    elif auth.cookies_from_browser:
        print(
            f"[cookies] browser={':'.join(auth.cookies_from_browser)}",
            flush=True,
        )

    rows = list(read_input(input_path, args.url_column))
    if args.limit > 0:
        rows = rows[: args.limit]

    done: set[str] = set()
    mode = "a" if args.resume and jsonl_path.exists() else "w"
    if args.resume:
        done = load_done_ids(jsonl_path, args.retry_review)
        print(f"[resume] 已跳过 {len(done)} 条", flush=True)

    pending = [
        (vid, url, dur) for vid, url, dur in rows if vid not in done
    ]
    print(
        f"[待处理] {len(pending)}/{len(rows)}  workers={args.workers}",
        flush=True,
    )

    write_lock = threading.Lock()
    n_ready = n_review = 0

    def handle(video_id: str, url: str, duration_hint: float | None) -> Result:
        return process_one(
            video_id=video_id,
            url=url,
            output_dir=output_dir,
            duration_limit=args.duration_limit,
            auth=auth,
            windows=args.windows,
            frames_per_window=args.frames_per_window,
            duration_hint=duration_hint,
        )

    with jsonl_path.open(mode, encoding="utf-8") as out:
        if args.workers <= 1:
            for n, (video_id, url, dur) in enumerate(pending, 1):
                result = handle(video_id, url, dur)
                out.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                out.flush()
                if result.status == "READY_FOR_VL":
                    n_ready += 1
                else:
                    n_review += 1
                print(
                    f"[{n}/{len(pending)}] {video_id} {result.status} "
                    f"{result.reason} samples={result.valid_sample_count} "
                    f"dyn={result.dynamic_window_ratio}",
                    flush=True,
                )
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {
                    ex.submit(handle, vid, url, dur): (vid, url)
                    for vid, url, dur in pending
                }
                done_n = 0
                for fut in as_completed(futs):
                    result = fut.result()
                    with write_lock:
                        out.write(
                            json.dumps(asdict(result), ensure_ascii=False) + "\n"
                        )
                        out.flush()
                        done_n += 1
                        if result.status == "READY_FOR_VL":
                            n_ready += 1
                        else:
                            n_review += 1
                    print(
                        f"[{done_n}/{len(pending)}] {result.video_id} "
                        f"{result.status} {result.reason} "
                        f"samples={result.valid_sample_count} "
                        f"dyn={result.dynamic_window_ratio}",
                        flush=True,
                    )

    print(f"[完成] READY_FOR_VL={n_ready} REVIEW={n_review} → {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

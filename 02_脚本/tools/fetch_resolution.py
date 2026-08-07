#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
"""
fetch_resolution.py — 从 YouTube 批量获取视频最高分辨率高度

用法:
  python3 fetch_resolution.py input.csv -o out.csv -w 2
  python3 fetch_resolution.py input.csv -n 30 --overwrite -w 2 -o out.csv

Bank 模式（自动跑完全部 bank 并汇总，无需手动接下一个）:
  python3 fetch_resolution.py input.csv --bank-size 10000 -o data/runs/_tmp/res_banks/ -w 2
  python3 fetch_resolution.py input.csv --bank-size 50 --max-banks 2 -o .../test_banks/ -w 2
  python3 fetch_resolution.py --merge-only -o data/runs/_tmp/res_banks/

  --bank-jobs 1  同时跑几个 bank 进程（默认 1=最稳最快实践；最多建议 2，且配合 -w 1）
  汇总: {out_dir}/merged_resolution.csv

Cookies:
  默认从 Chrome 导出到输出旁 .cookies_cache.txt；可用 --cookies / YT_DLP_COOKIES_FILE。
  有 cookies 时必须安装 deno（或 node）作 JS runtime，否则启动预检失败。
  bank 编排默认每个 bank 启动前刷新 cookies（--cookie-refresh-banks）；子进程读文件不抢浏览器锁。
  长跑建议专用 cookies（隐私窗登录→打开 robots.txt→导出→关掉该窗），勿用日常浏览会话。

断点续跑:
  进度写到 {output}.ckpt.csv（仅 video_id/max_height/fetch_status）；
  结束时再写完整 {output}。兼容旧版 {output}.part。
  no_height 为终态；empty_formats/error 默认可续跑。
  历史假 ok(height<144) 会迁成 empty_formats；真音频 nh 补跑请加 --retry-no-height。
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# 复用项目 core 模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.adaptive_api import AdaptiveConcurrencyGate  # noqa: E402
from core.yt_dlp_auth import (  # noqa: E402
    CookieManager,
    detect_js_runtimes,
    parse_browser_spec,
)
from core.progress import ThrottledProgress, mark_done  # noqa: E402
from core.sop import write_run_log  # noqa: E402

# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

YOUTUBE_URL_PREFIX = "https://www.youtube.com/watch?v="
SMOKE_VIDEO_ID = "jNQXAC9IVRw"  # 公开短视频，用于启动冒烟

# flush 触发条件
FLUSH_BATCH_SIZE = 100       # 每 N 条结果 flush 一次
FLUSH_INTERVAL_SEC = 30.0    # 或每 N 秒 flush 一次

# yt-dlp 默认配置
DEFAULT_WORKERS = 16
DEFAULT_SLEEP_INTERVAL = 0.5  # yt-dlp 请求间隔（秒）
MIN_VIDEO_HEIGHT = 144        # 低于此视为非有效视频清晰度（过滤 storyboard 90）

# cookies 刷新周期
COOKIE_REFRESH_SEC = 20 * 60  # 20 分钟
DEFAULT_COOKIE_REFRESH_BANKS = 1  # bank 编排：每 N 个 bank 刷新 cookies（0=仅启动时）

# 熔断
CIRCUIT_WINDOW = 50
CIRCUIT_FAIL_RATE = 0.80
CIRCUIT_MAX_NO_OK = 30

# fetch_status 枚举
STATUS_OK = "ok"
STATUS_NO_HEIGHT = "no_height"  # 元数据成功但无视频 height（音频等，终态）
STATUS_EMPTY_FORMATS = "empty_formats"  # formats 空/仅 storyboard（可重试）
STATUS_UNAVAILABLE = "unavailable"
STATUS_PRIVATE = "private"
STATUS_DELETED = "deleted"
STATUS_ERROR = "error"

# 终态：默认不再重跑（empty_formats / error 默认可续跑或 --retry-errors）
TERMINAL_STATUSES = {
    STATUS_OK,
    STATUS_NO_HEIGHT,
    STATUS_UNAVAILABLE,
    STATUS_PRIVATE,
    STATUS_DELETED,
}

CHECKPOINT_COLS = ("video_id", "max_height", "fetch_status")

_RE_RES_DIMS = re.compile(r"(\d{2,5})\s*[x×]\s*(\d{2,5})", re.IGNORECASE)
_RE_P_NOTE = re.compile(r"\b(\d{3,4})p\b", re.IGNORECASE)

# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class FetchResult:
    """单条视频的获取结果。"""
    idx: int                # DataFrame 行索引
    video_id: str           # YouTube video_id
    max_height: int | None  # 最高分辨率高度（像素，≥ MIN_VIDEO_HEIGHT）
    fetch_status: str       # ok / no_height / empty_formats / unavailable / private / deleted / error


@dataclass
class FetchStats:
    """获取统计计数器（线程安全）。"""
    ok: int = 0
    no_height: int = 0
    empty_formats: int = 0
    unavailable: int = 0
    private: int = 0
    deleted: int = 0
    error: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def count(self, status: str) -> None:
        with self._lock:
            if status == STATUS_OK:
                self.ok += 1
            elif status == STATUS_NO_HEIGHT:
                self.no_height += 1
            elif status == STATUS_EMPTY_FORMATS:
                self.empty_formats += 1
            elif status == STATUS_UNAVAILABLE:
                self.unavailable += 1
            elif status == STATUS_PRIVATE:
                self.private += 1
            elif status == STATUS_DELETED:
                self.deleted += 1
            else:
                self.error += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "ok": self.ok,
                "no_height": self.no_height,
                "empty_formats": self.empty_formats,
                "unavailable": self.unavailable,
                "private": self.private,
                "deleted": self.deleted,
                "error": self.error,
            }


@dataclass
class CircuitBreaker:
    """滑动窗口熔断：空 formats / error 占比过高或连续无 ok 时停跑。"""
    window: int = CIRCUIT_WINDOW
    fail_rate: float = CIRCUIT_FAIL_RATE
    max_no_ok: int = CIRCUIT_MAX_NO_OK
    _recent: list = field(default_factory=list)
    consecutive_no_ok: int = 0

    def record(self, status: str) -> bool:
        """记录一条结果；返回 True 表示应熔断。"""
        is_ok = status == STATUS_OK
        is_bad = status in (STATUS_EMPTY_FORMATS, STATUS_ERROR)
        self._recent.append(1 if is_bad else 0)
        if len(self._recent) > self.window:
            self._recent = self._recent[-self.window:]
        if is_ok:
            self.consecutive_no_ok = 0
        else:
            self.consecutive_no_ok += 1
        if self.consecutive_no_ok >= self.max_no_ok:
                return True
        if len(self._recent) >= self.window:
            rate = sum(self._recent) / len(self._recent)
            if rate >= self.fail_rate:
                return True
                return False


# ─────────────────────────────────────────────
# Thread-local YoutubeDL 实例
# ─────────────────────────────────────────────

_ydl_local = threading.local()


def _get_ydl(ydl_opts: dict[str, Any], cookie_ver: int = 0):
    """获取当前线程的 YoutubeDL 实例（thread-local 复用）。

    cookie_ver 用于检测 cookies 是否已刷新，若刷新则重建实例。
    必须传入 params 副本：YoutubeDL.__init__ 会原地修改 params，
    多线程共享同一 dict 会导致竞态（cookies/headers 错乱）。
    """
    import yt_dlp

    cookiefile = ydl_opts.get("cookiefile")
    sleep_iv = ydl_opts.get("sleep_interval_requests")
    js_key = tuple(sorted((ydl_opts.get("js_runtimes") or {}).keys()))
    opts_key = (cookiefile, sleep_iv, cookie_ver, js_key)

    if getattr(_ydl_local, "opts_key", None) != opts_key:
        # 浅拷贝顶层；嵌套 http_headers 等由 YoutubeDL 自行替换
        _ydl_local.ydl = yt_dlp.YoutubeDL(dict(ydl_opts))  # type: ignore[arg-type]
        _ydl_local.opts_key = opts_key

    return _ydl_local.ydl


# ─────────────────────────────────────────────
# 核心逻辑
# ─────────────────────────────────────────────

def _positive_int(val: Any) -> int | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        n = int(val)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _is_storyboard_format(f: dict) -> bool:
    fid = str(f.get("format_id") or "").lower()
    note = str(f.get("format_note") or "").lower()
    if fid.startswith("sb"):
        return True
    if "storyboard" in note:
        return True
    return False


def _height_from_resolution_str(text: Any) -> int | None:
    """从 '1920x1080' / '1080x1920' 取两边较大值（最高边，兼容竖屏）。"""
    if text is None:
        return None
    m = _RE_RES_DIMS.search(str(text))
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    h = max(a, b) if a > 0 and b > 0 else None
    if h is None or h < MIN_VIDEO_HEIGHT:
        return None
    return h


def _height_from_p_note(text: Any) -> int | None:
    """从 '1080p' / '720p' 等 note 提取高度。"""
    if text is None:
        return None
    m = _RE_P_NOTE.search(str(text))
    if not m:
        return None
    h = _positive_int(m.group(1))
    if h is None or h < MIN_VIDEO_HEIGHT:
        return None
    return h


def extract_max_height(info: Any) -> int | None:
    """从 yt-dlp extract_info 返回的 info dict 中提取最高分辨率高度。

    优先级：format.height → format.resolution → info.height → format_note/Np。
    忽略 storyboard 与 height < MIN_VIDEO_HEIGHT（过滤假清晰度 90）。
    """
    if not isinstance(info, dict):
        return None

    heights: list[int] = []
    formats = info.get("formats") or []
    if not isinstance(formats, list):
        formats = []

    for f in formats:
        if not isinstance(f, dict) or _is_storyboard_format(f):
            continue
        h = _positive_int(f.get("height"))
        if h is not None and h >= MIN_VIDEO_HEIGHT:
            heights.append(h)

    if heights:
        return max(heights)

    for f in formats:
        if not isinstance(f, dict) or _is_storyboard_format(f):
            continue
        h = _height_from_resolution_str(f.get("resolution"))
        if h is not None:
            heights.append(h)
    if heights:
        return max(heights)

    top = _positive_int(info.get("height"))
    if top is not None and top >= MIN_VIDEO_HEIGHT:
        return top

    for f in formats:
        if not isinstance(f, dict) or _is_storyboard_format(f):
            continue
        for key in ("format_note", "format", "format_id"):
            h = _height_from_p_note(f.get(key))
            if h is not None:
                heights.append(h)
    if heights:
        return max(heights)

    h = _height_from_resolution_str(info.get("resolution"))
    if h is not None:
        return h
    h = _height_from_p_note(info.get("format"))
    if h is not None:
        return h

    return None


def classify_missing_height(info: dict) -> str:
    """无有效 max_height 时：空/仅 storyboard → empty_formats；否则真音频等 → no_height。"""
    formats = info.get("formats") or []
    if not isinstance(formats, list):
        formats = []
    usable = [
        f for f in formats
        if isinstance(f, dict) and not _is_storyboard_format(f)
    ]
    if not usable:
        return STATUS_EMPTY_FORMATS
    has_video = any((f.get("vcodec") or "none") != "none" for f in usable)
    if has_video:
        # 有视频轨却无高度：当作抽流失败，可重试
        return STATUS_EMPTY_FORMATS
    return STATUS_NO_HEIGHT


def classify_download_error(msg: str) -> str:
    """根据 yt-dlp 错误信息分类 fetch_status。"""
    msg_lower = msg.lower()

    # 风控 / 限流：必须标为 error（可 --retry-errors），绝不能当成终态 unavailable
    # 旧逻辑用裸 "sign in to" 会把 "Sign in to confirm you're not a bot" 误判为 unavailable
    if (
        "not a bot" in msg_lower
        or "confirm you're not a bot" in msg_lower
        or "confirm you\u2019re not a bot" in msg_lower  # curly apostrophe
        or "http error 429" in msg_lower
        or "too many requests" in msg_lower
        or "rate-limit" in msg_lower
        or "rate limit" in msg_lower
    ):
        return STATUS_ERROR

    # 格式选择失败（我们只读元数据/formats 列表）→ 可重试，勿匹配到下面的 "not available"
    if "requested format is not available" in msg_lower or "format is not available" in msg_lower:
        return STATUS_ERROR

    if "private" in msg_lower:
        return STATUS_PRIVATE
    if "deleted" in msg_lower or "video has been removed" in msg_lower or "terminated" in msg_lower:
        return STATUS_DELETED

    # 会员专属
    if "members" in msg_lower and ("join this" in msg_lower or "channel's members" in msg_lower or "members-only" in msg_lower):
        return STATUS_UNAVAILABLE

    # 地区限制/不可用
    if "not available" in msg_lower or "not made this video available" in msg_lower:
        return STATUS_UNAVAILABLE
    if "in your country" in msg_lower or "in your region" in msg_lower:
        return STATUS_UNAVAILABLE
    if "unavailable" in msg_lower:
        return STATUS_UNAVAILABLE
    if "copyright" in msg_lower or "removed by" in msg_lower:
        return STATUS_UNAVAILABLE
    if "region" in msg_lower or "geo-block" in msg_lower or "geo restricted" in msg_lower:
        return STATUS_UNAVAILABLE

    # 年龄限制（避免裸匹配 age / 裸匹配 sign in to）
    if (
        "age-restricted" in msg_lower
        or "age restricted" in msg_lower
        or "confirm your age" in msg_lower
        or "sign in to confirm your age" in msg_lower
    ):
        return STATUS_UNAVAILABLE

    return STATUS_ERROR


def fetch_one(
    video_id: str,
    ydl_opts: dict[str, Any],
    gate: AdaptiveConcurrencyGate,
    cookie_mgr: CookieManager | None,
    idx: int,
) -> FetchResult:
    """获取单个视频的 max_height（在 ThreadPoolExecutor 中调用）。

    使用 AdaptiveConcurrencyGate 控制并发，遇 429 自动降速。
    """
    from yt_dlp.utils import DownloadError

    url = f"{YOUTUBE_URL_PREFIX}{video_id}"
    gate.acquire()
    try:
        cookie_ver = cookie_mgr.refresh_version if cookie_mgr else 0
        ydl = _get_ydl(ydl_opts, cookie_ver)
        try:
        info = ydl.extract_info(url, download=False)
        except DownloadError as e:
            # 默认 format 选择失败时，降级再试一次（只要 formats 列表能拿到 height）
            msg = str(e)
            if "format is not available" in msg.lower():
                retry_opts = dict(ydl_opts)
                retry_opts["format"] = "best"
                retry_opts["ignore_no_formats_error"] = True
                ydl_retry = _get_ydl(retry_opts, cookie_ver + 10_000)  # 强制新实例
                info = ydl_retry.extract_info(url, download=False)
            else:
                raise
        if not info:
            return FetchResult(
                idx=idx, video_id=video_id, max_height=None, fetch_status=STATUS_ERROR,
            )
        # ignore_no_formats_error 下会员视频不抛错，只有空 formats + availability
        availability = str(info.get("availability") or "").lower()
        if availability in {"subscriber_only", "premium_only", "needs_auth"}:
            return FetchResult(
                idx=idx, video_id=video_id, max_height=None, fetch_status=STATUS_UNAVAILABLE,
            )
        height = extract_max_height(info)
        if height is None:
            status = classify_missing_height(info)
            return FetchResult(
                idx=idx, video_id=video_id, max_height=None, fetch_status=status,
            )
        return FetchResult(idx=idx, video_id=video_id, max_height=height, fetch_status=STATUS_OK)
    except DownloadError as e:
        msg = str(e)
        status = classify_download_error(msg)
        if "429" in msg or "rate" in msg.lower():
            gate.on_rate_limit()
        return FetchResult(idx=idx, video_id=video_id, max_height=None, fetch_status=status)
    except Exception as e:
        msg = str(e)
        status = classify_download_error(msg)
        if "429" in msg or "rate" in msg.lower():
            gate.on_rate_limit()
        return FetchResult(idx=idx, video_id=video_id, max_height=None, fetch_status=status)
    finally:
        gate.release()



def _ensure_parent_dir(path: str) -> str:
    """确保 path 的父目录存在；返回父目录。"""
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    return parent


def flush_checkpoint(df: pd.DataFrame, ckpt_path: str) -> None:
    """只写进度列 sidecar，避免每次整表 flush。"""
    cols = [c for c in CHECKPOINT_COLS if c in df.columns]
    if "fetch_status" not in cols:
        return
    mask = df["fetch_status"].fillna("").astype(str).str.strip() != ""
    slim = df.loc[mask, cols]
    _ensure_parent_dir(ckpt_path)
    tmp = ckpt_path + ".tmp"
    slim.to_csv(tmp, index=False)
    os.replace(tmp, ckpt_path)


def write_final_csv(df: pd.DataFrame, final_path: str, ckpt_path: str | None = None) -> None:
    """写完整输出，并清理 sidecar / 旧 .part。"""
    _ensure_parent_dir(final_path)
    tmp = final_path + ".flush_tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, final_path)
    for side in (ckpt_path, final_path + ".part", final_path + ".ckpt.csv"):
        if not side:
            continue
        if os.path.abspath(side) == os.path.abspath(final_path):
            continue
        if os.path.exists(side):
            try:
                os.unlink(side)
            except OSError:
                pass


def build_ydl_opts(
    sleep_interval: float,
    cookie_file: str | None,
    js_runtimes: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """构建 yt-dlp YoutubeDL 选项。

    只需 formats 里的 height，不下载文件；放宽 format 选择，避免
    “Requested format is not available” 直接失败。
    """
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "sleep_interval_requests": sleep_interval,
        "extract_flat": False,
        "skip_download": True,
        "ignoreerrors": False,
        # 默认 bestvideo*+bestaudio 在部分 client/风控下会选不到流；我们不下载，用 best 即可
        "format": "best",
        "ignore_no_formats_error": True,
    }
    if cookie_file and os.path.exists(cookie_file):
        opts["cookiefile"] = cookie_file
    if js_runtimes:
        opts["js_runtimes"] = js_runtimes
    return opts


def preflight_js_runtime(*, using_cookies: bool, allow_skip: bool = False) -> dict[str, dict] | None:
    """有 cookies 时强制 JS runtime；返回 yt-dlp js_runtimes 或 None（匿名）。"""
    runtimes = detect_js_runtimes()
    if using_cookies and not runtimes:
        print(
            "[ERROR] 使用 cookies 时必须安装 JS runtime（推荐 deno），"
            "否则 formats 会被掏空 / 只剩 storyboard。\n"
            "  安装: https://docs.deno.com/runtime/getting_started/installation/\n"
            "  或: curl -fsSL https://deno.land/install.sh | sh\n"
            "  确认: deno --version && yt-dlp -F --cookies ... --js-runtimes deno URL"
        )
        if not allow_skip:
            sys.exit(1)
        return None
    if runtimes:
        name = next(iter(runtimes))
        print(f"JS runtime: {name}")
    elif using_cookies:
        print("[WARN] 未检测到 deno/node，cookies 场景极易空 formats")
    return runtimes


def smoke_extract_formats(
    ydl_opts: dict[str, Any],
    *,
    video_id: str = SMOKE_VIDEO_ID,
) -> None:
    """启动冒烟：公开片须能抽出有效视频 height，否则 exit。"""
    from yt_dlp.utils import DownloadError

    url = f"{YOUTUBE_URL_PREFIX}{video_id}"
    print(f"冒烟抽流: {url}")
    try:
        ydl = _get_ydl(ydl_opts, cookie_ver=-1)
        info = ydl.extract_info(url, download=False)
    except DownloadError as e:
        print(f"[ERROR] 冒烟 extract_info 失败: {e}")
        print("  请检查 deno / cookies / 网络后重试。")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 冒烟异常: {e}")
        sys.exit(1)

    if not isinstance(info, dict):
        print("[ERROR] 冒烟无 info")
        sys.exit(1)

    formats = info.get("formats") or []
    n_all = len(formats) if isinstance(formats, list) else 0
    height = extract_max_height(info)
    print(f"  n_formats={n_all}  max_height={height}")
    if height is None:
        status = classify_missing_height(info) if isinstance(info, dict) else STATUS_EMPTY_FORMATS
        print(
            f"[ERROR] 冒烟无有效视频清晰度（status≈{status}）。"
            "常见原因：缺 deno、cookies 失效、限流。\n"
            "  手测: yt-dlp -F --cookies <file> --js-runtimes deno "
            f"{YOUTUBE_URL_PREFIX}{video_id}"
        )
        sys.exit(1)
    print(f"  冒烟通过 (max_height={height})")


def resolve_output_path(input_path: str, output_arg: str | None) -> tuple[str, str]:
    """解析输出路径，返回 (ckpt_path, final_path)。

    ckpt_path: 增量进度 sidecar（仅 video_id + 结果列）
    final_path: 完成后的完整输出
    """
    if output_arg:
        final = output_arg
    else:
        stem, ext = os.path.splitext(input_path)
        if ext.lower() in (".parquet", ".pq"):
            final = f"{stem}_resolution.csv"
        else:
            final = f"{stem}_resolution{ext or '.csv'}"

    ckpt = final + ".ckpt.csv"
    return ckpt, final


def _read_table(path: str) -> pd.DataFrame:
    """按扩展名读取 CSV 或 Parquet。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path).astype(str).fillna("")
    return pd.read_csv(path, dtype=str).fillna("")


def load_input_df(input_path: str, ckpt_path: str, final_path: str) -> pd.DataFrame:
    """加载数据：以 input 为基底，合并 sidecar / 旧 .part / 已有输出上的进度列。

    优先 .ckpt.csv，其次旧 .part，再次 -o 最终文件。
    按 video_id 合并，input 新增行会保留为待处理。
    """
    if not os.path.exists(input_path):
        print(f"[ERROR] 文件不存在: {input_path}")
        sys.exit(1)

    base = _read_table(input_path)
    input_mtime = os.path.getmtime(input_path)
    final_mtime = os.path.getmtime(final_path) if os.path.exists(final_path) else 0.0
    same_as_input = os.path.abspath(final_path) == os.path.abspath(input_path)
    legacy_part = final_path + ".part"

    progress: pd.DataFrame | None = None
    progress_label = ""
    candidates: list[tuple[str, str]] = []
    if os.path.exists(ckpt_path):
        candidates.append((ckpt_path, f"断点 {ckpt_path}"))
    if os.path.exists(legacy_part):
        candidates.append((legacy_part, f"旧断点 {legacy_part}"))
    if os.path.exists(final_path) and not same_as_input:
        candidates.append((final_path, f"已有输出 {final_path}"))

    best_mtime = -1.0
    for path, label in candidates:
        mtime = os.path.getmtime(path)
        if mtime >= input_mtime and mtime >= best_mtime:
            # final 还要 >= final 自身比较已含；对 part/ckpt 要求不早于 final
            if path == final_path or mtime >= final_mtime:
                progress = _read_table(path)
                progress_label = label
                best_mtime = mtime

    if progress is None:
        return base

    for col in ("max_height", "fetch_status"):
        if col not in progress.columns:
            progress[col] = ""

    if "video_id" not in base.columns or "video_id" not in progress.columns:
        print(f"[续跑] {progress_label}（无 video_id，按原表使用进度文件）")
        return progress

    print(f"[续跑] 合并进度自 {progress_label}")
    prog = (
        progress[["video_id", "max_height", "fetch_status"]]
        .astype(str)
        .fillna("")
        .drop_duplicates(subset=["video_id"], keep="last")
    )
    base = base.drop(columns=[c for c in ("max_height", "fetch_status") if c in base.columns])
    merged = base.merge(prog, on="video_id", how="left")
    merged["max_height"] = merged["max_height"].fillna("")
    merged["fetch_status"] = merged["fetch_status"].fillna("")
    return merged


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """确保 max_height 和 fetch_status 列存在；迁移历史假 ok。"""
    for col in ("max_height", "fetch_status"):
        if col not in df.columns:
            df[col] = ""
    status = df["fetch_status"].fillna("").astype(str).str.strip()
    height = df["max_height"].fillna("").astype(str).str.strip()
    # 历史：ok 但无 height → 真音频终态 no_height
    legacy = (status == STATUS_OK) & (height == "")
    if legacy.any():
        df.loc[legacy, "fetch_status"] = STATUS_NO_HEIGHT
    # 历史：ok 但 height < MIN_VIDEO_HEIGHT（storyboard 90）→ 可重试 empty_formats
    def _h_int(s: str) -> int | None:
        if not s:
            return None
        try:
            return int(float(s))
        except (TypeError, ValueError):
            return None

    status = df["fetch_status"].fillna("").astype(str).str.strip()
    height = df["max_height"].fillna("").astype(str).str.strip()
    fake_heights = height.map(_h_int)
    fake_ok = (status == STATUS_OK) & fake_heights.notna() & (fake_heights < MIN_VIDEO_HEIGHT)
    if fake_ok.any():
        n = int(fake_ok.sum())
        print(f"[迁移] {n} 条假 ok(height<{MIN_VIDEO_HEIGHT}) → {STATUS_EMPTY_FORMATS}")
        df.loc[fake_ok, "fetch_status"] = STATUS_EMPTY_FORMATS
        df.loc[fake_ok, "max_height"] = ""
    return df


def get_pending_mask(df: pd.DataFrame, args: argparse.Namespace) -> pd.Index:
    """计算待处理行的索引列表。

    逻辑：
    - --overwrite：所有行
    - 默认：非终态（含 empty_formats）且非 error
    - --retry-errors：额外包含 fetch_status == "error"
    - --retry-no-height：额外包含 fetch_status == "no_height"
    """
    if args.overwrite:
        df["max_height"] = ""
        df["fetch_status"] = ""
        return df.index

    status = df["fetch_status"].fillna("").astype(str).str.strip()
    mask = ~status.isin(TERMINAL_STATUSES)
    if not args.retry_errors:
        mask = mask & (status != STATUS_ERROR)
    if getattr(args, "retry_no_height", False):
        mask = mask | (status == STATUS_NO_HEIGHT)
    return df[mask].index


def print_summary(stats: FetchStats, total_processed: int, elapsed: float) -> None:
    """打印处理摘要。"""
    snap = stats.snapshot()
    print(f"\n{'─' * 50}")
    if total_processed:
        print(f"处理完成: {total_processed} 条 | 耗时: {elapsed:.0f}s ({elapsed/total_processed:.1f}s/条)")
    print(f"  成功 (ok):              {snap['ok']}")
    print(f"  空 formats (empty):     {snap['empty_formats']}")
    print(f"  无分辨率 (no_height):   {snap['no_height']}")
    print(f"  不可用 (unavailable):   {snap['unavailable']}")
    print(f"  私有 (private):         {snap['private']}")
    print(f"  已删除 (deleted):       {snap['deleted']}")
    print(f"  错误 (error):           {snap['error']}")
    print(f"{'─' * 50}")


# ─────────────────────────────────────────────
# Bank 模式
# ─────────────────────────────────────────────

DEFAULT_BANK_SIZE = 10_000
DEFAULT_BANK_JOBS = 1


def bank_path(out_dir: str, index: int) -> str:
    return os.path.join(out_dir, f"bank_{index:03d}.csv")


def list_bank_files(out_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(out_dir, "bank_*.csv")))
    return [f for f in files if not f.endswith(".part")]


def prepare_bank_files(
    input_path: str,
    out_dir: str,
    bank_size: int,
    max_banks: int | None,
) -> list[str]:
    """按行切分 input → out_dir/bank_XXX.csv；已存在则跳过（保留续跑进度）。"""
    os.makedirs(out_dir, exist_ok=True)
    df = _read_table(input_path)
    if "video_id" not in df.columns:
        print(f"[ERROR] 未找到 video_id 列。可用列: {', '.join(df.columns)}")
        sys.exit(1)
    n = len(df)
    n_banks = max(1, math.ceil(n / bank_size))
    if max_banks is not None and max_banks > 0:
        n_banks = min(n_banks, max_banks)
    paths: list[str] = []
    for i in range(n_banks):
        path = bank_path(out_dir, i)
        paths.append(path)
        part = path + ".part"
        ckpt = path + ".ckpt.csv"
        if os.path.exists(path) or os.path.exists(part) or os.path.exists(ckpt):
            continue
        chunk = df.iloc[i * bank_size : (i + 1) * bank_size].copy()
        chunk.to_csv(path, index=False)
        print(f"  创建 {os.path.basename(path)}  rows={len(chunk)}")
    print(f"Bank 计划: {n_banks} 个 × 最多 {bank_size} 行（源表 {n} 行）")
    return paths


def bank_pending_report(path: str, args: argparse.Namespace) -> tuple[bool, str]:
    """返回 (需要跑?, 说明)。"""
    working, final = resolve_output_path(path, path)
    if not os.path.exists(working) and not os.path.exists(final) and not os.path.exists(path):
        return True, "文件不存在，需创建/抓取"
    df = load_input_df(path, working, final)
    df = ensure_columns(df)
    status = df["fetch_status"].fillna("").astype(str).str.strip()
    n_error = int((status == STATUS_ERROR).sum())
    n_empty_fmt = int((status == STATUS_EMPTY_FORMATS).sum())
    n_empty = int((status == "").sum())
    n_ok = int((status == STATUS_OK).sum())
    n_term = int(status.isin(TERMINAL_STATUSES).sum())

    n_nh = int((status == STATUS_NO_HEIGHT).sum())
    probe = argparse.Namespace(
        overwrite=False,
        retry_errors=bool(getattr(args, "retry_errors", False)),
        retry_no_height=bool(getattr(args, "retry_no_height", False)),
    )
    pending = get_pending_mask(df, probe)
    if len(pending) > 0:
        return True, f"待处理 {len(pending)} 条"

    # pending=0：区分真完成 vs 全是 error / no_height（默认不重试）
    if n_error and not getattr(args, "retry_errors", False):
        return False, (
            f"无待处理（error={n_error}, ok={n_ok}, 终态={n_term}；"
            f"error 默认不重跑，加 --retry-errors 或 --overwrite）"
        )
    if n_nh and not getattr(args, "retry_no_height", False):
        return False, (
            f"无待处理（no_height={n_nh}, ok={n_ok}；"
            f"加 --retry-no-height 可重提）"
        )
    return False, (
        f"已完成（ok={n_ok}, 终态={n_term}, empty_status={n_empty}, "
        f"empty_formats={n_empty_fmt}）"
    )


def bank_has_pending(path: str, args: argparse.Namespace) -> bool:
    """该 bank 是否还有待抓取行。"""
    need, _ = bank_pending_report(path, args)
    return need


def merge_bank_dir(out_dir: str, merged_name: str = "merged_resolution.csv") -> str:
    """按 bank_XXX 顺序汇总为一个 CSV。"""
    files = list_bank_files(out_dir)
    if not files:
        print(f"[ERROR] 目录中无 bank_*.csv: {out_dir}")
        sys.exit(1)
    frames = []
    for f in files:
        frames.append(_read_table(f))
        print(f"  合并 {os.path.basename(f)}  rows={len(frames[-1])}")
    merged = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(out_dir, merged_name)
    tmp = out_path + ".tmp"
    merged.to_csv(tmp, index=False)
    os.replace(tmp, out_path)
    print(f"汇总完成: {out_path}  rows={len(merged)}")
    return out_path


def _build_bank_cmd(args: argparse.Namespace, bank_csv: str, cookies_file: str | None) -> list[str]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        bank_csv,
        "-o",
        bank_csv,
        "-w",
        str(args.workers),
        "--sleep-interval",
        str(args.sleep_interval),
        "--no-smoke",  # 编排层已冒烟；子进程跳过
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.retry_errors:
        cmd.append("--retry-errors")
    if getattr(args, "retry_no_height", False):
        cmd.append("--retry-no-height")
    if args.no_cookies:
        cmd.append("--no-cookies")
    elif cookies_file:
        cmd.extend(["--cookies", cookies_file])
    elif args.cookies:
        cmd.extend(["--cookies", args.cookies])
    elif args.cookies_from_browser:
        cmd.extend(["--cookies-from-browser", args.cookies_from_browser])
    return cmd


def _run_one_bank_subprocess(cmd: list[str], label: str) -> int:
    print(f"\n═══ {label} 开始 ═══", flush=True)
    print(" ", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(cmd, env=env)
    print(f"═══ {label} 结束 exit={proc.returncode} ═══", flush=True)
    return int(proc.returncode)


def ensure_shared_cookies(
    args: argparse.Namespace,
    out_dir: str,
    *,
    force_refresh: bool = False,
) -> str | None:
    """编排层预取/刷新 cookies，供各 bank 子进程 --cookies 复用。

    force_refresh=True 时对浏览器缓存路径强制重新 extract（忽略已有文件）。
    若用户显式 --cookies / 环境变量文件，则原样返回，不刷新。
    """
    if args.no_cookies:
        return None
    if args.cookies:
        return args.cookies
    env_cookies = (os.getenv("YT_DLP_COOKIES_FILE", "") or "").strip()
    if env_cookies:
        return env_cookies
    cache_path = os.path.join(out_dir, ".cookies_cache.txt")
    browser_spec = (
        (args.cookies_from_browser or "").strip()
        or os.getenv("YT_DLP_COOKIES_FROM_BROWSER", "").strip()
        or "chrome"
    )
    browser, profile = parse_browser_spec(browser_spec)
    mgr = CookieManager(cache_path, browser=browser, profile=profile)
    action = "刷新" if force_refresh and os.path.exists(cache_path) else "预取"
    print(f"{action} cookies ({browser}{(':' + profile) if profile else ''}) → {cache_path}")
    if force_refresh and os.path.exists(cache_path):
        try:
            os.unlink(cache_path)
        except OSError:
            pass
    if mgr.extract_now():
        if not CookieManager.cookie_file_has_youtube_login(cache_path):
            print("  [WARN] cookies 中未见 YouTube 登录标记")
        return cache_path
    print("  [WARN] cookies 预取失败，各 bank 将自行尝试 / 匿名")
    return None


def run_bank_orchestrator(args: argparse.Namespace) -> None:
    """自动切 bank → 按 jobs 跑完 → 汇总。无需手动接下一个。"""
    out_dir = args.output
    if not out_dir:
        stem, _ = os.path.splitext(os.path.abspath(args.input))
        out_dir = stem + "_banks"
    out_dir = os.path.abspath(out_dir)
    if out_dir.lower().endswith(".csv"):
        out_dir = out_dir[: -len(".csv")] + "_banks"

    bank_size = args.bank_size
    bank_jobs = max(1, int(args.bank_jobs))
    refresh_every = max(0, int(getattr(args, "cookie_refresh_banks", DEFAULT_COOKIE_REFRESH_BANKS)))
    print(f"Bank 模式: size={bank_size}  jobs={bank_jobs}  workers/bank={args.workers}")
    print(f"输出目录: {out_dir}")
    print(f"cookies 刷新: 每 {refresh_every} 个 bank（0=仅启动时）")
    if bank_jobs >= 2 and args.workers >= 2:
        print("[WARN] bank-jobs≥2 且 -w≥2 总并发偏高，易 429；建议 jobs=1 -w=2 或 jobs=2 -w=1")

    # 编排层预检：有 cookies 意图时强制 JS runtime + 可选冒烟
    using_cookies = not args.no_cookies
    js_runtimes = preflight_js_runtime(using_cookies=using_cookies)

    paths = prepare_bank_files(args.input, out_dir, bank_size, args.max_banks)
    cookies_file = ensure_shared_cookies(args, out_dir, force_refresh=False)

    if using_cookies and not getattr(args, "no_smoke", False):
        smoke_opts = build_ydl_opts(args.sleep_interval, cookies_file, js_runtimes)
        smoke_extract_formats(smoke_opts)
    elif abs(args.sleep_interval - DEFAULT_SLEEP_INTERVAL) < 1e-9:
        print(
            f"[建议] 稳定长跑可将 --sleep-interval 调到 1.0–2.0"
            f"（当前默认 {DEFAULT_SLEEP_INTERVAL}）"
        )

    banks_since_refresh = 0

    def cookies_before_bank() -> str | None:
        nonlocal cookies_file, banks_since_refresh
        # refresh_every=1 → 每个 bank 启动前刷新；0 → 仅用启动预取
        if refresh_every > 0 and banks_since_refresh % refresh_every == 0:
            cookies_file = (
                ensure_shared_cookies(args, out_dir, force_refresh=True) or cookies_file
            )
        banks_since_refresh += 1
        return cookies_file

    todo: list[tuple[int, str]] = []
    for i, p in enumerate(paths):
        if args.overwrite:
            todo.append((i, p))
            continue
        need, reason = bank_pending_report(p, args)
        if need:
            todo.append((i, p))
        else:
            print(f"  跳过 bank_{i:03d}：{reason}", flush=True)

    if not todo:
        print("所有 bank 无需抓取，直接汇总。", flush=True)
        merge_bank_dir(out_dir)
        return

    print(f"待跑 bank: {len(todo)} / {len(paths)}")
    codes: list[int] = []

    if bank_jobs == 1:
        for i, p in todo:
            os.makedirs(out_dir, exist_ok=True)
            if not os.path.exists(p):
                print(f"[ERROR] bank 文件丢失: {p}（目录可能被删；请重新 --bank-size 切分或恢复备份）", flush=True)
                codes.append(1)
                continue
            ck = cookies_before_bank()
            rc = _run_one_bank_subprocess(
                _build_bank_cmd(args, p, ck),
                f"bank_{i:03d}",
            )
            codes.append(rc)
            if rc == 130:
                print("[中断] 停止后续 bank；已完成的可续跑。")
                break
            if rc != 0:
                print(f"[WARN] bank_{i:03d} exit={rc}；检查 deno/cookies/熔断后可续跑。")
    else:
        pending: dict = {}
        with ThreadPoolExecutor(max_workers=bank_jobs) as pool:
            it = iter(todo)

            def submit_next(item: tuple[int, str]):
                i, p = item
                ck = cookies_before_bank()
                fut = pool.submit(
                    _run_one_bank_subprocess,
                    _build_bank_cmd(args, p, ck),
                    f"bank_{i:03d}",
                )
                pending[fut] = (i, p)

            for _ in range(min(bank_jobs, len(todo))):
                submit_next(next(it))
            while pending:
                done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    i, p = pending.pop(fut)
                    rc = fut.result()
                    codes.append(rc)
                    if rc == 130:
                        print("[中断] 取消排队中的 bank")
                        pending.clear()
                        break
                    try:
                        submit_next(next(it))
                    except StopIteration:
                        continue

    if any(c == 130 for c in codes):
        sys.exit(130)
    if any(c != 0 for c in codes):
        print(f"[WARN] 部分 bank 非零退出: {codes}，仍尝试汇总已完成文件")
    merge_bank_dir(out_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 YouTube 批量获取视频最高分辨率高度 (max_height)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "运维提示:\n"
            "  • 有 cookies 时须安装 deno（推荐）或 node，否则 formats 会空/只剩 storyboard。\n"
            "  • 稳定 cookies：隐私窗登录 YouTube → 打开 robots.txt → 导出 → 关掉该窗。\n"
            "  • 长跑建议 --sleep-interval 1.0–2.0；bank 推荐 --bank-jobs 1 -w 2。\n"
            "  • empty_formats/error 默认可续跑；历史假 ok(90) 会迁成 empty_formats。\n"
            "  • 真音频 no_height 补跑: --retry-no-height；error 补跑: --retry-errors。\n"
        ),
    )
    parser.add_argument("input", nargs="?", default=None,
                        help="输入 CSV/Parquet（--merge-only 时可省略）")
    parser.add_argument("-w", "--workers", type=int, default=2,
                        help="每个进程内线程数 (默认: 2；bank 推荐 2)")
    parser.add_argument("-n", "--limit", type=int, default=None,
                        help="最多处理 N 条（单文件模式测试用）")
    parser.add_argument("--overwrite", action="store_true",
                        help="清空已有 max_height/fetch_status，全量重跑")
    parser.add_argument("--retry-errors", action="store_true",
                        help="重试 fetch_status=error 的行")
    parser.add_argument("--retry-no-height", action="store_true",
                        help="重试 fetch_status=no_height（真音频等；空 formats 已默认可续）")
    parser.add_argument("--no-cookies", action="store_true",
                        help="不使用 cookies（匿名访问，风控更严）")
    parser.add_argument("--cookies", type=str, default=None,
                        help="Netscape cookies 文件")
    parser.add_argument("--cookies-from-browser", type=str, default=None,
                        help="浏览器规格，如 chrome 或 chrome:Default")
    parser.add_argument(
        "--cookie-refresh-banks",
        type=int,
        default=DEFAULT_COOKIE_REFRESH_BANKS,
        help=(
            f"bank 编排每 N 个 bank 刷新 cookies（默认 {DEFAULT_COOKIE_REFRESH_BANKS}；"
            "0=仅启动预取）"
        ),
    )
    parser.add_argument("--no-smoke", action="store_true",
                        help="跳过启动冒烟抽流（不推荐）")
    parser.add_argument("--sleep-interval", type=float, default=DEFAULT_SLEEP_INTERVAL,
                        help=(
                            f"yt-dlp 请求间隔秒数 (默认: {DEFAULT_SLEEP_INTERVAL}；"
                            "稳定长跑建议 1.0–2.0)"
                        ))
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="输出 CSV；bank 模式下为输出目录")
    parser.add_argument("--bank-size", type=int, default=None,
                        help=f"启用 bank 模式，每 bank 行数 (常用 {DEFAULT_BANK_SIZE})")
    parser.add_argument("--bank-jobs", type=int, default=DEFAULT_BANK_JOBS,
                        help="同时跑几个 bank 进程 (默认 1；最多建议 2)")
    parser.add_argument("--max-banks", type=int, default=None,
                        help="只跑前 N 个 bank（测试用）")
    parser.add_argument("--merge-only", action="store_true",
                        help="只汇总 -o 目录下 bank_*.csv → merged_resolution.csv")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.merge_only:
        if not args.output:
            print("[ERROR] --merge-only 需要 -o <bank目录>")
            sys.exit(1)
        merge_bank_dir(args.output)
        return

    if args.input is None:
        print("[ERROR] 需要 input 路径（或使用 --merge-only -o <dir>）")
        sys.exit(1)

    if args.bank_size is not None:
        if args.bank_size <= 0:
            print("[ERROR] --bank-size 必须 > 0")
            sys.exit(1)
        run_bank_orchestrator(args)
        return

    run_fetch_job(args)


# ─────────────────────────────────────────────
# CLI — 单文件抓取
# ─────────────────────────────────────────────

def run_fetch_job(args: argparse.Namespace) -> None:

    # ── 解析路径 ─────────────────────────────
    ckpt_path, final_path = resolve_output_path(args.input, args.output)
    print(f"输入: {args.input}")
    print(f"输出: {final_path}")
    print(f"进度 sidecar: {ckpt_path}")

    # ── 读取表 ───────────────────────────────
    df = load_input_df(args.input, ckpt_path, final_path)
    if "video_id" not in df.columns:
        print(f"[ERROR] 未找到 video_id 列。可用列: {', '.join(df.columns)}")
        sys.exit(1)

    df = ensure_columns(df)

    # ── 确定待处理行 ─────────────────────────
    pending_idx = get_pending_mask(df, args)
    if args.limit is not None and args.limit > 0:
        pending_idx = pending_idx[:args.limit]

    total_pending = len(pending_idx)
    if total_pending == 0:
        print("没有待处理的行，退出。")
        write_final_csv(df, final_path, ckpt_path)
            print(f"已输出到: {final_path}")
        out_dir = os.path.dirname(os.path.abspath(final_path)) or "."
        mark_done(
            out_dir, "resolution",
            input=args.input, output=final_path,
            done=0, total=0, pending=0,
        )
        write_run_log(
            "resolution", args.input, out_dir,
            stats={
                "pending": 0,
                "rows": len(df),
                "output_path": final_path,
                "note": "no_pending",
            },
            command=f"fetch_resolution.py {args.input} -o {final_path}",
        )
        return

    print(f"待处理: {total_pending} 条 / 合计: {len(df)} 条")
    print(f"并发: {args.workers} workers / 请求间隔: {args.sleep_interval}s")

    # ── Cookies 初始化 ────────────────────────
    cookie_mgr: CookieManager | None = None
    if not args.no_cookies:
        cookies_file = (args.cookies or os.getenv("YT_DLP_COOKIES_FILE", "")).strip() or None
        browser_spec = (
            (args.cookies_from_browser or "").strip()
            or os.getenv("YT_DLP_COOKIES_FROM_BROWSER", "").strip()
            or "chrome"
        )
        # 缓存放到输出目录旁，避免写到只读/奇怪的 input 目录
        out_dir = os.path.dirname(os.path.abspath(final_path)) or "."
        cache_path = os.path.join(out_dir, ".cookies_cache.txt")

        if cookies_file:
            cookie_mgr = CookieManager(cookies_file, refreshable=False)
            print(f"使用 cookies 文件: {cookies_file}")
            if not cookie_mgr.bind_existing_file():
                cookie_mgr = None
        else:
            browser, profile = parse_browser_spec(browser_spec)
            cookie_mgr = CookieManager(
                cache_path, browser=browser, profile=profile,
                refresh_sec=COOKIE_REFRESH_SEC,
            )
            spec_show = f"{browser}:{profile}" if profile else browser
            print(f"从浏览器提取 cookies ({spec_show}) → {cache_path}")
        if cookie_mgr.extract_now():
                print(f"  cookies 已缓存 (每 {COOKIE_REFRESH_SEC // 60} 分钟自动刷新)")
        else:
                print("  提取失败，回退为匿名访问")
            cookie_mgr = None

        if cookie_mgr and not CookieManager.cookie_file_has_youtube_login(cookie_mgr.cache_path):
            print("  [WARN] cookies 中未见 YouTube 登录标记 (LOGIN_INFO/PSID)，风控风险高")
            print("         请确认浏览器已登录 youtube.com，或换 --cookies-from-browser chrome:配置名")
    else:
        print("cookies: 已禁用（匿名访问）")

    using_cookies = cookie_mgr is not None
    js_runtimes = preflight_js_runtime(using_cookies=using_cookies)

    cookie_file = cookie_mgr.cache_path if cookie_mgr else None
    ydl_opts = build_ydl_opts(args.sleep_interval, cookie_file, js_runtimes)
    if cookie_file:
        print(f"yt-dlp cookiefile: {cookie_file}")
    if using_cookies and not getattr(args, "no_smoke", False):
        smoke_extract_formats(ydl_opts)
    elif abs(args.sleep_interval - DEFAULT_SLEEP_INTERVAL) < 1e-9:
        print(
            f"[建议] 稳定长跑可将 --sleep-interval 调到 1.0–2.0"
            f"（当前默认 {DEFAULT_SLEEP_INTERVAL}）"
        )

    gate = AdaptiveConcurrencyGate(args.workers)
    stats = FetchStats()
    breaker = CircuitBreaker()
    circuit_tripped = False

    # 结果缓存：idx -> FetchResult
    results: dict[int, FetchResult] = {}
    results_lock = threading.Lock()

    # ── 进度条 ───────────────────────────────
    try:
        from tqdm import tqdm
        pbar = tqdm(
            total=total_pending,
            desc="获取分辨率",
            unit="条",
            miniters=1,
            mininterval=0.3,
            maxinterval=3.0,
            dynamic_ncols=True,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}] {postfix}"
            ),
        )
    except ImportError:
        pbar = None
        print("[提示] tqdm 未安装，将使用文本进度（pip install tqdm）")

    t_start = time.monotonic()
    last_flush_time = time.monotonic()
    last_cookie_refresh = time.monotonic()
    out_dir = os.path.dirname(os.path.abspath(final_path)) or "."
    prog = ThrottledProgress(
        out_dir, "resolution",
        interval_sec=5.0, every_n=50,
        input=args.input, output=final_path, total=total_pending,
    )
    prog.tick(force=True, done=0)

    def do_flush() -> None:
        """将结果缓存写入 CSV。"""
        nonlocal last_flush_time
        with results_lock:
            if not results:
                return
            for r in results.values():
                df.at[r.idx, "max_height"] = str(r.max_height) if r.max_height is not None else ""
                df.at[r.idx, "fetch_status"] = r.fetch_status
            results.clear()

        flush_checkpoint(df, ckpt_path)
        last_flush_time = time.monotonic()

    executor = ThreadPoolExecutor(max_workers=args.workers)
    future_to_idx: dict = {}
        completed = 0
    interrupted = False

    def handle_result(result: FetchResult) -> None:
        """写入缓存、更新统计与进度条。"""
        nonlocal completed, last_cookie_refresh, circuit_tripped
            stats.count(result.fetch_status)
            with results_lock:
                results[result.idx] = result
            completed += 1

        if breaker.record(result.fetch_status):
            circuit_tripped = True

            if pbar:
                snap = stats.snapshot()
                pbar.update(1)
                pbar.set_postfix({
                    "ok": snap["ok"],
                "ef": snap["empty_formats"],
                "nh": snap["no_height"],
                    "err": snap["error"],
                    "unav": snap["unavailable"],
                    "pri": snap["private"],
                    "del": snap["deleted"],
                }, refresh=True)
            elif completed % 10 == 0:
                snap = stats.snapshot()
            print(
                f"\r  进度: {completed}/{total_pending}  "
                f"ok={snap['ok']} ef={snap['empty_formats']} "
                f"nh={snap['no_height']} err={snap['error']}",
                end="",
            )

            if cookie_mgr:
                now = time.monotonic()
                if now - last_cookie_refresh >= COOKIE_REFRESH_SEC:
                    if cookie_mgr.maybe_refresh():
                        if pbar:
                            pbar.set_postfix({"cookie": "刷新"}, refresh=True)
                    last_cookie_refresh = now

            need_flush = len(results) >= FLUSH_BATCH_SIZE
            time_elapsed = time.monotonic() - last_flush_time
            if need_flush or (results and time_elapsed >= FLUSH_INTERVAL_SEC):
                do_flush()

        snap = stats.snapshot()
        prog.tick(done=completed, **snap)

    try:
        # 提交任务；空 video_id 直接记为 error，并计入进度
        for idx in pending_idx:
            if circuit_tripped:
                break
            video_id = str(df.at[idx, "video_id"]).strip()
            if not video_id or video_id.lower() == "nan":
                handle_result(FetchResult(
                    idx=idx, video_id=video_id,
                    max_height=None, fetch_status=STATUS_ERROR,
                ))
                continue
            fut = executor.submit(fetch_one, video_id, ydl_opts, gate, cookie_mgr, idx)
            future_to_idx[fut] = idx

        for fut in as_completed(future_to_idx):
            if circuit_tripped:
                # 仍消费已完成的 future，取消未完成的
                for f in future_to_idx:
                    f.cancel()
            try:
                if fut.cancelled():
                    continue
                result = fut.result()
            except Exception:
                idx = future_to_idx[fut]
                result = FetchResult(
                    idx=idx, video_id=str(df.at[idx, "video_id"]),
                    max_height=None, fetch_status=STATUS_ERROR,
                )
            handle_result(result)
            if circuit_tripped:
                break

        do_flush()

    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[中断] 已保存进度到: {ckpt_path}")
        print("  下次运行将自动从中断处继续。")
        if results:
            do_flush()
        for f in future_to_idx:
            f.cancel()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    if pbar:
        pbar.close()

    if interrupted:
        sys.exit(130)

    if circuit_tripped:
        do_flush()
        write_final_csv(df, final_path, ckpt_path)
        snap = stats.snapshot()
        print(
            "\n[熔断] empty_formats/error 占比过高或连续无有效 ok。\n"
            "  请检查: deno 是否在 PATH、cookies 是否有效、代理/限流、降低 -w。\n"
            "  修好后直接续跑（empty_formats 默认可重试）；必要时 --retry-errors。"
        )
        print_summary(stats, completed, time.monotonic() - t_start)
        sys.exit(2)

    # ── 收尾：写完整输出并清理 sidecar ─────────
    elapsed = time.monotonic() - t_start
    write_final_csv(df, final_path, ckpt_path)
        print(f"输出: {final_path}")

    print_summary(stats, total_pending, elapsed)
    snap = stats.snapshot()
    mark_done(
        out_dir, "resolution",
        input=args.input, output=final_path,
        done=completed, total=total_pending,
        elapsed_sec=round(elapsed, 1), **snap,
    )
    write_run_log(
        "resolution", args.input, out_dir,
        stats={
            **snap,
            "pending": total_pending,
            "elapsed_sec": round(elapsed, 1),
            "output_path": final_path,
            "ckpt_path": ckpt_path,
        },
        command=f"fetch_resolution.py {args.input} -o {final_path}",
    )


if __name__ == "__main__":
    main()

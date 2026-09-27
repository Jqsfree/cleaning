#!/usr/bin/env python3
"""
core/yt_dlp_auth.py — yt-dlp cookies / 浏览器认证共享工具

供 tools/fetch_resolution.py、qc/vision_storyboard.py 等复用。
优先级：--cookies / YT_DLP_COOKIES_FILE > --cookies-from-browser / 环境变量 > 默认 chrome。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_COOKIE_REFRESH_SEC = 20 * 60


def detect_js_runtimes() -> dict[str, dict] | None:
    """检测 yt-dlp 可用的 JS runtime（cookies 场景必需，见 yt-dlp EJS wiki）。"""
    if shutil.which("deno"):
        return {"deno": {}}
    if shutil.which("node"):
        return {"node": {}}
    return None


@dataclass(frozen=True)
class YtDlpAuth:
    """yt-dlp 认证：cookies 文件或浏览器二选一。"""
    cookies_file: str | None = None
    cookies_from_browser: tuple[str, ...] | None = None


def parse_browser_spec(spec: str) -> tuple[str, str | None]:
    """'chrome' 或 'chrome:Profile 1' → (browser, profile)。"""
    parts = [p.strip() for p in (spec or "").split(":") if p.strip()]
    if not parts:
        return "chrome", None
    return parts[0], (parts[1] if len(parts) > 1 else None)


def browser_spec_tuple(spec: str) -> tuple[str, ...]:
    """转为 yt-dlp cookiesfrombrowser 元组。"""
    browser, profile = parse_browser_spec(spec)
    return (browser, profile) if profile else (browser,)


def save_cookie_jar(jar: Any, cache_path: str) -> None:
    """原子写入 Netscape cookies，避免多线程读到半写入文件。"""
    d = os.path.dirname(os.path.abspath(cache_path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".cookies_", suffix=".tmp")
    os.close(fd)
    try:
        jar.save(tmp, ignore_discard=True, ignore_expires=True)
        os.replace(tmp, cache_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def cookie_file_has_youtube_login(cache_path: str) -> bool:
    """粗检：是否含 YouTube/Google 登录相关 cookie。"""
    try:
        text = open(cache_path, encoding="utf-8", errors="replace").read()
    except OSError:
        return False
    markers = ("LOGIN_INFO", "__Secure-1PSID", "__Secure-3PSID", "SAPISID")
    return any(m in text for m in markers)


def resolve_yt_dlp_auth(
    cookies_arg: str | None = None,
    cookies_from_browser_arg: str | None = None,
    *,
    default_browser: str = "chrome",
    exit_on_missing_file: bool = True,
) -> YtDlpAuth:
    """解析认证：文件 > CLI 浏览器 > 环境变量 > default_browser。"""
    cookies_path = (cookies_arg or os.getenv("YT_DLP_COOKIES_FILE", "") or "").strip()
    if cookies_path:
        if not os.path.exists(cookies_path):
            msg = f"[ERROR] cookies 文件不存在: {cookies_path}"
            if exit_on_missing_file:
                print(msg)
                sys.exit(1)
            raise FileNotFoundError(cookies_path)
        return YtDlpAuth(cookies_file=cookies_path)

    browser_spec = (
        (cookies_from_browser_arg or "").strip()
        or os.getenv("YT_DLP_COOKIES_FROM_BROWSER", "").strip()
        or default_browser
    )
    return YtDlpAuth(cookies_from_browser=browser_spec_tuple(browser_spec))


def apply_yt_dlp_auth(ydl_opts: dict, auth: YtDlpAuth) -> None:
    if auth.cookies_file:
        ydl_opts["cookiefile"] = auth.cookies_file
    elif auth.cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = auth.cookies_from_browser


def extract_browser_cookies_to_file(
    browser: str,
    profile: str | None,
    cache_path: str,
) -> None:
    from yt_dlp.cookies import extract_cookies_from_browser

    jar = extract_cookies_from_browser(browser, profile)
    save_cookie_jar(jar, cache_path)


def prefetch_browser_cookies(auth: YtDlpAuth, cache_path: str) -> YtDlpAuth:
    """从浏览器导出 cookies 到文件，供多线程安全复用。"""
    if not auth.cookies_from_browser:
        return auth
    browser = auth.cookies_from_browser[0]
    profile = auth.cookies_from_browser[1] if len(auth.cookies_from_browser) > 1 else None
    extract_browser_cookies_to_file(browser, profile, cache_path)
    return YtDlpAuth(cookies_file=cache_path)


class CookieManager:
    """从浏览器提取 cookies，定期刷新到缓存文件（线程安全）。"""

    def __init__(
        self,
        cache_path: str,
        browser: str = "chrome",
        profile: str | None = None,
        *,
        refreshable: bool = True,
        refresh_sec: float = DEFAULT_COOKIE_REFRESH_SEC,
        warn: Callable[..., None] = print,
    ):
        self._cache_path = cache_path
        self._browser = browser
        self._profile = profile
        self._refreshable = refreshable
        self._refresh_sec = refresh_sec
        self._warn = warn
        self._lock = threading.Lock()
        self._last_refresh = 0.0
        self._refresh_version = 0

    @property
    def cache_path(self) -> str:
        return self._cache_path

    @property
    def refresh_version(self) -> int:
        with self._lock:
            return self._refresh_version

    @staticmethod
    def cookie_file_has_youtube_login(cache_path: str) -> bool:
        return cookie_file_has_youtube_login(cache_path)

    def extract_now(self) -> bool:
        if not self._refreshable:
            return os.path.exists(self._cache_path)
        with self._lock:
            try:
                extract_browser_cookies_to_file(
                    self._browser, self._profile, self._cache_path,
                )
                self._last_refresh = time.monotonic()
                self._refresh_version += 1
                return True
            except Exception as e:
                self._warn(f"  [WARN] cookies 提取失败: {e}")
                return False

    def bind_existing_file(self) -> bool:
        if not os.path.exists(self._cache_path):
            self._warn(f"  [WARN] cookies 文件不存在: {self._cache_path}")
            return False
        with self._lock:
            self._last_refresh = time.monotonic()
            self._refresh_version += 1
        return True

    def maybe_refresh(self) -> bool:
        if not self._refreshable:
            return False
        with self._lock:
            if self._last_refresh > 0:
                if time.monotonic() - self._last_refresh < self._refresh_sec:
                    return False
        return self.extract_now()

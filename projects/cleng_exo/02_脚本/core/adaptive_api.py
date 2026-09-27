"""API / yt-dlp 并发自适应：遇 429 或瞬时错误尖峰时降并行，冷却后回升。

设计原则:
- 不做全局退避（不暂停所有线程），只通过调节 max 来限流
- 文本 QC 请求小（max_tokens=5），靠并发撑吞吐，全局退避会严重拖慢
- 视频 QC 请求重，少量并发就够了，降低上限自然限流
- 滑动窗口熔断：瞬时错误率过高时立刻把 max 压到 1，避免越跑越慢
"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Callable, Iterator, Literal


ErrorKind = Literal["transient", "permanent"]

# storyboard / API 常见瞬时错误（可 ERROR 重试）
_TRANSIENT_TOKENS = frozenset({
    "rate_limited",
    "rate_limit",
    "rate_limit_error",
    "bot_challenge",
    "empty_formats",
    "yt_dlp_error",
    "storyboard_not_found",
    "storyboard_download_failed",
    "api_connection_error",
    "api_error",
    "timeout",
    "429",
    "too many requests",
    "connection",
    "temporarily",
    "unavailable",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
})

# 明确不可靠重试的永久类（重试轮跳过）
_PERMANENT_TOKENS = frozenset({
    "invalid_response",
    "l0_",
    "channel_blacklist",
    "title:",
    "too_short",
    "too_long",
    "private",
    "deleted",
    "removed",
    "copyright",
    "age.restricted",
    "members.only",
    "premium",
    "geo",
    "not available",
    "video unavailable",
    "shutdown_requested",
})


def classify_error_kind(reason: str | None) -> ErrorKind:
    """将 QC error_reason 分为 transient（可重试）或 permanent（跳过重试）。"""
    if not reason:
        return "transient"
    s = str(reason).strip().lower()
    if not s:
        return "transient"
    for tok in _PERMANENT_TOKENS:
        if tok in s:
            return "permanent"
    for tok in _TRANSIENT_TOKENS:
        if tok in s:
            return "transient"
    # api_error:... / future_exception:... 默认当瞬时
    if s.startswith("api_error") or s.startswith("future_exception") or s.startswith("exception:"):
        return "transient"
    return "permanent"


def is_transient_error(reason: str | None) -> bool:
    return classify_error_kind(reason) == "transient"


class AdaptiveConcurrencyGate:
    """限制同时在飞的请求数；429 / 熔断时递减上限，冷却后自动回升。

    行为:
    - acquire() 阻塞直到有可用槽位（不做全局暂停）
    - on_rate_limit() 立即降低并发上限（最少降至 1）
    - record_outcome(ok/transient) 滑动窗口熔断：错误率超阈值 → max=1
    - 连续 RECOVERY_INTERVAL 秒未触发限流后，每轮 +1 逐步恢复到 initial
    """

    RECOVERY_INTERVAL = 2.0   # 无 429 多少秒后每步 +1 恢复

    def __init__(
        self,
        initial: int,
        *,
        window_size: int = 40,
        error_rate_threshold: float = 0.45,
        label: str = "API",
    ):
        self._initial = max(1, initial)
        self._max = self._initial
        self._in_flight = 0
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._last_rate_limit = 0.0
        self._recovery_step_at = 0.0
        self._label = label
        self._window_size = max(5, window_size)
        self._error_rate_threshold = min(1.0, max(0.05, error_rate_threshold))
        # True = transient failure, False = success（永久错误不计入熔断窗口）
        self._outcomes: deque[bool] = deque(maxlen=self._window_size)
        self._tripped = False

    @property
    def max_concurrent(self) -> int:
        with self._lock:
            return self._max

    @property
    def initial(self) -> int:
        return self._initial

    @property
    def is_tripped(self) -> bool:
        with self._lock:
            return self._tripped

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def acquire(self) -> None:
        with self._cv:
            while True:
                now = time.monotonic()
                self._maybe_recover(now)

                if self._in_flight < self._max:
                    self._in_flight += 1
                    return

                self._cv.wait(timeout=1.0)

    def release(self) -> None:
        with self._cv:
            self._in_flight = max(0, self._in_flight - 1)
            self._cv.notify_all()

    @contextmanager
    def slot(self) -> Iterator[None]:
        """acquire/release 上下文，可替代 Semaphore。"""
        self.acquire()
        try:
            yield
        finally:
            self.release()

    def on_rate_limit(self, log_fn: Callable[[str], None] | None = None) -> None:
        """收到 429 时调用：仅降低并发上限，不做全局暂停。"""
        with self._cv:
            self._last_rate_limit = time.monotonic()
            self._recovery_step_at = 0.0
            if self._max > 1:
                self._max -= 1
                if log_fn:
                    log_fn(f"{self._label} 429: 并发上限降至 {self._max}")
            else:
                self._tripped = True
                if log_fn:
                    log_fn(f"{self._label} 429: 并发已为 1（熔断态）")
            self._cv.notify_all()

    def record_outcome(
        self,
        *,
        ok: bool = False,
        transient_error: bool = False,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        """记录成功或瞬时失败，供滑动窗口熔断。

        永久错误请勿调用（或 ok=False 且 transient_error=False 会被忽略）。
        """
        if not ok and not transient_error:
            return
        with self._cv:
            self._outcomes.append(bool(transient_error) if not ok else False)
            if ok and self._tripped:
                # 成功有助于恢复；实际回升仍走 _maybe_recover
                pass
            if not ok and transient_error:
                self._maybe_trip_locked(log_fn)
            self._cv.notify_all()

    def trip(self, log_fn: Callable[[str], None] | None = None) -> None:
        """强制熔断：max=1。"""
        with self._cv:
            self._max = 1
            self._tripped = True
            self._last_rate_limit = time.monotonic()
            self._recovery_step_at = 0.0
            if log_fn:
                log_fn(f"{self._label} 熔断: 并发上限强制为 1")
            self._cv.notify_all()

    def _maybe_trip_locked(self, log_fn: Callable[[str], None] | None) -> None:
        n = len(self._outcomes)
        if n < max(8, self._window_size // 2):
            return
        err_n = sum(1 for x in self._outcomes if x)
        rate = err_n / n
        if rate >= self._error_rate_threshold and self._max > 1:
            self._max = 1
            self._tripped = True
            self._last_rate_limit = time.monotonic()
            self._recovery_step_at = 0.0
            if log_fn:
                log_fn(
                    f"{self._label} 熔断: 近 {n} 条瞬时错误率 {rate:.0%} "
                    f">= {self._error_rate_threshold:.0%}，并发降至 1"
                )

    def _maybe_recover(self, now: float) -> None:
        """如果距上次限流已超过 RECOVERY_INTERVAL 秒，逐步恢复并发上限。"""
        if self._max >= self._initial:
            self._tripped = False
            self._recovery_step_at = 0.0
            return
        if self._last_rate_limit <= 0:
            return
        # 首次可恢复时刻 = 上次限流 + interval；之后每 interval +1
        if self._recovery_step_at == 0.0:
            self._recovery_step_at = self._last_rate_limit + self.RECOVERY_INTERVAL
        while self._max < self._initial and now >= self._recovery_step_at:
            self._max += 1
            self._recovery_step_at += self.RECOVERY_INTERVAL
        if self._max >= self._initial:
            self._tripped = False
            self._recovery_step_at = 0.0


class DualResourceScheduler:
    """YouTube meta + DashScope API 双资源独立限流。"""

    def __init__(
        self,
        yt_initial: int = 2,
        api_initial: int = 2,
        *,
        window_size: int = 40,
        error_rate_threshold: float = 0.45,
    ):
        self.yt_meta = AdaptiveConcurrencyGate(
            yt_initial,
            window_size=window_size,
            error_rate_threshold=error_rate_threshold,
            label="yt_meta",
        )
        self.api = AdaptiveConcurrencyGate(
            api_initial,
            window_size=window_size,
            error_rate_threshold=error_rate_threshold,
            label="API",
        )

    def yt_ready_for_retry(self) -> bool:
        """熔断未解除且仍为 1 时，可选择跳过 ERROR 重试轮。"""
        return not (self.yt_meta.is_tripped and self.yt_meta.max_concurrent <= 1)

    def api_ready_for_retry(self) -> bool:
        return not (self.api.is_tripped and self.api.max_concurrent <= 1)

"""AdaptiveConcurrencyGate / DualResourceScheduler / 错误分类单元测试。"""

import sys
import threading
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR))

from core.adaptive_api import (  # noqa: E402
    AdaptiveConcurrencyGate,
    DualResourceScheduler,
    classify_error_kind,
    is_transient_error,
)


class TestAdaptiveConcurrencyGate:
    def test_rate_limit_lowers_max(self):
        gate = AdaptiveConcurrencyGate(4)
        gate.on_rate_limit()
        assert gate.max_concurrent == 3

    def test_acquire_blocks_at_max(self):
        gate = AdaptiveConcurrencyGate(1)
        gate.acquire()
        done = threading.Event()

        def waiter():
            gate.acquire()
            done.set()
            gate.release()

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        assert not done.is_set()
        gate.release()
        t.join(timeout=2)
        assert done.is_set()

    def test_slot_context_manager(self):
        gate = AdaptiveConcurrencyGate(1)
        with gate.slot():
            assert gate.in_flight == 1
        assert gate.in_flight == 0

    def test_circuit_breaker_trips_on_error_rate(self):
        gate = AdaptiveConcurrencyGate(
            4, window_size=10, error_rate_threshold=0.5, label="test",
        )
        for _ in range(8):
            gate.record_outcome(transient_error=True)
        assert gate.max_concurrent == 1
        assert gate.is_tripped

    def test_permanent_outcome_ignored(self):
        gate = AdaptiveConcurrencyGate(4, window_size=10, error_rate_threshold=0.5)
        for _ in range(10):
            gate.record_outcome(ok=False, transient_error=False)
        assert gate.max_concurrent == 4
        assert not gate.is_tripped

    def test_recovery_after_cool_down(self):
        gate = AdaptiveConcurrencyGate(3)
        gate.RECOVERY_INTERVAL = 0.05
        gate.on_rate_limit()
        gate.on_rate_limit()
        assert gate.max_concurrent == 1
        time.sleep(0.12)
        gate.acquire()
        gate.release()
        assert gate.max_concurrent >= 2


class TestClassifyError:
    def test_transient_codes(self):
        assert classify_error_kind("rate_limited") == "transient"
        assert classify_error_kind("bot_challenge") == "transient"
        assert classify_error_kind("empty_formats") == "transient"
        assert classify_error_kind("api_error:TimeoutError:x") == "transient"
        assert is_transient_error("429 too many requests")

    def test_permanent_codes(self):
        assert classify_error_kind("invalid_response:XYZ") == "permanent"
        assert classify_error_kind("l0_anime") == "permanent"
        assert classify_error_kind("Video unavailable") == "permanent"
        assert not is_transient_error("invalid_response:foo")


class TestDualResourceScheduler:
    def test_independent_gates(self):
        sched = DualResourceScheduler(yt_initial=2, api_initial=4)
        assert sched.yt_meta.max_concurrent == 2
        assert sched.api.max_concurrent == 4
        sched.api.on_rate_limit()
        assert sched.api.max_concurrent == 3
        assert sched.yt_meta.max_concurrent == 2

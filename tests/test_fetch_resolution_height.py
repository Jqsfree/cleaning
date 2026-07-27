"""fetch_resolution.extract_max_height / classify_missing_height / CircuitBreaker 单测。"""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR / "tools"))

from fetch_resolution import (  # noqa: E402
    STATUS_EMPTY_FORMATS,
    STATUS_ERROR,
    STATUS_NO_HEIGHT,
    STATUS_OK,
    CircuitBreaker,
    classify_missing_height,
    extract_max_height,
)


class TestExtractMaxHeight:
    def test_format_height(self):
        info = {"formats": [{"height": 720}, {"height": 1080}, {"height": None}]}
        assert extract_max_height(info) == 1080

    def test_resolution_string(self):
        info = {"formats": [{"resolution": "1920x1080"}, {"resolution": "1280x720"}]}
        assert extract_max_height(info) == 1920  # max edge

    def test_vertical_resolution(self):
        info = {"formats": [{"resolution": "1080x1920"}]}
        assert extract_max_height(info) == 1920

    def test_top_level_height(self):
        info = {"height": 720, "formats": [{"acodec": "opus"}]}
        assert extract_max_height(info) == 720

    def test_format_note_p(self):
        info = {"formats": [{"format_note": "1080p"}, {"format_note": "720p"}]}
        assert extract_max_height(info) == 1080

    def test_audio_only_none(self):
        info = {"formats": [{"acodec": "opus", "vcodec": "none"}, {"format_id": "140"}]}
        assert extract_max_height(info) is None

    def test_height_preferred_over_note(self):
        info = {
            "formats": [
                {"height": 480, "format_note": "1080p"},
            ]
        }
        assert extract_max_height(info) == 480

    def test_empty_info(self):
        assert extract_max_height({}) is None
        assert extract_max_height(None) is None

    def test_storyboard_only_none(self):
        info = {
            "formats": [
                {"format_id": "sb0", "height": 90, "format_note": "storyboard"},
                {"format_id": "sb1", "height": 90},
            ]
        }
        assert extract_max_height(info) is None

    def test_storyboard_ignored_when_real_present(self):
        info = {
            "formats": [
                {"format_id": "sb0", "height": 90, "format_note": "storyboard"},
                {"format_id": "137", "height": 1080, "vcodec": "avc1"},
            ]
        }
        assert extract_max_height(info) == 1080

    def test_height_below_min_ignored(self):
        info = {"formats": [{"height": 90}, {"height": 120}]}
        assert extract_max_height(info) is None


class TestClassifyMissingHeight:
    def test_empty_formats(self):
        assert classify_missing_height({"formats": []}) == STATUS_EMPTY_FORMATS

    def test_storyboard_only(self):
        info = {"formats": [{"format_id": "sb0", "height": 90, "format_note": "storyboard"}]}
        assert classify_missing_height(info) == STATUS_EMPTY_FORMATS

    def test_audio_only_no_height(self):
        info = {"formats": [{"format_id": "140", "acodec": "opus", "vcodec": "none"}]}
        assert classify_missing_height(info) == STATUS_NO_HEIGHT

    def test_video_without_height_empty_formats(self):
        info = {"formats": [{"format_id": "137", "vcodec": "avc1", "acodec": "none"}]}
        assert classify_missing_height(info) == STATUS_EMPTY_FORMATS


class TestCircuitBreaker:
    def test_consecutive_no_ok(self):
        b = CircuitBreaker(window=50, fail_rate=0.8, max_no_ok=5)
        for _ in range(4):
            assert b.record(STATUS_EMPTY_FORMATS) is False
        assert b.record(STATUS_ERROR) is True

    def test_ok_resets_streak(self):
        b = CircuitBreaker(window=50, fail_rate=0.8, max_no_ok=5)
        for _ in range(4):
            b.record(STATUS_EMPTY_FORMATS)
        assert b.record(STATUS_OK) is False
        for _ in range(4):
            assert b.record(STATUS_EMPTY_FORMATS) is False

    def test_fail_rate_window(self):
        b = CircuitBreaker(window=10, fail_rate=0.8, max_no_ok=100)
        for _ in range(7):
            assert b.record(STATUS_EMPTY_FORMATS) is False
        for _ in range(2):
            assert b.record(STATUS_OK) is False
        # 窗口满 10：8 empty + 2 ok → rate 0.8 → 熔断
        assert b.record(STATUS_EMPTY_FORMATS) is True

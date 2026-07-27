"""welding_l0 单元测试。"""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR))

from core.welding_l0 import welding_l0_prefilter  # noqa: E402


class TestWeldingL0Prefilter:
    def test_channel_blacklist(self):
        assert welding_l0_prefilter(channel="TEDx Talks") == "l0:channel_blacklist"

    def test_too_short(self):
        assert welding_l0_prefilter(duration_str="30") == "l0:too_short:30s"

    def test_too_long(self):
        assert welding_l0_prefilter(duration_str=str(7 * 3600)) == "l0:too_long:25200s"

    def test_title_music(self):
        assert welding_l0_prefilter(title="Best Pop Music Mix 2024") == "l0:title:music"

    def test_welding_title_passes(self):
        assert welding_l0_prefilter(
            title="MIG Welding Tutorial for Beginners",
            channel="WeldingTipsAndTricks",
            duration_str="600",
        ) is None

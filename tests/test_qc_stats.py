"""core.qc_stats 单元测试。"""

import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR))

from core.qc_stats import QcStatsBoard, QcTfErrCounts  # noqa: E402


def _sample_df():
    return pd.DataFrame({
        "video_id": ["a", "b", "c", "d"],
        "qc_vision_result": ["T", "F", "ERROR", ""],
    })


class TestQcTfErrCounts:
    def test_from_dataframe(self):
        c = QcTfErrCounts.from_dataframe(_sample_df())
        assert c.n_t == 1 and c.n_f == 1 and c.n_err == 1
        assert c.pending == 1
        assert c.done == 3


class TestQcStatsBoard:
    def test_record_and_sync_updates_both(self):
        df = _sample_df()
        board = QcStatsBoard.from_dataframe(df, total_rows=4)
        df.at[3, "qc_vision_result"] = "T"
        board.record_and_sync("T", df)
        assert board.pass_counts.n_t == 1
        assert board.global_counts.n_t == 2
        pf = board.tqdm_postfix(4.5)
        assert pf["T"] == 1 and pf["ΣT"] == 2 and pf["s"] == "4.5"

    def test_reset_pass(self):
        board = QcStatsBoard.from_dataframe(_sample_df())
        board.record_pass("T")
        board.reset_pass(pass_total=10)
        assert board.pass_counts.n_t == 0
        assert board.pass_counts.total == 10

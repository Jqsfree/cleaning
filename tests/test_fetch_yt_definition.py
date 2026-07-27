"""fetch_yt_definition.filter_ge720 / pending mask 单测。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR / "tools"))

from fetch_yt_definition import (  # noqa: E402
    _default_ge720_paths,
    filter_ge720,
    get_pending_mask,
)


def test_filter_ge720(tmp_path: Path):
    orig = tmp_path / "orig.csv"
    defn = tmp_path / "defn.csv"
    keep = tmp_path / "keep.csv"
    drop = tmp_path / "drop.csv"
    pd.DataFrame(
        {
            "video_id": ["a", "b", "c"],
            "title": ["ta", "tb", "tc"],
            "is_hd": ["should_strip", "x", "y"],
        }
    ).to_csv(orig, index=False)
    pd.DataFrame(
        {
            "video_id": ["a", "b", "c"],
            "is_hd": ["1", "0", ""],
            "yt_definition": ["hd", "sd", ""],
            "definition_status": ["ok", "ok", "not_found"],
        }
    ).to_csv(defn, index=False)

    n_keep, n_drop = filter_ge720(str(orig), str(defn), str(keep), str(drop), strip_aux=True)
    assert n_keep == 1 and n_drop == 2
    k = pd.read_csv(keep, dtype=str)
    assert list(k["video_id"]) == ["a"]
    assert "is_hd" not in k.columns
    d = pd.read_csv(drop, dtype=str)
    assert set(d["video_id"]) == {"b", "c"}


def test_default_ge720_paths():
    k, d = _default_ge720_paths("/tmp/0724_yt_definition.csv")
    assert k.endswith("0724_大于720.csv")
    assert d.endswith("0724_低于720或缺失.csv")


def test_pending_mask():
    df = pd.DataFrame(
        {
            "definition_status": ["ok", "not_found", "error", "", "ok"],
            "yt_definition": ["hd", "", "", "", "sd"],
            "is_hd": ["1", "", "", "", "0"],
        }
    )
    idx = get_pending_mask(df, overwrite=False, retry_errors=False)
    # "" pending; error excluded without retry
    assert list(df.loc[idx].index) == [3]
    idx2 = get_pending_mask(df, overwrite=False, retry_errors=True)
    assert set(df.loc[idx2].index) == {2, 3}

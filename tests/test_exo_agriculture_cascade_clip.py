"""exo_agriculture CLIP margin / decide 纯计算测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_脚本"))

from categories.exo_agriculture.cascade_clip import decide, margins_from_feats  # noqa: E402


def test_margins_pos_beats_neg():
    feats = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    pos = np.array([[1.0, 0.0]], dtype=np.float32)
    neg = np.array([[0.0, 1.0]], dtype=np.float32)
    pairs = {"q1_field_harvest": (pos, neg)}
    scores = margins_from_feats(feats, pairs)
    assert scores["q1_field_harvest"][0] > 0
    assert scores["q1_field_harvest"][1] < 0


def test_decide_require_harvest_and_not_fake():
    scores = {
        "q1_field_harvest": np.array([-0.2, 0.2, 0.2, 0.2]),
        "q2_not_talk_diy": np.array([0.2, -0.1, 0.2, 0.2]),
        "q3_not_fake": np.array([0.2, 0.2, -0.1, 0.2]),
    }
    ok = np.array([True, True, True, True])
    thr = {k: 0.0 for k in scores}
    out = decide(
        scores,
        thresholds=thr,
        require=["q1_field_harvest", "q3_not_fake"],
        ok_mask=ok,
    )
    assert list(out) == ["clip_fail", "clip_pass", "clip_fail", "clip_pass"]


def test_no_thumb_kept_outside_fail():
    scores = {
        "q1_field_harvest": np.array([0.5]),
        "q3_not_fake": np.array([0.5]),
    }
    out = decide(
        scores,
        thresholds={"q1_field_harvest": 0.0, "q3_not_fake": 0.0},
        require=["q1_field_harvest", "q3_not_fake"],
        ok_mask=np.array([False]),
    )
    assert out[0] == "no_thumb"

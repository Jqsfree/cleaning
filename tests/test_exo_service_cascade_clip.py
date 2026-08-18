"""L3 CLIP margin / decide 纯计算测试（不加载模型）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_脚本"))

from categories.exo_service.cascade_clip import decide, margins_from_feats  # noqa: E402


def test_margins_pos_beats_neg():
    feats = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    pos = np.array([[1.0, 0.0]], dtype=np.float32)
    neg = np.array([[0.0, 1.0]], dtype=np.float32)
    pairs = {"q3_serving": (pos, neg)}
    scores = margins_from_feats(feats, pairs)
    assert scores["q3_serving"][0] > 0
    assert scores["q3_serving"][1] < 0


def test_decide_require_subset():
    scores = {
        "q1_two_people": np.array([-0.2, 0.2, 0.2]),
        "q2_not_talk_diy": np.array([0.2, -0.1, 0.2]),
        "q3_serving": np.array([0.2, 0.2, 0.2]),
    }
    ok = np.array([True, True, True])
    thr = {k: 0.0 for k in scores}
    out = decide(
        scores,
        thresholds=thr,
        require=["q3_serving"],
        ok_mask=ok,
    )
    # q1/q2 不进合取：仅 q3 决定
    assert list(out) == ["clip_pass", "clip_pass", "clip_pass"]


def test_no_thumb():
    scores = {"q3_serving": np.array([0.5])}
    out = decide(
        scores,
        thresholds={"q3_serving": 0.0},
        require=["q3_serving"],
        ok_mask=np.array([False]),
    )
    assert out[0] == "no_thumb"

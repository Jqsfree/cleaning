"""exo_fitness CLIP margin / decide 纯计算测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_脚本"))

from categories.exo_agriculture.cascade_clip import (  # noqa: E402
    decide,
    load_cfg,
    margins_from_feats,
    require_keys,
    write_live_pass,
)

_FITNESS_CFG = (
    ROOT / "02_脚本" / "categories" / "exo_fitness" / "rules" / "cascade_fitness_clip.toml"
)


def test_fitness_cfg_require_keys():
    cfg = load_cfg(_FITNESS_CFG)
    assert require_keys(cfg) == ["q1_gym_workout", "q2_not_talk_diy", "q3_not_fake"]


def test_margins_pos_beats_neg():
    feats = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    pos = np.array([[1.0, 0.0]], dtype=np.float32)
    neg = np.array([[0.0, 1.0]], dtype=np.float32)
    pairs = {"q1_gym_workout": (pos, neg)}
    scores = margins_from_feats(feats, pairs)
    assert scores["q1_gym_workout"][0] > 0
    assert scores["q1_gym_workout"][1] < 0


def test_decide_require_gym_workout_and_not_fake():
    scores = {
        "q1_gym_workout": np.array([-0.2, 0.2, 0.2, 0.2]),
        "q2_not_talk_diy": np.array([0.2, -0.1, 0.2, 0.2]),
        "q3_not_fake": np.array([0.2, 0.2, -0.1, 0.2]),
    }
    ok = np.array([True, True, True, True])
    thr = {k: 0.0 for k in scores}
    out = decide(
        scores,
        thresholds=thr,
        require=["q1_gym_workout", "q2_not_talk_diy", "q3_not_fake"],
        ok_mask=ok,
    )
    assert list(out) == ["clip_fail", "clip_fail", "clip_fail", "clip_pass"]


def test_no_thumb_kept_outside_fail():
    scores = {
        "q1_gym_workout": np.array([0.5]),
        "q3_not_fake": np.array([0.5]),
    }
    out = decide(
        scores,
        thresholds={"q1_gym_workout": 0.0, "q3_not_fake": 0.0},
        require=["q1_gym_workout", "q3_not_fake"],
        ok_mask=np.array([False]),
    )
    assert out[0] == "no_thumb"


def test_write_live_pass_only_clip_pass(tmp_path):
    import pandas as pd

    work = pd.DataFrame({"video_id": ["a", "b", "c"], "title": ["ta", "tb", "tc"]})
    scored = pd.DataFrame(
        {
            "video_id": ["a", "b", "c"],
            "clip_decision": ["clip_pass", "clip_fail", "clip_pass"],
            "clip_q1_gym_workout": [0.1, -0.2, 0.2],
        }
    )
    out = tmp_path / "live_pass.csv"
    n = write_live_pass(work, scored, out)
    assert n == 2
    got = pd.read_csv(out, dtype=str)
    assert list(got["video_id"]) == ["a", "c"]
    assert "title" in got.columns

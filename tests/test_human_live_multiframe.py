from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR))

from core.human_live_multiframe import (  # noqa: E402
    classify_thumbnail_person,
    classify_multiframe_rule,
    make_blind_sample,
    parse_vlm_label,
    resolve_label_groups,
    summarize_person_frames,
)
from tools.export_human_live_multiframe import _feature_tensor  # noqa: E402


def test_majority_large_person_frames_pass():
    boxes = [
        [(0, 0, 50, 80)],
        [(0, 0, 45, 80)],
        [(0, 0, 50, 70)],
        [(0, 0, 45, 75)],
        [],
        [],
    ]
    summary = summarize_person_frames(
        boxes,
        frame_sizes=[(100, 100)] * 6,
        min_person_area_ratio=0.25,
    )

    assert summary["large_person_frames"] == 4
    assert classify_multiframe_rule(summary, [0.1] * 6) == "T"


def test_game_dominant_frames_reject_small_facecam():
    boxes = [[(0, 0, 20, 20)]] * 6
    summary = summarize_person_frames(
        boxes,
        frame_sizes=[(100, 100)] * 6,
        min_person_area_ratio=0.08,
    )

    assert summary["person_frames"] == 6
    assert summary["large_person_frames"] == 0
    assert classify_multiframe_rule(summary, [0.9] * 6) == "F"


def test_borderline_person_coverage_is_uncertain():
    boxes = [[(0, 0, 40, 40)]] * 3 + [[], [], []]
    summary = summarize_person_frames(
        boxes,
        frame_sizes=[(100, 100)] * 6,
        min_person_area_ratio=0.08,
    )

    assert classify_multiframe_rule(summary, [0.1] * 6) == "U"


def test_missing_frames_remain_error():
    assert classify_multiframe_rule(None, None) == "ERROR"


def test_thumbnail_person_gate_drops_none_and_small_person():
    no_person = classify_thumbnail_person([], frame_size=(100, 100))
    small = classify_thumbnail_person(
        [(0, 0, 20, 20)],
        frame_size=(100, 100),
        min_person_area_ratio=0.08,
    )
    large = classify_thumbnail_person(
        [(0, 0, 40, 40)],
        frame_size=(100, 100),
        min_person_area_ratio=0.08,
    )

    assert no_person["action"] == "highconf_drop"
    assert no_person["reason"] == "no_person"
    assert small["action"] == "highconf_drop"
    assert small["reason"] == "small_person"
    assert large["action"] == "keep_candidate"


def test_thumbnail_decode_error_is_not_drop():
    result = classify_thumbnail_person(
        None,
        frame_size=None,
        error="missing_thumbnail",
    )

    assert result["action"] == "keep_error"
    assert result["reason"] == "missing_thumbnail"


def test_parse_vlm_label_requires_standalone_tfu():
    assert parse_vlm_label("T") == "T"
    assert parse_vlm_label("answer: F") == "F"
    assert parse_vlm_label("UNCERTAIN U") == "U"
    assert parse_vlm_label("The result is unclear") == "ERROR"


def test_feature_tensor_supports_transformers_pooling_output():
    class Output:
        pooler_output = torch.ones((2, 3))

    assert _feature_tensor(Output()).shape == (2, 3)
    assert _feature_tensor(torch.ones((1, 4))).shape == (1, 4)


def test_blind_sample_excludes_labels_and_spreads_channels():
    pool = pd.DataFrame({
        "video_id": [f"v{i}" for i in range(30)],
        "channel": [f"c{i}" for i in range(30)],
        "duration_seconds": list(range(30)),
    })

    sample = make_blind_sample(pool, {"v0", "v1"}, n=10, seed=7)

    assert len(sample) == 10
    assert not set(sample["video_id"]) & {"v0", "v1"}
    assert sample["video_id"].is_unique
    assert sample["channel"].is_unique
    assert sample["sample_stratum"].nunique() >= 4


def test_label_group_falls_back_to_source_ref_without_leakage():
    frame = pd.DataFrame({
        "video_id": ["a", "b", "c"],
        "channel": ["", "", "named"],
        "source_ref": ["channel-x", "channel-x", "channel-y"],
    })

    groups = resolve_label_groups(frame)

    assert groups.tolist() == ["channel-x", "channel-x", "named"]

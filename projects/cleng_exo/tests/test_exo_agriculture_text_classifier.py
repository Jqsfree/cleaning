from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import exo_agriculture_text_classifier as clf  # noqa: E402


def test_build_text_title_only_not_keyword():
    row = pd.Series({
        "title": "Harvesting mango in orchard",
        "channel": "Farm Life",
        "keyword": "污染采集词",
        "description": "should not appear",
    })
    text = clf.build_text(row)
    assert text == "Harvesting mango in orchard"
    assert "Farm Life" not in text
    assert "污染采集词" not in text
    assert "should not appear" not in text


def test_load_training_frame_prefers_human_qc_result(tmp_path: Path):
    path = tmp_path / "human.csv"
    pd.DataFrame([
        {
            "video_id": "a",
            "title": "harvest",
            "channel": "c",
            "qc_result": "T",
            "qc_text_result": "F",
        },
        {
            "video_id": "b",
            "title": "tutorial",
            "channel": "c",
            "qc_result": "F",
            "qc_text_result": "U",
        },
        {
            "video_id": "c",
            "title": "skip",
            "channel": "c",
            "qc_result": "U",
            "qc_text_result": "F",
        },
    ]).to_csv(path, index=False, encoding="utf-8-sig")

    frame = clf.load_training_frame([path])
    assert list(frame["video_id"]) == ["a", "b"]
    assert list(frame["label_kind"]) == ["T", "F"]
    assert list(frame["y"]) == [1, 0]


def test_load_training_frame_startswith_tf_and_keep_last(tmp_path: Path):
    a = tmp_path / "round1.csv"
    b = tmp_path / "round2.csv"
    pd.DataFrame([
        {"video_id": "x", "title": "old", "channel": "c", "qc_result": "T"},
        {"video_id": "y", "title": "noise", "channel": "c", "qc_result": "T|无声音"},
        {"video_id": "z", "title": "skip", "channel": "c", "qc_result": ""},
    ]).to_csv(a, index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"video_id": "x", "title": "new", "channel": "c", "qc_result": "F"},
        {"video_id": "y", "title": "noise2", "channel": "c", "qc_result": "F"},
    ]).to_csv(b, index=False, encoding="utf-8-sig")

    frame = clf.load_training_frame([a, b])
    by_id = frame.set_index("video_id")
    assert by_id.loc["x", "label_kind"] == "F"
    assert by_id.loc["x", "title"] == "new"
    assert by_id.loc["y", "label_kind"] == "F"
    assert "z" not in by_id.index



def test_pick_recall_threshold_maximizes_f_within_t_hurt_cap():
    labels = np.asarray(["F"] * 20 + ["T"] * 10)
    scores = np.asarray(
        [0.05] * 8 + [0.15] * 4 + [0.25] * 8
        + [0.10] * 3 + [0.30] * 7
    )
    picked = clf.pick_recall_threshold(labels, scores, max_t_hurt_rate=0.10, min_drop=5)
    assert picked is not None
    assert picked["f_recall"] > 0.2
    assert picked["t_hurt_rate"] <= 0.10


def test_pick_strict_threshold_zero_t_hurt():
    labels = np.asarray(["F"] * 8 + ["T", "T"])
    scores = np.asarray([0.01, 0.02, 0.03, 0.04, 0.08, 0.12, 0.30, 0.40, 0.50, 0.70])
    picked = clf.pick_strict_threshold(
        labels, scores, min_precision=0.95, max_u_hurt_rate=0.05, min_drop=3,
    )
    assert picked is not None
    assert picked["t_hurt"] == 0
    assert picked["drop_precision"] == 1.0
    assert picked["n_drop"] >= 3


def test_farm_action_vs_certain_drop_titles():
    assert clf.is_farm_action_title("Harvesting mango in orchard")
    assert clf.is_farm_action_title("Growing Tomatoes Ep 4, Tomato Bed Preparation")
    assert clf.is_farm_action_title("Raising baby silkworms on bamboo beds")
    assert clf.is_farm_action_title("Drip irrigation watering solution in the field")
    assert clf.is_certain_drop_title("How to grow tomatoes garden tips")
    assert clf.is_certain_drop_title("The Most Beautiful Village In The World")


def test_contrast_score_prefers_keep_sim():
    scores = clf.contrast_score(np.array([0.72, 0.30]), np.array([0.40, 0.70]), tau=0.08)
    assert scores[0] > 0.9
    assert scores[1] < 0.1


def test_rescue_keeps_farm_action_not_tutorials():
    assert clf.should_rescue_crop_harvest("PEACH PICKING TRIP!!! (i rode a tractor)")
    assert clf.should_rescue_crop_harvest("Growing Tomatoes Ep 4, Tomato Bed Preparation")
    assert not clf.should_rescue_crop_harvest("How to Prune Tomatoes | Garden Tips Tutorial")
    assert not clf.should_rescue_crop_harvest("Tractor Farming Game Harvester Wheat Farming")


def test_fewshot_prototypes_uses_human_t_and_certain_f():
    frame = pd.DataFrame([
        {"title": "Picking mangoes in orchard", "channel": "Farm", "label_kind": "T"},
        {"title": "Silkworm farm daily work", "channel": "Farm", "label_kind": "T"},
        {"title": "Beautiful village travel documentary", "channel": "TV", "label_kind": "F"},
        {"title": "Random unclear title", "channel": "Farm", "label_kind": "F"},
    ])
    keep, drop = clf.fewshot_prototypes(frame)
    assert any("Picking mangoes" in x for x in keep)
    assert any("Silkworm" in x for x in keep)
    assert any("village" in x.lower() for x in drop)
    assert not any("Random unclear" in x for x in drop)


def test_resolve_torch_device_explicit_wins():
    assert clf.resolve_torch_device("cpu") == "cpu"
    assert clf.resolve_torch_device("mps") == "mps"
    auto = clf.resolve_torch_device(None)
    assert auto in {"mps", "cuda", "cpu"}

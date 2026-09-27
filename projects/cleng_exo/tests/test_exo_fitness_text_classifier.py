from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import exo_fitness_text_classifier as clf  # noqa: E402


def test_build_text_skips_keyword():
    row = pd.Series({
        "title": "Barbell squat session",
        "channel": "GymReal",
        "keyword": "cardio playlist",
    })
    text = clf.build_text(row)
    assert "Barbell squat session" in text
    assert "GymReal" in text
    assert "cardio playlist" not in text


def test_load_qc_result_maps_f(tmp_path: Path):
    path = tmp_path / "qc.csv"
    pd.DataFrame([
        {"video_id": "a", "title": "Jargon explained", "channel": "Run", "qc_result": "F"},
        {"video_id": "b", "title": "HIIT class", "channel": "Gym", "qc_result": "T"},
        {"video_id": "c", "title": "x", "channel": "y", "qc_result": "U"},
    ]).to_csv(path, index=False)
    frame = clf.load_training_frame([path])
    assert list(frame["video_id"]) == ["a", "b"]
    assert list(frame["y"]) == [0, 1]


def test_pick_strict_no_t_hurt():
    labels = np.asarray(["F"] * 8 + ["T", "T"])
    scores = np.asarray([0.02, 0.03, 0.04, 0.05, 0.10, 0.40, 0.50, 0.55, 0.70, 0.80])
    picked = clf.pick_strict_threshold(labels, scores, min_precision=0.90, min_drop=3)
    assert picked is not None
    assert picked["t_hurt"] == 0
    assert picked["drop_precision"] >= 0.90

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import exo_medical_text_classifier as clf  # noqa: E402


def test_build_text_skips_keyword():
    row = pd.Series({
        "title": "Suture workshop",
        "description": "OR skills lab",
        "keyword": "污染采集词",
        "channel": "MedSchool",
    })
    text = clf.build_text(row)
    assert "Suture workshop" in text
    assert "OR skills lab" in text
    assert "污染采集词" not in text
    assert "MedSchool" not in text


def test_load_training_frame_maps_f_to_zero(tmp_path: Path):
    path = tmp_path / "qc.csv"
    pd.DataFrame([
        {"video_id": "a", "title": "NCLEX prep", "qc_text_result": "F"},
        {"video_id": "b", "title": "suture 101", "qc_text_result": "U"},
        {"video_id": "c", "title": "err", "qc_text_result": "ERROR"},
    ]).to_csv(path, index=False)
    frame = clf.load_training_frame([path])
    assert list(frame["video_id"]) == ["a", "b"]
    assert list(frame["y"]) == [0, 1]


def test_load_human_train_export(tmp_path: Path):
    path = tmp_path / "train_export.csv"
    pd.DataFrame([
        {"video_id": "a", "title": "lecture", "human_label": "fail", "description": ""},
        {"video_id": "b", "title": "suture", "human_label": "pass", "description": "lab"},
    ]).to_csv(path, index=False)
    frame = clf.load_training_frame([path])
    assert list(frame["y"]) == [0, 1]
    assert list(frame["label_kind"]) == ["F", "T"]


def test_pick_strict_threshold_prefers_safe_drops():
    labels = np.asarray(["F"] * 8 + ["U", "U"])
    scores = np.asarray([0.02, 0.03, 0.04, 0.05, 0.10, 0.40, 0.50, 0.55, 0.70, 0.80])
    picked = clf.pick_strict_threshold(
        labels, scores, min_precision=0.90, max_u_hurt_rate=0.05, min_drop=3,
    )
    assert picked is not None
    assert picked["drop_precision"] >= 0.90
    assert picked["u_hurt"] == 0

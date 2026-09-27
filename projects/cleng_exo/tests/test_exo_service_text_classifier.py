from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import exo_service_text_classifier as clf  # noqa: E402


def _write_qc(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def test_build_text_uses_title_and_description_not_keyword():
    row = pd.Series({
        "title": "Barber working on shift",
        "description": "Fade in the chair",
        "channel": "Shop Floor",
        "keyword": "污染采集词",
    })
    text = clf.build_text(row)
    assert text == "Barber working on shift Fade in the chair"
    assert "污染采集词" not in text
    assert "Shop Floor" not in text


def test_load_training_frame_excludes_error_and_deduplicates(tmp_path: Path):
    first = tmp_path / "run01.csv"
    second = tmp_path / "run02.csv"
    _write_qc(first, [
        {"video_id": "a", "title": "noise", "channel": "c", "qc_text_result": "F"},
        {"video_id": "b", "title": "maybe", "channel": "c", "qc_text_result": "U"},
        {"video_id": "err", "title": "failed", "channel": "c", "qc_text_result": "ERROR"},
    ])
    _write_qc(second, [
        {"video_id": "a", "title": "duplicate", "channel": "c", "qc_text_result": "U"},
        {"video_id": "c", "title": "target", "channel": "c", "qc_text_result": "T"},
    ])

    frame = clf.load_training_frame([first, second])

    assert list(frame["video_id"]) == ["a", "b", "c"]
    assert list(frame["y"]) == [0, 1, 1]
    assert set(frame["label_kind"]) == {"F", "U", "T"}
    assert list(frame["_qc_round"]) == ["run01", "run01", "run02"]


def test_threshold_metrics_keep_t_and_u_out_of_drop():
    labels = np.asarray(["F", "F", "U", "T"])
    scores = np.asarray([0.01, 0.20, 0.30, 0.40])

    metrics = clf.threshold_metrics(labels, scores, threshold=0.25)

    assert metrics["n_drop"] == 2
    assert metrics["drop_precision"] == 1.0
    assert metrics["t_hurt"] == 0
    assert metrics["u_hurt"] == 0
    assert metrics["u_hurt_rate"] == 0.0


def test_pick_strict_threshold_requires_safety_and_minimum_coverage():
    labels = np.asarray(["F"] * 8 + ["U", "T"])
    scores = np.asarray([0.01, 0.02, 0.03, 0.04, 0.08, 0.12, 0.30, 0.40, 0.50, 0.60])

    picked = clf.pick_strict_threshold(
        labels,
        scores,
        min_precision=0.95,
        max_u_hurt_rate=0.01,
        min_drop=3,
    )

    assert picked is not None
    assert picked["drop_threshold"] > 0.04
    assert picked["drop_precision"] == 1.0
    assert picked["t_hurt"] == 0
    assert picked["u_hurt"] == 0


def test_validate_independent_gate_rejects_u_hurt():
    labels = np.asarray(["F"] * 19 + ["U"])
    scores = np.asarray([0.01] * 20)

    result = clf.validate_independent_gate(labels, scores, threshold=0.15)

    assert result["passed"] is False
    assert "U" in result["reasons"][0] or any("U" in reason for reason in result["reasons"])


def test_pick_veto_threshold_maximizes_f_recall_and_keeps_a_visual_pool():
    labels = np.asarray(["F"] * 20 + ["U"] * 10)
    scores = np.asarray([0.05] * 18 + [0.40, 0.45] + [0.70] * 10)
    picked = clf.pick_veto_threshold(labels, scores, min_keep_rate=0.20, min_drop=5)
    assert picked is not None
    assert picked["f_recall"] >= 0.8
    assert picked["keep_rate"] >= 0.20


def test_train_and_calibrate_writes_model_without_error_rows(tmp_path: Path):
    snap = tmp_path / "qc.csv"
    rows = [
        {
            "video_id": f"f{i}",
            "title": f"makeup tutorial {i}",
            "description": "grwm lipstick",
            "channel": "beauty",
            "qc_text_result": "F",
        }
        for i in range(12)
    ] + [
        {
            "video_id": "t1",
            "title": "barber fade haircut on shift",
            "description": "salon floor",
            "channel": "salon",
            "qc_text_result": "T",
        },
        {
            "video_id": "u1",
            "title": "maybe spa work",
            "description": "unclear",
            "channel": "spa",
            "qc_text_result": "U",
        },
        {
            "video_id": "e1",
            "title": "api fail",
            "description": "",
            "channel": "x",
            "qc_text_result": "ERROR",
        },
    ]
    _write_qc(snap, rows)
    model = tmp_path / "clf.pkl"
    calib = tmp_path / "calib.json"

    result = clf.train_and_calibrate(
        [snap],
        model_path=model,
        calibration_path=calib,
    )

    assert result["n_train"] == 14
    assert result["n_f"] == 12
    assert result["n_t"] == 1
    assert result["n_u"] == 1
    assert model.is_file()
    assert calib.is_file()
    assert result["feature_fields"] == ["title", "description"]
    assert "keyword" not in result["feature_fields"]


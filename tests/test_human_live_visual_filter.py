from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.human_live_visual_classifier import (
    _select_candidate_name,
    _source_oof_profiles,
    _source_weights,
)

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR))

from core.visual_filter import (  # noqa: E402
    acceptance_decision,
    build_feature_matrix,
    choose_action_thresholds,
    exclude_video_ids,
    load_embedding_rows,
    normalize_qc_result,
    select_active_learning_sample,
    split_labeled_frame,
    train_grouped_visual_model,
    wilson_lower_bound,
    write_embedding_store,
)


def test_normalize_qc_result_handles_bool_and_tfu():
    values = pd.Series([True, False, "T", "f", "U", "", None])
    assert normalize_qc_result(values).tolist() == [
        True, False, True, False, None, None, None,
    ]


def test_split_labeled_frame_keeps_groups_separate_and_conflicts_for_review():
    df = pd.DataFrame({
        "video_id": [f"v{i}" for i in range(12)],
        "qc_result": ["T", "F"] * 6,
        "channel": [f"c{i // 2}" for i in range(12)],
    })
    out = split_labeled_frame(df, conflict_ids={"v1"}, seed=42)

    assert out.loc[out["video_id"] == "v1", "split"].item() == "review"
    usable = out[out["split"].isin(["train", "calibration", "holdout"])]
    per_group = usable.groupby("channel")["split"].nunique()
    assert (per_group == 1).all()
    assert set(usable["split"]) == {"train", "calibration", "holdout"}


def test_split_labeled_frame_falls_back_to_source_ref_for_blank_channel():
    df = pd.DataFrame({
        "video_id": [f"v{i}" for i in range(12)],
        "qc_result": ["T", "F"] * 6,
        "channel": [""] * 12,
        "source_ref": [f"https://youtube.com/@c{i // 2}" for i in range(12)],
    })

    out = split_labeled_frame(df, seed=42)

    assert out["label_group"].eq(out["source_ref"]).all()
    assert (out.groupby("source_ref")["split"].nunique() == 1).all()


def test_source_weights_give_each_source_equal_total_weight():
    source = pd.Series(["machine"] * 4 + ["human"] * 2)

    weights = _source_weights(source)

    assert np.isclose(weights[:4].sum(), weights[4:].sum())


def test_source_oof_profiles_use_machine_rows_for_machine_threshold():
    y = np.array([0, 0, 1, 1, 0, 1])
    probabilities = np.array([0.01, 0.10, 0.80, 0.90, 0.99, 0.02])
    source = pd.Series(["machine"] * 4 + ["human"] * 2)

    profiles = _source_oof_profiles(
        y,
        probabilities,
        source,
        target_pass_rate=0.85,
        max_overturn=0.08,
        min_labels=2,
    )

    assert profiles["machine"]["keep_method"] == "target_met"
    assert profiles["machine"]["drop_method"] == "target_met"
    assert profiles["machine"]["keep_precision"] == 1.0
    assert profiles["human"]["keep_method"] == "best_effort"


def test_mixed_candidate_selected_only_when_machine_oof_improves_and_targets_met():
    baseline = {
        "machine_oof_auc": 0.70,
        "profiles": {
            "machine": {"keep_method": "target_met", "drop_method": "target_met"},
        },
    }
    mixed = {
        "machine_oof_auc": 0.75,
        "profiles": {
            "machine": {"keep_method": "target_met", "drop_method": "target_met"},
        },
    }

    assert _select_candidate_name(baseline, mixed) == "mixed"
    mixed["profiles"]["machine"]["keep_method"] = "best_effort"
    assert _select_candidate_name(baseline, mixed) == "machine_only"


def test_exclude_video_ids_removes_all_training_labels():
    pool = pd.DataFrame({"video_id": ["a", "b", "c", "d"]})

    filtered = exclude_video_ids(pool, ["b", "d", "missing"])

    assert filtered["video_id"].tolist() == ["a", "c"]


def test_embedding_store_round_trip_by_video_id(tmp_path: Path):
    ids = ["a", "b", "c"]
    vectors = np.array([[1, 0], [0, 1], [0.5, 0.5]], dtype=np.float32)
    write_embedding_store(tmp_path, ids, vectors)

    loaded, found = load_embedding_rows(tmp_path, ["c", "missing", "a"])

    assert found == ["c", "a"]
    np.testing.assert_allclose(
        loaded.astype(np.float32),
        np.array([[0.5, 0.5], [1, 0]], dtype=np.float32),
        atol=1e-3,
    )


def test_choose_action_thresholds_respects_precision_and_overturn_targets():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    p = np.array([0.01, 0.05, 0.10, 0.25, 0.55, 0.70, 0.80, 0.90, 0.95, 0.99])
    thresholds = choose_action_thresholds(
        y,
        p,
        target_pass_rate=0.80,
        max_overturn=0.10,
        min_keep_labels=3,
        min_drop_labels=3,
    )

    assert thresholds["drop_threshold"] <= 0.25
    assert thresholds["keep_threshold"] <= 0.80
    assert thresholds["keep_precision"] >= 0.80
    assert thresholds["drop_overturn"] <= 0.10


def test_grouped_visual_model_returns_oof_probabilities():
    rng = np.random.default_rng(3)
    negative = rng.normal(-1, 0.2, size=(12, 4))
    positive = rng.normal(1, 0.2, size=(12, 4))
    embeddings = np.vstack([negative, positive]).astype(np.float32)
    y = np.array([0] * 12 + [1] * 12)
    groups = np.array([f"g{i // 2}" for i in range(24)])
    features = build_feature_matrix(
        embeddings,
        pos_sim=np.linspace(0.1, 0.9, 24),
        neg_sim=np.linspace(0.9, 0.1, 24),
        duration_seconds=np.full(24, 600),
    )

    model, oof = train_grouped_visual_model(
        features, y, groups, n_splits=3, seed=11,
    )

    assert len(oof) == len(y)
    assert np.isfinite(oof).all()
    assert oof[y == 1].mean() > oof[y == 0].mean()
    assert hasattr(model, "predict_proba")


def test_active_learning_sample_has_three_disjoint_routes():
    n = 40
    scored = pd.DataFrame({
        "video_id": [f"v{i}" for i in range(n)],
        "visual_prob": np.linspace(0.01, 0.99, n),
        "ml_action": (
            ["highconf_drop"] * 10
            + ["uncertain"] * 15
            + ["keep_candidate"] * 15
        ),
    })
    embeddings = np.stack([
        np.array([i / n, 1 - i / n], dtype=np.float32) for i in range(n)
    ])

    sample = select_active_learning_sample(
        scored,
        embeddings,
        n_boundary=6,
        n_diverse_keep=5,
        n_drop=4,
        seed=7,
    )

    assert sample["video_id"].is_unique
    assert sample["sample_route"].value_counts().to_dict() == {
        "boundary": 6,
        "diverse_keep": 5,
        "drop_overturn": 4,
    }


def test_acceptance_gate_uses_wilson_hours_and_overturn():
    lower = wilson_lower_bound(244, 270, confidence=0.90)
    assert lower > 0.85

    accepted = acceptance_decision(
        pass_count=244,
        labeled_count=270,
        kept_hours=90_000,
        overturn_count=3,
        drop_labeled_count=50,
        confidence=0.90,
        min_pass_lower=0.85,
        min_hours=80_000,
        max_overturn=0.08,
    )
    rejected = acceptance_decision(
        pass_count=220,
        labeled_count=270,
        kept_hours=90_000,
        overturn_count=3,
        drop_labeled_count=50,
        confidence=0.90,
        min_pass_lower=0.85,
        min_hours=80_000,
        max_overturn=0.08,
    )

    assert accepted["decision"] == "accept"
    assert rejected["decision"] == "reject"

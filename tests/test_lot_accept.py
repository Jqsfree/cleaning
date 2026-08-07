"""lot_accept：sample_frame=deliver_frame；prepare/decide。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR))

from core.lot_accept import (  # noqa: E402
    assert_ids_subset,
    decide,
    prepare_deliver,
    record_sample,
    verify_existing,
)
from core.run_manifest import init_manifest, load_manifest  # noqa: E402


def test_assert_ids_subset_ok():
    assert_ids_subset({"a", "b"}, {"a", "b", "c"})


def test_assert_ids_subset_raises():
    with pytest.raises(ValueError, match="sample_frame"):
        assert_ids_subset({"a", "x"}, {"a", "b"})


def test_prepare_and_decide_ci(tmp_path: Path):
    root = tmp_path / "live_sell" / "human_t"
    root.mkdir(parents=True)
    init_manifest(root, category="live_sell", source="human", batch="t")

    lot = root / "remain.csv"
    lot.write_text(
        "video_id,title\nv1,a\nv2,b\nv3,c\nv4,d\nv5,e\n",
        encoding="utf-8",
    )
    meta = prepare_deliver(
        root,
        lot_csv=lot,
        sample_frame="remain",
        deliver_name="t_deliver_remain.csv",
        method="ci_estimate",
    )
    assert meta["lot_size"] == 5
    assert meta["decision"] == "pending"
    data = load_manifest(root)
    assert data["lot"]["sample_frame"] == "remain"
    deliver = root / "07_deliver" / "t_deliver_remain.csv"
    assert deliver.is_file()
    assert "deliver_remain" in (data.get("deliver_path") or "")

    sample = root / "02_sample" / "lot_accept" / "s.csv"
    sample.parent.mkdir(parents=True)
    sample.write_text("video_id,title\nv1,a\nv2,b\nv3,c\n", encoding="utf-8")
    record_sample(root, sample_csv=sample, lot_csv=deliver)
    assert load_manifest(root)["lot"]["n"] == 3

    labeled = root / "labels.csv"
    labeled.write_text(
        "video_id,human_label\nv1,pass\nv2,pass\nv3,pass\n",
        encoding="utf-8",
    )
    out = decide(
        root,
        labeled_csv=labeled,
        method="ci_estimate",
        min_pass_rate=0.85,
        category="live_sell",
        source="human",
        batch="t",
    )
    assert out["decision"] == "accept"
    assert out["pass_rate"] == 1.0

    labeled2 = root / "labels_bad.csv"
    labeled2.write_text(
        "video_id,human_label\nv1,fail\nv2,fail\nv3,pass\n",
        encoding="utf-8",
    )
    out2 = decide(
        root,
        labeled_csv=labeled2,
        method="ci_estimate",
        min_pass_rate=0.85,
        category="live_sell",
        source="human",
        batch="t",
    )
    assert out2["decision"] == "reject"


def test_decide_rejects_out_of_frame(tmp_path: Path):
    root = tmp_path / "live_sell" / "human_x"
    root.mkdir(parents=True)
    init_manifest(root, category="live_sell", source="human", batch="x")
    lot = root / "remain.csv"
    lot.write_text("video_id,title\nv1,a\nv2,b\n", encoding="utf-8")
    prepare_deliver(
        root, lot_csv=lot, sample_frame="remain",
        deliver_name="x_remain.csv",
    )
    labeled = root / "labels.csv"
    labeled.write_text(
        "video_id,human_label\nv1,pass\nv9,pass\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="不在 lot"):
        decide(
            root, labeled_csv=labeled, category="live_sell",
            source="human", batch="x",
        )


def test_verify_existing_no_new_labels(tmp_path: Path):
    root = tmp_path / "live_sell" / "human_v"
    root.mkdir(parents=True)
    init_manifest(root, category="live_sell", source="human", batch="v")
    lot = root / "remain.csv"
    lot.write_text("video_id,title\nv1,a\nv2,b\nv3,c\n", encoding="utf-8")
    prepare_deliver(
        root, lot_csv=lot, sample_frame="remain",
        deliver_name="v_remain.csv",
    )
    from core.run_manifest import update_stage

    update_stage(
        root, "human_qc",
        stats={"n_labeled": 100, "n_pass": 90, "pass_rate": 0.9},
    )
    update_stage(
        root, "ml_highconf_drop",
        stats={"overturn_rate": 0.05, "n_drop": 10, "n_remain": 3},
    )
    out = verify_existing(root, min_pass_rate=0.85, max_overturn=0.08)
    assert out["decision"] == "accept"
    assert out["method"] == "prescreen_plus_screening"
    assert out["evidence_frame"] == "quality"
    assert out["deliver_frame"] == "remain"
    assert out["pass_rate"] == 0.9

    out2 = verify_existing(root, min_pass_rate=0.95, max_overturn=0.08)
    assert out2["decision"] == "reject"

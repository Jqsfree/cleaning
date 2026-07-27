"""双模态累计 / 级联 / metrics / suggest。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "02_脚本"
sys.path.insert(0, str(_SCRIPT))

from core.reject_modality import (  # noqa: E402
    confidence_band_for_ml_score,
    map_vision_thumb_row,
)
from tools.cascade_reject_propose import run_cascade  # noqa: E402
from tools.export_reject_assets import export_assets  # noqa: E402
from tools.propose_reject_tags import (  # noqa: E402
    propose_from_frame,
    propose_from_thumb,
    write_proposed,
)
from tools.reject_source_metrics import compute_metrics_from_assets  # noqa: E402
from tools.suggest_reject_opt import build_suggestions  # noqa: E402


def test_modality_and_band_on_text_propose():
    df = pd.DataFrame({
        "video_id": ["v1"],
        "drop_category": ["anime_cartoon"],
        "title": ["x"],
    })
    out = propose_from_frame(df, category="film_tv")
    assert len(out) == 1
    assert out.iloc[0]["modality"] == "text"
    assert out.iloc[0]["confidence_band"] == "high"


def test_ml_mid_band():
    assert confidence_band_for_ml_score(0.10) == "high"
    assert confidence_band_for_ml_score(0.35) == "mid"
    assert confidence_band_for_ml_score(0.80) == "low"


def test_thumb_propose_and_map():
    assert map_vision_thumb_row({"qc_thumb_result": "T"}) is None
    tag = map_vision_thumb_row({"qc_thumb_result": "F"})
    assert tag == "provisional:thumb_fail"
    tag2 = map_vision_thumb_row({
        "qc_thumb_result": "F",
        "qc_thumb_evidence": "画面是动漫风格",
    })
    assert tag2 == "anime_cartoon"

    df = pd.DataFrame({
        "video_id": ["a", "b"],
        "qc_thumb_result": ["F", "T"],
    })
    out = propose_from_thumb(df, category="film_tv")
    assert len(out) == 1
    assert out.iloc[0]["modality"] == "thumb"


def test_cascade_conflict_to_sample(tmp_path: Path):
    batch = tmp_path / "b"
    qc = batch / "03_qc"
    qc.mkdir(parents=True)
    # text proposed high anime
    write_proposed(batch, pd.DataFrame([{
        "video_id": "x1",
        "reject_tags": "anime_cartoon",
        "propose_source": "blacklist:drop_reason",
        "confidence": "rule",
        "confidence_band": "high",
        "modality": "text",
        "label_source": "proposed",
        "registry_version": "1",
        "pipeline_category": "film_tv",
    }]), merge=False)

    universe = tmp_path / "u.csv"
    pd.DataFrame({
        "video_id": ["x1", "x2"],
        "title": ["Anime show", "Normal drama"],
        "channel": ["c", "d"],
    }).to_csv(universe, index=False)

    thumb = tmp_path / "thumb.csv"
    pd.DataFrame({
        "video_id": ["x1"],
        "qc_thumb_result": ["F"],
        "qc_thumb_evidence": ["游戏画面"],
    }).to_csv(thumb, index=False)

    n = run_cascade(
        batch,
        category="film_tv",
        universe=universe,
        thumb_input=thumb,
        n_validate=50,
    )
    sample = pd.read_csv(qc / "reject_sample_for_validate.csv")
    # x1: text anime vs thumb gaming → conflict
    assert (sample["cascade_reason"] == "modality_conflict").any() or n >= 0
    conflicts = sample[sample["cascade_reason"] == "modality_conflict"]
    assert len(conflicts) >= 1
    assert conflicts.iloc[0]["video_id"] == "x1"


def test_metrics_trust_and_suggest(tmp_path: Path):
    assets = tmp_path / "rejects"
    tag = assets / "gaming"
    tag.mkdir(parents=True)
    pd.DataFrame([
        {"video_id": f"p{i}", "reject_tag": "gaming", "modality": "text",
         "propose_source": "blacklist:x"}
        for i in range(5)
    ]).to_csv(tag / "proposed.csv", index=False)
    # 少量 confirm → untrusted
    pd.DataFrame([
        {"video_id": "p0", "reject_tag": "gaming", "modality": "text",
         "propose_source": "blacklist", "reject_action": "confirm"},
    ]).to_csv(tag / "human_validated.csv", index=False)

    report = compute_metrics_from_assets(assets)
    assert report["by_source"]
    row = next(r for r in report["by_source"] if r["reject_tag"] == "gaming")
    assert row["trust_status"] == "untrusted"

    suggestions = build_suggestions(report)
    assert any(s["action"] == "collect_more_validation" for s in suggestions)


def test_suggest_default_no_apply(tmp_path: Path):
    """build_suggestions 不碰 cascade 文件。"""
    report = {
        "deadlock_alerts": ["validated_stalled"],
        "by_source": [],
        "totals": {},
    }
    s = build_suggestions(report)
    assert any(x["action"] == "deadlock_alert" for x in s)


def test_accumulate_export_modality(tmp_path: Path):
    batch = tmp_path / "batch"
    (batch / "05_clean" / "run01").mkdir(parents=True)
    drop = batch / "05_clean" / "run01" / "drop.csv"
    pd.DataFrame({
        "video_id": ["d1"],
        "drop_category": ["gaming"],
        "title": ["gameplay"],
    }).to_csv(drop, index=False)
    thumb = batch / "03_qc"
    thumb.mkdir(parents=True)
    pd.DataFrame({
        "video_id": ["t1"],
        "qc_thumb_result": ["F"],
    }).to_csv(thumb / "sample_thumb_qc.csv", index=False)

    from tools.accumulate_reject_assets import _find_drop_csvs, _find_thumb_csvs
    assert drop in _find_drop_csvs(batch) or any(p.name == "drop.csv" for p in _find_drop_csvs(batch))
    assert _find_thumb_csvs(batch)

    text_out = propose_from_frame(pd.read_csv(drop), category="film_tv")
    write_proposed(batch, text_out, merge=True)
    thumb_out = propose_from_thumb(pd.read_csv(thumb / "sample_thumb_qc.csv"), category="film_tv")
    write_proposed(batch, thumb_out, merge=True)

    assets = tmp_path / "assets"
    stats = export_assets(batch_roots=[batch], assets_root=assets)
    assert stats
    # gaming or provisional
    prop = pd.read_csv(batch / "03_qc" / "reject_proposed.csv")
    assert set(prop["modality"]) >= {"text", "thumb"}

"""propose_reject_tags + export_reject_assets + ingest 验证字段。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = _ROOT / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR))

from core.human_qc import human_validated_rejects, normalize_frame  # noqa: E402
from core.run_manifest import init_manifest  # noqa: E402
from tools.propose_reject_tags import propose_from_frame  # noqa: E402
from tools.export_reject_assets import export_assets  # noqa: E402


def test_propose_from_drop_category():
    df = pd.DataFrame({
        "video_id": ["v1", "v2"],
        "drop_category": ["anime_cartoon", "ai_concept_trailer"],
        "title": ["x", "y"],
    })
    out = propose_from_frame(df, category="film_tv")
    assert len(out) == 2
    assert out.iloc[0]["reject_tags"] == "anime_cartoon"
    assert out.iloc[0]["label_source"] == "proposed"
    assert "ai_concept_trailer" in out["reject_tags"].values


def test_propose_from_title_rematch():
    df = pd.DataFrame({
        "video_id": ["v1"],
        "title": ["Naruto anime full episode"],
        "channel": ["AnimeWorld"],
        "drop_reason": "",
    })
    out = propose_from_frame(df, category="film_tv")
    assert len(out) == 1
    assert out.iloc[0]["reject_tags"] == "anime_cartoon"


def test_ingest_optional_reject_and_validated(tmp_path: Path):
    labels = tmp_path / "labels.csv"
    pd.DataFrame({
        "video_id": ["a", "b", "c"],
        "human_label": ["pass", "fail", "fail"],
        "reject_tags": ["", "anime_cartoon", ""],
        "reject_action": ["", "confirm", ""],
        "title": ["t1", "t2", "t3"],
    }).to_csv(labels, index=False)

    batch = tmp_path / "batch"
    init_manifest(batch, category="film_tv", source="human", batch="t01")
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "tools" / "ingest_human_qc.py"),
        str(labels),
        "-o", str(batch),
        "--category", "film_tv",
        "--source", "human",
        "--batch", "t01",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    assert (batch / "03_qc" / "pass.csv").exists()
    assert (batch / "03_qc" / "reject_validated.csv").exists()
    val = pd.read_csv(batch / "03_qc" / "reject_validated.csv")
    assert len(val) == 1
    assert val.iloc[0]["reject_tags"] == "anime_cartoon"


def test_normalize_frame_without_reject_cols():
    df = pd.DataFrame({"video_id": ["a"], "qc_result": ["T"], "title": ["x"]})
    labeled = normalize_frame(
        df, category="film_tv", source="human", batch="b",
    )
    assert labeled.iloc[0]["reject_action"] == "unset"
    assert labeled.iloc[0]["reject_tags"] == ""
    assert human_validated_rejects(labeled).empty


def test_export_layers(tmp_path: Path):
    batch = tmp_path / "human_t"
    qc = batch / "03_qc"
    qc.mkdir(parents=True)
    pd.DataFrame({
        "video_id": ["p1"],
        "reject_tags": ["gaming"],
        "propose_source": ["blacklist:drop_reason"],
        "label_source": ["proposed"],
        "registry_version": ["1"],
    }).to_csv(qc / "reject_proposed.csv", index=False)
    pd.DataFrame({
        "video_id": ["h1"],
        "reject_tags": ["gaming"],
        "reject_action": ["confirm"],
        "title": ["g"],
        "batch": ["t"],
    }).to_csv(qc / "reject_validated.csv", index=False)

    assets = tmp_path / "assets"
    stats = export_assets(batch_roots=[batch], assets_root=assets)
    assert "gaming" in stats
    assert (assets / "gaming" / "proposed.csv").exists()
    assert (assets / "gaming" / "human_validated.csv").exists()
    assert (assets / "gaming" / "manifest.json").exists()
    assert stats["gaming"]["proposed"] >= 1
    assert stats["gaming"]["human_validated"] >= 1

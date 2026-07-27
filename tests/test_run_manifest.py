"""core.run_manifest 单测。"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR))

from core.run_manifest import (  # noqa: E402
    find_deliver_paths,
    format_list_table,
    format_paths_only,
    format_summary,
    init_manifest,
    iter_manifests,
    load_manifest,
    update_stage,
)


def test_manifest_roundtrip(tmp_path: Path):
    root = tmp_path / "film_tv" / "human_0724"
    path = init_manifest(
        root,
        category="film_tv",
        source="human",
        batch="0724",
        input_path="/tmp/in.csv",
    )
    assert path.exists()
    data = load_manifest(root)
    assert data["source"] == "human"
    assert data["batch"] == "0724"

    update_stage(root, "quality", paths={"keep": str(root / "01_quality" / "a.csv")})
    update_stage(root, "deliver", deliver_path=str(root / "07_deliver" / "out.csv"))
    data2 = load_manifest(root)
    assert "quality" in data2["stages"]
    assert data2["deliver_path"].endswith("out.csv")
    text = format_summary(data2)
    assert "human" in text and "quality" in text
    paths_txt = format_paths_only(data2)
    assert "quality.keep=" in paths_txt
    assert "deliver=" in paths_txt


def test_list_and_find_deliver(tmp_path: Path):
    runs = tmp_path / "runs"
    b1 = runs / "film_tv" / "human_0724"
    b2 = runs / "film_tv" / "machine_0725"
    init_manifest(b1, category="film_tv", source="human", batch="0724")
    init_manifest(b2, category="film_tv", source="machine", batch="0725")
    deliver = b1 / "07_deliver"
    deliver.mkdir(parents=True)
    out = deliver / "0724_deliver_ge720.csv"
    out.write_text("video_id\nv1\n", encoding="utf-8")
    update_stage(b1, "deliver", deliver_path=str(out))

    rows = iter_manifests(runs)
    assert len(rows) == 2
    table = format_list_table(rows)
    assert "film_tv" in table and "0724" in table

    hits = find_deliver_paths(runs, category="film_tv", source="human")
    assert len(hits) == 1
    assert hits[0]["deliver_path"].endswith("0724_deliver_ge720.csv")

    # 无 deliver_path 时 glob 07_deliver
    d2 = b2 / "07_deliver"
    d2.mkdir(parents=True)
    (d2 / "x.csv").write_text("a\n", encoding="utf-8")
    hits2 = find_deliver_paths(runs, category="film_tv", batch="0725")
    assert any(h["deliver_path"].endswith("x.csv") for h in hits2)

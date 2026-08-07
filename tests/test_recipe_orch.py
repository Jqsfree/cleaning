"""recipe / contracts / provenance / orchestrate 烟测。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR))

from core.batch_sop import evaluate_recipe_checklist, next_action  # noqa: E402
from core.contracts import validate_table  # noqa: E402
from core.provenance import build_provenance, file_sha256  # noqa: E402
from core.recipe import deliver_tool, load_recipe, needs_ge720  # noqa: E402
from core.run_manifest import init_manifest, load_manifest, update_stage  # noqa: E402


def test_recipe_film_tv_vs_live_sell_deliver():
    ft = load_recipe("film_tv")
    ls = load_recipe("live_sell")
    assert needs_ge720(ft, "human")
    assert not needs_ge720(ls, "human")
    assert deliver_tool(ft, "human") == "ge720"
    assert deliver_tool(ls, "human") == "copy_remain"


def test_recipe_qc_only_no_clean():
    r = load_recipe("ego_repair")
    ids = [s["id"] for s in r["flow"]["human"]]
    assert "clean" not in ids
    assert r["meta"]["has_cleaner"] is False


def test_next_action_empty_batch(tmp_path: Path):
    root = tmp_path / "film_tv" / "human_t1"
    root.mkdir(parents=True)
    init_manifest(root, category="film_tv", source="human", batch="t1")
    act = next_action(root, category="film_tv", source="human")
    assert act is not None
    assert act["id"] == "quality"


def test_next_after_quality(tmp_path: Path):
    root = tmp_path / "live_sell" / "human_t2"
    q = root / "01_quality"
    q.mkdir(parents=True)
    (q / "a_quality_0804.csv").write_text("video_id,title\nv1,t\n", encoding="utf-8")
    init_manifest(root, category="live_sell", source="human", batch="t2")
    act = next_action(root, category="live_sell", source="human")
    assert act is not None
    assert act["id"] == "sample"


def test_checklist_ego_no_clean_missing(tmp_path: Path):
    root = tmp_path / "ego_repair" / "human_t3"
    root.mkdir(parents=True)
    init_manifest(root, category="ego_repair", source="human", batch="t3")
    rows = evaluate_recipe_checklist(root, category="ego_repair", source="human")
    ids = {r["id"] for r in rows}
    assert "clean" not in ids


def test_contracts_missing_columns(tmp_path: Path):
    p = tmp_path / "bad.csv"
    p.write_text("foo\n1\n", encoding="utf-8")
    issues = validate_table(p, layer="quality", category="film_tv")
    assert any("HARD:" in i and "video_id" in i for i in issues)


def test_provenance_in_manifest(tmp_path: Path):
    root = tmp_path / "film_tv" / "human_t4"
    root.mkdir(parents=True)
    raw = tmp_path / "in.csv"
    raw.write_text("video_id,title\nv1,hello\n", encoding="utf-8")
    init_manifest(root, category="film_tv", source="human", batch="t4")
    prov = build_provenance(input_path=raw)
    assert "input_sha256" in prov
    assert file_sha256(raw)
    update_stage(root, "quality", paths={"keep": "x.csv"}, provenance=prov)
    entry = load_manifest(root)["stages"]["quality"]
    assert entry["provenance"]["input_sha256"]


def test_orchestrate_next_cli(tmp_path: Path):
    root = tmp_path / "film_tv" / "human_cli"
    root.mkdir(parents=True)
    init_manifest(root, category="film_tv", source="human", batch="cli")
    orch = _SCRIPT_DIR / "pipeline" / "orchestrate.py"
    proc = subprocess.run(
        [sys.executable, str(orch), "next", "-o", str(root)],
        capture_output=True,
        text=True,
        cwd=str(_SCRIPT_DIR.parent),
    )
    assert proc.returncode == 0, proc.stderr
    assert "NEXT: quality" in proc.stdout


def test_orchestrate_run_requires_upto(tmp_path: Path):
    root = tmp_path / "film_tv" / "human_x"
    root.mkdir(parents=True)
    init_manifest(root, category="film_tv", source="human", batch="x")
    orch = _SCRIPT_DIR / "pipeline" / "orchestrate.py"
    proc = subprocess.run(
        [sys.executable, str(orch), "run", "-o", str(root)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0


def test_orchestrate_run_upto_sample(tmp_path: Path):
    root = tmp_path / "live_sell" / "human_s"
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "video_id,title,duration_seconds,channel,keyword\n"
        "vid001,Live Selling Demo Product,600,Shop,kw\n"
        "vid002,Another Live Sale Item Here,900,Shop,kw\n",
        encoding="utf-8",
    )
    orch = _SCRIPT_DIR / "pipeline" / "orchestrate.py"
    proc = subprocess.run(
        [
            sys.executable, str(orch), "run",
            "-o", str(root),
            "--category", "live_sell",
            "--source", "human",
            "--batch", "s",
            "--upto", "sample",
            "--input", str(raw),
            "--sample-n", "2",
        ],
        capture_output=True,
        text=True,
        cwd=str(_SCRIPT_DIR.parent),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (root / "01_quality").is_dir()
    assert (root / "02_sample").is_dir()
    man = load_manifest(root)
    assert "quality" in man.get("stages", {})
    # provenance on quality
    assert man["stages"]["quality"].get("provenance", {}).get("input_sha256")


def test_live_sell_layer1_stop_and_machine_next(tmp_path: Path):
    r = load_recipe("live_sell")
    assert r["meta"].get("layer1_stop") == "rules"
    root = tmp_path / "live_sell" / "machine_l1"
    q = root / "01_quality"
    s = root / "02_sample"
    q.mkdir(parents=True)
    s.mkdir(parents=True)
    (q / "a_quality_0804.csv").write_text("video_id,title\nv1,t\n", encoding="utf-8")
    (s / "a_sample_0804.csv").write_text("video_id,title\nv1,t\n", encoding="utf-8")
    init_manifest(root, category="live_sell", source="machine", batch="l1")
    act = next_action(root, category="live_sell", source="machine")
    assert act is not None
    assert act["id"] == "text_qc"


def test_orchestrate_text_qc_requires_flag(tmp_path: Path):
    root = tmp_path / "live_sell" / "machine_t"
    root.mkdir(parents=True)
    init_manifest(root, category="live_sell", source="machine", batch="t")
    (root / "01_quality").mkdir()
    (root / "02_sample").mkdir()
    (root / "01_quality" / "a_quality_0804.csv").write_text(
        "video_id,title\nv1,t\n", encoding="utf-8",
    )
    (root / "02_sample" / "a_sample_0804.csv").write_text(
        "video_id,title\nv1,t\n", encoding="utf-8",
    )
    orch = _SCRIPT_DIR / "pipeline" / "orchestrate.py"
    proc = subprocess.run(
        [
            sys.executable, str(orch), "run",
            "-o", str(root),
            "--category", "live_sell",
            "--source", "machine",
            "--upto", "text_qc",
        ],
        capture_output=True,
        text=True,
        cwd=str(_SCRIPT_DIR.parent),
    )
    assert proc.returncode != 0
    assert "run-text-qc" in (proc.stderr + proc.stdout)


def test_write_rules_notes(tmp_path: Path):
    root = tmp_path / "live_sell" / "machine_n"
    qc = root / "03_qc"
    qc.mkdir(parents=True)
    (root / "01_quality").mkdir(parents=True)
    (root / "02_sample").mkdir(parents=True)
    (root / "01_quality" / "a_quality_0804.csv").write_text(
        "video_id,title\nv1,t\n", encoding="utf-8",
    )
    (root / "02_sample" / "x_sample.csv").write_text(
        "video_id,title\nv1,t\n", encoding="utf-8",
    )
    qc_csv = qc / "x_text_qc.csv"
    qc_csv.write_text(
        "video_id,title,qc_text_result\n"
        "v1,a,T\nv2,b,F\nv3,c,U\nv4,d,ERROR\n",
        encoding="utf-8",
    )
    init_manifest(root, category="live_sell", source="machine", batch="n")
    helper = _SCRIPT_DIR / "tools" / "write_rules_notes.py"
    proc = subprocess.run(
        [sys.executable, str(helper), "-o", str(root)],
        capture_output=True,
        text=True,
        cwd=str(_SCRIPT_DIR.parent),
    )
    assert proc.returncode == 0, proc.stderr
    notes = root / "04_rules" / "NOTES.md"
    assert notes.is_file()
    text = notes.read_text(encoding="utf-8")
    assert "T=1" in text and "F=1" in text and "U=1" in text
    assert "停在" in text
    act = next_action(root, category="live_sell", source="machine")
    # quality/sample/text_qc/rules 齐 → next 为后续层 clean（第一层已完成）
    assert act is not None
    assert act["id"] == "clean"
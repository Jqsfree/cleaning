"""架构硬化：batch_layout / clean_gates / io / rules / manifest merge。"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR))

from core.batch_layout import (  # noqa: E402
    evaluate_checklist,
    infer_batch_root,
    looks_like_batch_root,
    require_output_dir,
)
from core.category_registry import (  # noqa: E402
    CLEANER_MODULES,
    QC_ONLY_CATEGORIES,
    list_cleaner_categories,
)
from core.clean_gates import (  # noqa: E402
    assert_clean_gates,
    assert_clean_not_raw,
    human_fail_input_ok,
    machine_rules_evidence,
)
from core.io import warn_csv_row_skew  # noqa: E402
from core.rules_loader import load_blacklist, save_hit_cache  # noqa: E402
from core.run_manifest import (  # noqa: E402
    init_manifest,
    load_manifest,
    maybe_update_stage,
    update_stage,
)
from core.sql_builder import count_rule_hits, load_raw_table, require_columns  # noqa: E402


def test_batch_root_inference(tmp_path: Path):
    root = tmp_path / "film_tv" / "human_0724"
    q = root / "01_quality"
    q.mkdir(parents=True)
    assert looks_like_batch_root(root)
    assert infer_batch_root(q) == root.resolve()
    assert infer_batch_root(root / "05_clean" / "run01") == root.resolve()


def test_require_output_dir():
    with pytest.raises(SystemExit):
        require_output_dir(None)
    assert require_output_dir("a/b/") == "a/b"


def test_manifest_merge_preserves_stages(tmp_path: Path):
    root = tmp_path / "film_tv" / "machine_0727"
    init_manifest(root, category="film_tv", source="machine", batch="0727")
    update_stage(root, "quality", paths={"keep": "a.csv"})
    update_stage(root, "human_qc", paths={"pass": "p.csv"}, stats={"n": 1})
    init_manifest(
        root, category="film_tv", source="machine", batch="0727",
        input_path="raw.csv",
    )
    data = load_manifest(root)
    assert "quality" in data["stages"]
    assert "human_qc" in data["stages"]
    assert data["input"] == "raw.csv"

    init_manifest(
        root, category="film_tv", source="machine", batch="0727", reinit=True,
    )
    data2 = load_manifest(root)
    assert data2["stages"] == {}


def test_update_stage_merges_paths(tmp_path: Path):
    root = tmp_path / "b"
    init_manifest(root, category="film_tv", source="human", batch="1")
    update_stage(root, "quality", paths={"keep": "a.csv"})
    update_stage(root, "quality", paths={"drop": "b.csv"}, stats={"n": 2})
    entry = load_manifest(root)["stages"]["quality"]
    assert entry["paths"]["keep"] == "a.csv"
    assert entry["paths"]["drop"] == "b.csv"
    assert entry["stats"]["n"] == 2


def test_machine_rules_evidence(tmp_path: Path):
    root = tmp_path / "film_tv" / "machine_1"
    (root / "04_rules").mkdir(parents=True)
    (root / "04_rules" / "NOTES.md").write_text("ok", encoding="utf-8")
    ok, msg = machine_rules_evidence(root)
    assert ok and "04_rules" in msg

    root2 = tmp_path / "film_tv" / "machine_2"
    root2.mkdir(parents=True)
    ok2, _ = machine_rules_evidence(root2)
    assert not ok2


def test_human_fail_input():
    ok, _ = human_fail_input_ok("/x/03_qc/fail.csv")
    assert ok
    ok2, _ = human_fail_input_ok("/x/01_quality/keep.csv")
    assert not ok2


def test_assert_clean_not_raw(tmp_path: Path):
    raw = tmp_path / "raw.csv"
    raw.write_text("a\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        assert_clean_not_raw(
            keep_path=raw, ran_quality=False, source="machine",
        )
    assert_clean_not_raw(
        keep_path=raw, ran_quality=False, source="machine", skip_evidence=True,
    )


def test_load_raw_requires_columns(tmp_path: Path):
    p = tmp_path / "bad.csv"
    p.write_text("foo,bar\n1,2\n", encoding="utf-8")
    con = duckdb.connect()
    with pytest.raises(ValueError, match="缺少必填列"):
        load_raw_table(con, str(p))


def test_warn_csv_row_skew(tmp_path: Path, capsys):
    p = tmp_path / "a.csv"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    n = warn_csv_row_skew(str(p), loaded_rows=1)
    assert n == 1


def test_count_rule_hits_raises_on_bad_regex():
    con = duckdb.connect()
    con.execute("CREATE TABLE t AS SELECT 'hello' AS search_text")
    with pytest.raises(RuntimeError, match="规则命中统计失败"):
        count_rule_hits(
            con, "t",
            [{"category": "bad", "pattern": "(unclosed"}],
        )


def test_load_blacklist_hit_cache_order(tmp_path: Path):
    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / "blacklist.toml").write_text(
        '[[pass2]]\ncategory = "low"\npattern = "aaa"\n\n'
        '[[pass2]]\ncategory = "high"\npattern = "bbb"\n',
        encoding="utf-8",
    )
    save_hit_cache(rules, {"pass2": {"high": 100, "low": 1}})
    bl = load_blacklist(rules)
    # high 命中多应排在前面
    assert bl["pass2"].startswith("bbb")


def test_checklist(tmp_path: Path):
    root = tmp_path / "film_tv" / "machine_x"
    (root / "01_quality").mkdir(parents=True)
    (root / "01_quality" / "a.csv").write_text("x\n", encoding="utf-8")
    init_manifest(root, category="film_tv", source="machine", batch="x")
    rows = evaluate_checklist(root, "machine")
    by_id = {r["id"]: r["status"] for r in rows}
    assert by_id["quality"] == "ok"
    assert by_id["manifest"] == "ok"
    assert by_id["sample"] == "missing"
    assert by_id["rules"] == "missing"  # machine required


def test_dedup_find_07_deliver(tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        "dedup06",
        _SCRIPT_DIR / "pipeline" / "06_dedup.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)

    batch = tmp_path / "film_tv" / "human_1"
    d = batch / "07_deliver"
    d.mkdir(parents=True)
    f = d / "0724_deliver_ge720.csv"
    f.write_text("video_id\nv1\n", encoding="utf-8")
    found = mod.find_deliver_csvs(str(batch))
    assert any(p.endswith("0724_deliver_ge720.csv") for p in found)


def test_category_registry_single_source():
    assert "film_tv" in CLEANER_MODULES
    assert "ego_repair" in QC_ONLY_CATEGORIES
    assert list_cleaner_categories() == sorted(CLEANER_MODULES)


def test_maybe_update_stage_soft_init(tmp_path: Path):
    root = tmp_path / "film_tv" / "human_0804"
    q = root / "01_quality"
    q.mkdir(parents=True)
    keep = q / "x_quality_0804.csv"
    keep.write_text("video_id,title\nv1,t\n", encoding="utf-8")
    assert not (root / "manifest.json").exists()
    ok = maybe_update_stage(
        q, "quality", paths={"keep": str(keep)}, stats={"keep": 1},
    )
    assert ok
    data = load_manifest(root)
    assert data["source"] == "human"
    assert data["batch"] == "0804"
    assert data["category"] == "film_tv"
    assert data["stages"]["quality"]["paths"]["keep"] == str(keep)


def test_machine_clean_blocked_without_evidence(tmp_path: Path):
    root = tmp_path / "film_tv" / "machine_0804"
    cdir = root / "05_clean" / "run01"
    cdir.mkdir(parents=True)
    inp = root / "01_quality" / "keep.csv"
    inp.parent.mkdir(parents=True)
    inp.write_text("video_id,title\nv1,t\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        assert_clean_gates(
            source="machine",
            input_path=str(inp),
            output_dir=str(cdir),
            rules_ready=True,
            allow_clean=False,
            skip_evidence=False,
        )


def test_human_run_py_blocks_quality_keep_clean(tmp_path: Path):
    """人工经 run.py 对 quality keep 做 clean 须被拒（无 --skip-evidence）。"""
    root = tmp_path / "film_tv" / "human_0804"
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "video_id,title,duration_seconds,channel_name,keyword\n"
        "vid001,Hello World Movie Trailer,120,Ch,kw\n",
        encoding="utf-8",
    )
    run_py = _SCRIPT_DIR / "pipeline" / "run.py"
    proc = subprocess.run(
        [
            sys.executable, str(run_py), str(raw),
            "--category", "film_tv",
            "--source", "human",
            "-o", str(root),
            "--stages", "quality,clean",
            "--allow-clean",
        ],
        capture_output=True,
        text=True,
        cwd=str(_SCRIPT_DIR.parent),
    )
    assert proc.returncode == 2
    assert "不合 SOP" in (proc.stdout + proc.stderr)


def test_standalone_quality_updates_manifest(tmp_path: Path):
    root = tmp_path / "film_tv" / "machine_0804"
    qdir = root / "01_quality"
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "video_id,title,duration_seconds,channel_name,keyword\n"
        "vid001,Hello World Movie Trailer,120,Ch,kw\n",
        encoding="utf-8",
    )
    q_py = _SCRIPT_DIR / "pipeline" / "01_quality.py"
    proc = subprocess.run(
        [sys.executable, str(q_py), str(raw), "-o", str(qdir)],
        capture_output=True,
        text=True,
        cwd=str(_SCRIPT_DIR.parent),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = load_manifest(root)
    assert "quality" in data["stages"]
    assert data["stages"]["quality"]["paths"].get("keep")

"""drop 回流抽样单测。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR))

from core.run_manifest import init_manifest, load_manifest  # noqa: E402

# 直接测抽样函数
sys.path.insert(0, str(_SCRIPT_DIR / "tools"))
import sample_drop_for_reqc as sdr  # noqa: E402


def test_sample_size_manual():
    df = pd.DataFrame({"video_id": [f"v{i}" for i in range(100)]})
    out = sdr.sample_drop(df, n=20, seed=1)
    assert len(out) == 20
    out2 = sdr.sample_drop(df, n=20, seed=1)
    assert list(out["video_id"]) == list(out2["video_id"])


def test_sample_size_capped():
    df = pd.DataFrame({"video_id": ["a", "b", "c"]})
    out = sdr.sample_drop(df, n=50, seed=0)
    assert len(out) == 3


def test_cli_writes_drop_reflux(tmp_path: Path):
    import subprocess

    batch = tmp_path / "film_tv" / "human_0724"
    init_manifest(batch, category="film_tv", source="human", batch="0724")
    drop = tmp_path / "drop.csv"
    pd.DataFrame({"video_id": [f"d{i}" for i in range(50)], "title": ["x"] * 50}).to_csv(
        drop, index=False,
    )
    cli = _SCRIPT_DIR / "tools" / "sample_drop_for_reqc.py"
    r = subprocess.run(
        [
            sys.executable, str(cli), str(drop),
            "--batch-root", str(batch),
            "-n", "10",
            "--seed", "7",
        ],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    sample = batch / "03_qc" / "drop_reflux" / "sample.csv"
    assert sample.exists()
    assert len(pd.read_csv(sample)) == 10
    assert (batch / "03_qc" / "drop_reflux" / "README.txt").exists()
    man = load_manifest(batch)
    assert "drop_reflux" in man.get("stages", {})
    assert "ingest_human_qc" in r.stdout

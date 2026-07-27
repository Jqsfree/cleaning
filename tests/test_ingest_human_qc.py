"""人工质检入库：标签归一化 + pass/fail 写出。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR))

from core.human_qc import (  # noqa: E402
    normalize_frame,
    normalize_label,
    pass_rate,
    split_pass_fail,
)
from core.run_manifest import init_manifest, load_manifest  # noqa: E402


def test_normalize_label_variants():
    assert normalize_label("T") == "pass"
    assert normalize_label("合格") == "pass"
    assert normalize_label("F") == "fail"
    assert normalize_label("不合格") == "fail"
    assert normalize_label("pass") == "pass"
    assert normalize_label("???") is None
    assert normalize_label("") is None


def test_normalize_frame_and_split(tmp_path: Path):
    df = pd.DataFrame({
        "video_id": ["a", "b", "c", "d"],
        "qc_result": ["T", "F", "合格", "未知"],
        "title": ["t1", "t2", "t3", "t4"],
    })
    labeled = normalize_frame(
        df,
        category="film_tv",
        source="human",
        batch="0724",
        dimension="text",
    )
    # 「未知」被丢弃
    assert len(labeled) == 3
    assert set(labeled["human_label"]) == {"pass", "fail"}
    assert labeled["qc_dimension"].iloc[0] == "text"

    pass_df, fail_df = split_pass_fail(labeled)
    assert len(pass_df) == 2
    assert len(fail_df) == 1
    assert abs(pass_rate(labeled) - 2 / 3) < 1e-9


def test_ingest_cli_writes_files(tmp_path: Path):
    # 通过直接调用 normalize + 写文件验证契约；CLI 路径用 subprocess 更重
    batch = tmp_path / "film_tv" / "human_0724"
    init_manifest(
        batch, category="film_tv", source="human", batch="0724",
    )
    raw = pd.DataFrame({
        "video_id": ["v1", "v2", "v3"],
        "human_label": ["pass", "fail", "pass"],
        "title": ["A", "B", "C"],
        "channel": ["c1", "c2", "c3"],
    })
    src = tmp_path / "labels.csv"
    raw.to_csv(src, index=False)

    import subprocess
    cli = _SCRIPT_DIR / "tools" / "ingest_human_qc.py"
    r = subprocess.run(
        [
            sys.executable, str(cli), str(src),
            "-o", str(batch),
            "--category", "film_tv",
            "--source", "human",
            "--batch", "0724",
            "--dimension", "overall",
        ],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert (batch / "03_qc" / "pass.csv").exists()
    assert (batch / "03_qc" / "fail.csv").exists()
    assert (batch / "03_qc" / "labeled.csv").exists()
    assert (batch / "03_qc" / "train_export.csv").exists()
    pass_df = pd.read_csv(batch / "03_qc" / "pass.csv")
    fail_df = pd.read_csv(batch / "03_qc" / "fail.csv")
    assert len(pass_df) == 2 and len(fail_df) == 1
    man = load_manifest(batch)
    assert "human_qc" in man.get("stages", {})
    assert "唯一 KPI" in r.stdout or "人工合格率" in r.stdout

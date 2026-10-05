"""tools/batch_ops/sample_qc 单测（纯函数层，不走 CLI / 不写项目记录）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_脚本"))

from tools.batch_ops.sample_qc import (  # noqa: E402
    calc_sample_size,
    effective_margin,
    build_sample,
)


def _pool(path: Path, n_per_kw: int = 40, kws=("咖啡店", "理发", "美容")) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for kw in kws:
        for i in range(n_per_kw):
            rows.append({
                "video_id": f"{kw}_{i}",
                "title": f"{kw} 场景 {i}",
                "keyword": kw,
                "duration_seconds": 120 + i,
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ── 样本量公式 ──

def test_calc_sample_size_matches_textbook_value():
    # N 很大时退化为 n = Z²p(1-p)/e² = 1.96²·0.25/0.05² ≈ 384
    assert calc_sample_size(10_000_000, 95, 0.05) == 384


def test_calc_sample_size_applies_finite_population_correction():
    assert calc_sample_size(300, 95, 0.05) < calc_sample_size(10_000_000, 95, 0.05)


def test_calc_sample_size_never_below_one():
    assert calc_sample_size(1, 95, 0.05) >= 1


def test_effective_margin_shrinks_with_more_samples():
    wide = effective_margin(100, 100_000)
    narrow = effective_margin(10_000, 100_000)
    assert wide > narrow > 0


def test_effective_margin_roundtrips_formula():
    """请求 ±5% 时，按公式反推的样本量应给出 ≈5% 的实际误差。"""
    n_total = 1_000_000
    n = calc_sample_size(n_total, 95, 0.05)
    assert effective_margin(n, n_total, 95, 0.5) == pytest.approx(0.05, abs=1e-3)


def test_effective_margin_degenerate_inputs():
    assert effective_margin(0, 100) != effective_margin(0, 100)  # NaN
    assert effective_margin(10, 0) != effective_margin(10, 0)   # NaN


# ── build_sample ──

def test_srs_sample_size_and_artifacts(tmp_path: Path):
    pool = _pool(tmp_path / "pool.csv", n_per_kw=40)
    out_dir = tmp_path / "data/runs/exo_service/machine_0813/02_sample"

    stats = build_sample(str(pool), str(out_dir), sample_size=10, seed=7)

    assert stats["total_rows"] == 120
    assert stats["sample_size"] == 10
    assert stats["stratify"] is False
    assert Path(stats["sample_csv"]).is_file()
    assert Path(stats["sample_parquet"]).is_file()
    assert Path(stats["audit_stats_csv"]).is_file()
    assert stats["audit_strata_csv"] == ""
    assert len(pd.read_csv(stats["sample_csv"])) == 10


def test_srs_is_reproducible(tmp_path: Path):
    pool = _pool(tmp_path / "pool.csv")
    a = build_sample(str(pool), str(tmp_path / "a"), sample_size=10, seed=7)
    b = build_sample(str(pool), str(tmp_path / "b"), sample_size=10, seed=7)
    assert (
        sorted(pd.read_csv(a["sample_csv"])["video_id"])
        == sorted(pd.read_csv(b["sample_csv"])["video_id"])
    )


def test_output_naming_uses_category_and_date(tmp_path: Path):
    pool = _pool(tmp_path / "pool.csv")
    out_dir = tmp_path / "data/runs/exo_service/machine_0813/02_sample"

    stats = build_sample(str(pool), str(out_dir), sample_size=5)

    expected = f"exo_service_sample_{time.strftime('%m%d')}.csv"
    assert Path(stats["sample_csv"]).name == expected
    assert Path(stats["sample_parquet"]).name == expected.replace(".csv", ".parquet")


def test_name_tag_overrides_category(tmp_path: Path):
    pool = _pool(tmp_path / "pool.csv")
    out_dir = tmp_path / "data/runs/exo_service/machine_0813/02_sample"

    stats = build_sample(str(pool), str(out_dir), sample_size=5, name_tag="custom")

    assert Path(stats["sample_csv"]).name.startswith("custom_sample_")


def test_sample_size_is_capped_by_pool(tmp_path: Path):
    pool = _pool(tmp_path / "pool.csv", n_per_kw=2)
    stats = build_sample(str(pool), str(tmp_path / "out"), sample_size=999)

    assert stats["sample_size"] == 6
    assert stats["sample_size_target"] == 6


def test_stratified_quota_sums_to_target(tmp_path: Path):
    """回归：最大余额法须保证 Σ配额 == n_target（旧实现 270 会缩水到 180）。"""
    pool = _pool(tmp_path / "pool.csv", n_per_kw=40)   # 3 层 × 40
    out_dir = tmp_path / "data/runs/exo_service/machine_0813/02_sample"

    stats = build_sample(str(pool), str(out_dir), sample_size=30, stratify=True, seed=42)

    assert stats["sample_size"] == 30
    assert stats["sample_size_target"] == 30
    assert len(pd.read_csv(stats["sample_csv"])) == 30


def test_stratified_writes_weights(tmp_path: Path):
    pool = _pool(tmp_path / "pool.csv", n_per_kw=40)
    out_dir = tmp_path / "data/runs/exo_service/machine_0813/02_sample"

    stats = build_sample(str(pool), str(out_dir), sample_size=30, stratify=True)

    strata = pd.read_csv(stats["audit_strata_csv"])
    assert list(strata.columns) == ["keyword", "pool_rows", "alloc", "weight"]
    assert strata["alloc"].sum() == 30
    assert (strata["weight"] > 0).all()
    assert (strata["weight"].round(4) == (strata["pool_rows"] / strata["alloc"]).round(4)).all()


def test_stratified_handles_skewed_strata_without_shrinking_target(tmp_path: Path):
    """层大小悬殊时，最大余额法仍须凑满目标（小数部分大的层优先补 1）。"""
    path = tmp_path / "pool.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for kw, n in (("大", 5000), ("中", 800), ("小", 40), ("微", 3)):
        rows += [{"video_id": f"{kw}_{i}", "title": kw, "keyword": kw,
                  "duration_seconds": 120} for i in range(n)]
    pd.DataFrame(rows).to_csv(path, index=False)

    stats = build_sample(str(path), str(tmp_path / "out"), sample_size=270, stratify=True)

    assert stats["sample_size"] == 270
    strata = pd.read_csv(stats["audit_strata_csv"])
    assert strata["alloc"].sum() == 270


def test_margin_effective_beats_requested_when_sample_is_manual(tmp_path: Path):
    pool = _pool(tmp_path / "pool.csv", n_per_kw=40)
    stats = build_sample(str(pool), str(tmp_path / "out"), sample_size=10)

    # 手动小样本 → 实际误差大于默认 ±5%，须如实落盘
    assert stats["margin_effective"] > 0.05

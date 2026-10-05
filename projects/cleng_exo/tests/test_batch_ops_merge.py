"""tools/batch_ops/merge_csvs 单测（纯函数层，不走 CLI / 不写项目记录）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_脚本"))

from tools.batch_ops.merge_csvs import (  # noqa: E402
    discover_csvs,
    merge_csvs,
    render_merge_record,
    resolve_stem,
    source_label,
)

COLS = ["video_id", "title", "keyword", "duration_seconds"]


def _write(path: Path, rows: list[tuple]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLS).to_csv(path, index=False)
    return path


@pytest.fixture()
def src(tmp_path: Path) -> Path:
    """两个真 CSV + 两类 macOS 资源叉（须被排除）。"""
    d = tmp_path / "合集"
    _write(d / "a.csv", [
        ("v1", "a1", "咖啡店", 120),
        ("v2", "a2", "理发", 200),
        ("v1", "a1_dup", "咖啡店", 120),   # 同文件内重复
    ])
    _write(d / "b.csv", [
        ("v3", "b1", "美容", 300),
        ("v2", "b2", "理发", 250),          # 跨文件重复，a.csv 优先
    ])
    (d / "._a.csv").write_text("junk", encoding="utf-8")
    (d / "__MACOSX").mkdir()
    (d / "__MACOSX" / "._b.csv").write_text("junk", encoding="utf-8")
    return d


def test_discover_excludes_macos_artifacts(src: Path):
    assert [p.name for p in discover_csvs([src])] == ["a.csv", "b.csv"]


def test_discover_missing_input_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        discover_csvs([tmp_path / "nope"])


def test_discover_dedups_repeated_input(src: Path):
    assert len(discover_csvs([src, src])) == 2


def test_merge_dedups_by_video_id_first_file_wins(src: Path, tmp_path: Path):
    out = tmp_path / "out" / "merged.csv"
    summary = merge_csvs(discover_csvs([src]), out)

    assert summary["total_rows"] == 5
    assert summary["keep"] == 3
    assert summary["removed"] == 2
    assert summary["n_duplicate_keys"] == 2

    df = pd.read_csv(out, dtype=str).set_index("video_id")
    assert sorted(df.index) == ["v1", "v2", "v3"]
    assert df.loc["v1", "title"] == "a1"   # a.csv 优先
    assert df.loc["v2", "title"] == "a2"


def test_merge_is_deterministic(src: Path, tmp_path: Path):
    a = tmp_path / "a" / "m.csv"
    b = tmp_path / "b" / "m.csv"
    merge_csvs(discover_csvs([src]), a)
    merge_csvs(discover_csvs([src]), b)
    assert a.read_bytes() == b.read_bytes()


def test_merge_preserves_source_schema_by_default(src: Path, tmp_path: Path):
    out = tmp_path / "merged.csv"
    merge_csvs(discover_csvs([src]), out)
    assert list(pd.read_csv(out, nrows=0).columns) == COLS


def test_merge_can_keep_source_filename(src: Path, tmp_path: Path):
    out = tmp_path / "merged.csv"
    merge_csvs(discover_csvs([src]), out, keep_source_filename=True)
    assert "filename" in pd.read_csv(out, nrows=0).columns


def test_merge_union_by_name_fills_missing_columns(tmp_path: Path):
    d = tmp_path / "src"
    _write(d / "a.csv", [("v1", "a", "咖啡店", 120)])
    (d / "b.csv").write_text(
        "video_id,title,keyword,duration_seconds,extra\n"
        "v2,b,美容,300,zzz\n",
        encoding="utf-8",
    )
    out = tmp_path / "merged.csv"
    summary = merge_csvs(discover_csvs([d]), out)

    assert "extra" in summary["columns"]
    df = pd.read_csv(out, dtype=str).set_index("video_id")
    assert len(df) == 2
    assert pd.isna(df.loc["v1", "extra"])


def test_merge_removed_pct_and_duration(src: Path, tmp_path: Path):
    summary = merge_csvs(discover_csvs([src]), tmp_path / "m.csv")
    assert summary["removed_pct"] == 40.0
    # 总时长按产物口径（去重后）：v1 120 + v2 200 + v3 300 = 620s
    assert summary["total_duration_hours"] == round(620 / 3600, 1)
    assert summary["total_duration_hours"] != round((620 + 120 + 250) / 3600, 1)


def test_merge_reports_output_column_count(src: Path, tmp_path: Path):
    """回归：记录里的「列数」须是产物实际列数，不能把 filename 辅助列算进去。"""
    out = tmp_path / "merged.csv"
    summary = merge_csvs(discover_csvs([src]), out)
    assert len(summary["columns"]) == len(pd.read_csv(out, nrows=0).columns) == len(COLS)
    assert "filename" not in summary["columns"]


def test_merge_raises_on_missing_dedup_key(tmp_path: Path):
    d = tmp_path / "src"
    d.mkdir()
    pd.DataFrame([{"title": "x"}]).to_csv(d / "a.csv", index=False)
    with pytest.raises(ValueError, match="缺少去重键列"):
        merge_csvs(discover_csvs([d]), tmp_path / "m.csv")


def test_merge_cleans_workfiles(src: Path, tmp_path: Path):
    out_dir = tmp_path / "out"
    merge_csvs(discover_csvs([src]), out_dir / "m.csv")
    assert not (out_dir / ".merge.duckdb").exists()
    assert not (out_dir / ".duckdb_tmp").exists()


def test_merge_per_file_breakdown_sums_to_kept(src: Path, tmp_path: Path):
    summary = merge_csvs(discover_csvs([src]), tmp_path / "m.csv")
    per_file = {r["source_file"]: r for r in summary["per_file"]}

    assert sum(r["rows_in"] for r in summary["per_file"]) == summary["total_rows"]
    assert sum(r["rows_kept"] for r in summary["per_file"]) == summary["keep"]
    assert per_file["a.csv"] == {
        "source_file": "a.csv", "rows_in": 3, "rows_kept": 2, "rows_removed": 1,
    }
    assert per_file["b.csv"]["rows_kept"] == 1


def test_resolve_stem_uses_chinese_category_label(tmp_path: Path):
    out_dir = tmp_path / "raw" / "exo_service"
    assert resolve_stem(out_dir, batch="0813") == "商业服务_merged_0813"


def test_resolve_stem_falls_back_to_source_name(tmp_path: Path):
    out_dir = tmp_path / "work" / "无品类"
    assert resolve_stem(out_dir, batch="0915", fallback="某合集") == "某合集_merged_0915"


def test_resolve_stem_explicit_override(tmp_path: Path):
    assert resolve_stem(tmp_path, stem="自定义", batch="0915") == "自定义"


def test_resolve_stem_no_double_merged(tmp_path: Path):
    """回归：品类与兜底都推断不到时，不应产出 merged_merged_0915。"""
    assert resolve_stem(tmp_path / "work" / "x", batch="0915", fallback="") == "merged_0915"


def test_source_label_dir_and_file(src: Path):
    assert source_label([src]) == "合集"
    assert source_label([src / "a.csv"]) == "合集"


def test_render_merge_record_lists_every_source(src: Path, tmp_path: Path):
    out_dir = tmp_path / "raw" / "exo_service"
    summary = merge_csvs(discover_csvs([src]), out_dir / "商业服务_merged_0813.csv")
    text = render_merge_record(summary, out_dir=out_dir, batch="0813")

    assert "exo_service / 0813" in text
    assert "`a.csv`" in text and "`b.csv`" in text
    assert "商业服务_merged_0813.csv" in text
    assert "._a.csv" not in text          # 资源叉不入记录
    assert "去重后：3" in text

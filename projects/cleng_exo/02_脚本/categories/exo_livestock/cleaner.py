#!/usr/bin/env python3
"""categories/exo_livestock/cleaner.py — exo渔业牧业 文本黑名单（非渔牧主题闸门）。

不默认挂 02_clean；直接调用 categories.exo_livestock.cleaner.clean(...)。
pass2 黑名单 + 可选 rules/blocklist_keywords.csv（精确匹配 keyword，DuckDB）。
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import duckdb

from core.certain_noise_clean import run_clean
from core.sql_builder import sql_escape

_RULES_DIR = Path(__file__).resolve().parent / "rules"
_KW_BLOCK = _RULES_DIR / "blocklist_keywords.csv"


def _load_block_keywords(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            kw = (row.get("keyword") or "").strip().lower()
            if kw:
                out.add(kw)
    return out


def _apply_keyword_block(summary: dict, block: set[str]) -> dict:
    """DuckDB：从 keep 再丢 blocklist keyword；重写 keep/drop。"""
    if not block:
        return summary
    keep_path = Path(summary["keep_path"])
    drop_path = Path(summary["drop_path"])
    if not keep_path.is_file():
        return summary

    tmp_dir = keep_path.parent / ".kw_tmp"
    tmp_dir.mkdir(exist_ok=True)
    work = tmp_dir / "kw.duckdb"
    if work.exists():
        work.unlink()
    db = duckdb.connect(str(work))
    db.execute("SET memory_limit='4GB'")
    db.execute(f"SET temp_directory='{sql_escape(str(tmp_dir))}'")

    keep_esc = sql_escape(str(keep_path.resolve()))
    drop_esc = sql_escape(str(drop_path.resolve()))
    db.execute(
        f"CREATE TABLE keep0 AS SELECT * FROM read_csv_auto('{keep_esc}', header=true, ignore_errors=true, strict_mode=false)"
    )
    db.execute(
        f"CREATE TABLE drop0 AS SELECT * FROM read_csv_auto('{drop_esc}', header=true, ignore_errors=true, strict_mode=false)"
    )

    # normalize keyword for match
    db.execute(
        "CREATE TEMP TABLE blocked AS SELECT * FROM (VALUES "
        + ", ".join(f"('{sql_escape(k)}')" for k in sorted(block))
        + ") AS t(kw)"
    )
    db.execute("""
        CREATE TEMP TABLE kw_drop AS
        SELECT k.*,
               'keyword_blocklist' AS drop_step,
               CAST(k.keyword AS VARCHAR) AS drop_reason
        FROM keep0 k
        WHERE lower(trim(CAST(COALESCE(k.keyword, '') AS VARCHAR))) IN (SELECT kw FROM blocked)
    """)
    n_kw = db.execute("SELECT COUNT(*) FROM kw_drop").fetchone()[0]
    hours_drop = db.execute(
        "SELECT COALESCE(SUM(TRY_CAST(duration_seconds AS DOUBLE)),0)/3600.0 FROM kw_drop"
    ).fetchone()[0]

    new_keep = keep_path.with_name(keep_path.stem + "_kw.csv")
    new_drop = drop_path.with_name(drop_path.stem + "_kw.csv")
    db.execute(f"""
        COPY (
          SELECT * FROM keep0 k
          WHERE lower(trim(CAST(COALESCE(k.keyword, '') AS VARCHAR))) NOT IN (SELECT kw FROM blocked)
        ) TO '{sql_escape(str(new_keep))}' (FORMAT CSV, HEADER true)
    """)
    # union old drops + kw drops (align columns: drop0 may already have drop_step/reason)
    drop0_cols = [c[0] for c in db.execute("SELECT * FROM drop0 LIMIT 0").description]
    if "drop_step" in drop0_cols and "drop_reason" in drop0_cols:
        db.execute(f"""
            COPY (
              SELECT * FROM drop0
              UNION ALL BY NAME
              SELECT * FROM kw_drop
            ) TO '{sql_escape(str(new_drop))}' (FORMAT CSV, HEADER true)
        """)
    else:
        db.execute(f"""
            COPY (
              SELECT d.*, '' AS drop_step, '' AS drop_reason FROM drop0 d
              UNION ALL BY NAME
              SELECT * FROM kw_drop
            ) TO '{sql_escape(str(new_drop))}' (FORMAT CSV, HEADER true)
        """)

    n_keep = db.execute(
        "SELECT COUNT(*) FROM keep0 k WHERE lower(trim(CAST(COALESCE(k.keyword,'') AS VARCHAR))) NOT IN (SELECT kw FROM blocked)"
    ).fetchone()[0]
    db.close()
    try:
        work.unlink(missing_ok=True)
    except OSError:
        pass

    new_keep.replace(keep_path)
    new_drop.replace(drop_path)
    for p in tmp_dir.glob("*"):
        try:
            p.unlink()
        except OSError:
            pass
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    summary["total_keep"] = int(n_keep)
    summary["total_drop"] = int(summary["total_rows"]) - int(n_keep)
    summary["retention_pct"] = round(
        summary["total_keep"] / max(int(summary["total_rows"]), 1) * 100, 1
    )
    summary.setdefault("steps", {})["step3_keyword_blocklist"] = {
        "dropped": int(n_kw),
        "hours_drop": round(float(hours_drop or 0), 1),
        "keywords": sorted(block),
    }
    print(f"  keyword blocklist drop: {int(n_kw):,} ({float(hours_drop or 0):,.1f} h)")
    return summary


def clean(input_path, stem="exo_livestock", output_dir="output", raw_name="", run="run01", **kwargs):
    t0 = time.perf_counter()
    summary = run_clean(
        category="exo_livestock",
        rules_dir=_RULES_DIR,
        input_path=input_path,
        output_dir=output_dir,
        stem=stem,
        raw_name=raw_name,
        run=run,
        **kwargs,
    )
    summary = _apply_keyword_block(summary, _load_block_keywords(_KW_BLOCK))
    summary["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    summary_path = Path(summary["keep_path"]).parent / "clean_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary

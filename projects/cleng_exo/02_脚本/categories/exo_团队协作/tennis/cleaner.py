#!/usr/bin/env python3
"""categories/exo_团队协作/tennis/cleaner.py — 团队协作·双人网球 文本闸门。

口径 v0.2：**团队协作都保留**——网球双打/混双/团体、乒乓/羽球双打、
NBA/足球/排球等团队球类；丢单打、卡通、游戏 MV、无团队信号的絮片。

不默认挂 02_clean；直接调用::

    from categories.exo_团队协作.tennis.cleaner import clean
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import duckdb

from core.certain_noise_clean import run_clean
from core.io import duckdb_reader
from core.sql_builder import sql_escape

_RULES_DIR = Path(__file__).resolve().parent / "rules"

# 双打 / 混双 / 团体赛 / 双人口径
_MULTI_WORD = (
    r"(doubles?|mixed\s*doubles?|davis\s*cup|billie\s*jean\s*king|fed\s*cup|"
    r"双打|混双|双人|团体|bryan\s*brothers|team\s*(tennis|cup|event|match|vs)|"
    r"\bteam\b|pairs?\b|partners?|协作|配合)"
)
# Name/Name vs Name/Name（双打站位）
_SLASH_VS = (
    r"[A-Za-z][A-Za-z''\-]{1,20}/\s*[A-Za-z][A-Za-z''\-]{1,20}.+\bvs?\b.+"
    r"[A-Za-z][A-Za-z''\-]{1,20}/\s*[A-Za-z]"
)
# 天生团队球类（整场即多人协作/对抗）
_TEAM_SPORT = (
    r"(\bnba\b|\bwnba\b|basketball|football|soccer|\bnfl\b|\bnhl\b|"
    r"volleyball|cricket|\bhockey\b|足球|篮球|排球|棒球|"
    r"world\s*table\s*tennis|\bwtt\b|table\s*tennis|ping[\s-]*pong|乒乓球|"
    r"badminton|\bbwf\b|羽毛球)"
)


def _apply_multi_keep(keep_path: str, output_dir: str, base: str) -> dict:
    """certain-noise keep 上再闸：须有多人/团队信号，或本身是团队球类。"""
    date_tag = time.strftime("%m%d")
    multi_drop_path = os.path.join(output_dir, f"{base}_multi_drop_{date_tag}.csv")
    work = Path(output_dir) / ".multi_keep.duckdb"
    if work.exists():
        work.unlink()
    db = duckdb.connect(str(work))
    db.execute(f"CREATE TABLE k AS SELECT * FROM {duckdb_reader(keep_path, ignore_errors=True)}")
    n_in = db.execute("SELECT COUNT(*) FROM k").fetchone()[0]
    # title / keyword 多人信号，或天生团队球类（NBA/乒乓/羽球等）
    db.execute(
        f"""
        CREATE TABLE keep2 AS
        SELECT * FROM k
        WHERE
            regexp_matches(
                lower(COALESCE(title,'') || ' ' || COALESCE(keyword,'')),
                '{sql_escape(_MULTI_WORD)}', 'i'
            )
            OR regexp_matches(COALESCE(title,''), '{sql_escape(_SLASH_VS)}', 'i')
            OR regexp_matches(
                lower(COALESCE(title,'') || ' ' || COALESCE(channel,'') || ' ' || COALESCE(keyword,'')),
                '{sql_escape(_TEAM_SPORT)}', 'i'
            )
        """
    )
    db.execute(
        """
        CREATE TABLE drop2 AS
        SELECT k.*, 'multi_team_gate' AS drop_step, 'no_team_signal' AS drop_reason
        FROM k
        ANTI JOIN keep2 USING (video_id)
        """
    )
    n_keep = db.execute("SELECT COUNT(*) FROM keep2").fetchone()[0]
    n_drop = db.execute("SELECT COUNT(*) FROM drop2").fetchone()[0]
    h_keep = db.execute(
        "SELECT COALESCE(SUM(TRY_CAST(duration_seconds AS DOUBLE)),0)/3600.0 FROM keep2"
    ).fetchone()[0]
    db.execute(f"COPY keep2 TO '{sql_escape(keep_path)}' (FORMAT CSV, HEADER true)")
    db.execute(f"COPY drop2 TO '{sql_escape(multi_drop_path)}' (FORMAT CSV, HEADER true)")
    db.close()
    work.unlink(missing_ok=True)
    return {
        "multi_gate_in": n_in,
        "multi_gate_keep": n_keep,
        "multi_gate_drop": n_drop,
        "multi_gate_keep_hours": round(float(h_keep or 0), 1),
        "multi_drop_path": multi_drop_path,
    }


def clean(input_path, stem="双人网球", output_dir="output", raw_name="", run="run01", **kwargs):
    summary = run_clean(
        category="exo_团队协作",
        rules_dir=_RULES_DIR,
        input_path=input_path,
        output_dir=output_dir,
        stem=stem,
        raw_name=raw_name,
        run=run,
        **kwargs,
    )
    keep_path = summary.get("keep_path") or ""
    base = raw_name if raw_name else (stem or "双人网球")
    if keep_path and os.path.exists(keep_path):
        extra = _apply_multi_keep(keep_path, output_dir, base)
        summary.update(extra)
        summary["final_keep"] = extra["multi_gate_keep"]
        summary["final_keep_hours"] = extra["multi_gate_keep_hours"]
        print(
            f"  multi/team gate: keep {extra['multi_gate_keep']:,} "
            f"/ {extra['multi_gate_keep_hours']}h "
            f"(drop {extra['multi_gate_drop']:,})"
        )
    return summary

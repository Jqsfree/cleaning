#!/usr/bin/env python3
"""exo_service 文本级联 L1 DROP + L2 行业路由。

L1: 仅确定噪声 DROP
L2: 路由到行业 → commercial_service_candidates（允许误报，不判合格）

  PYTHONPATH=02_脚本 python 02_脚本/tools/run_exo_service_cascade_text.py \\
    data/runs/exo_service/machine_0813/01_quality/商业服务_merged_0813_quality_0813.csv \\
    -o data/runs/exo_service/machine_0813/06_tools/cascade_v2/
"""

from __future__ import annotations

import json
import re
import time
import tomllib
from pathlib import Path
from typing import Any

import duckdb

from core.log import log
from core.sql_builder import add_search_text, load_raw_table, sql_escape

_RULES = Path(__file__).resolve().parent / "rules"
_L1 = _RULES / "cascade_l1_drop.toml"
_L2 = _RULES / "cascade_l2_route.toml"


def load_l1(path: Path | None = None) -> list[dict[str, str]]:
    cfg = tomllib.loads((path or _L1).read_text(encoding="utf-8"))
    rows = cfg.get("drop") or []
    if not rows:
        raise ValueError(f"L1 无 drop 规则: {path or _L1}")
    for r in rows:
        re.compile(r["pattern"])
    return [{"category": r["category"], "pattern": r["pattern"]} for r in rows]


def load_l2(path: Path | None = None) -> tuple[list[str], list[dict[str, str]]]:
    cfg = tomllib.loads((path or _L2).read_text(encoding="utf-8"))
    order = list(cfg.get("primary_order") or [])
    routes = cfg.get("route") or []
    if not routes:
        raise ValueError(f"L2 无 route 规则: {path or _L2}")
    for r in routes:
        re.compile(r["pattern"])
        if r["industry"] not in order:
            order.append(r["industry"])
    return order, [{"industry": r["industry"], "pattern": r["pattern"]} for r in routes]


def _union_drop_sql(drops: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for d in drops:
        cat = sql_escape(d["category"])
        pat = sql_escape(d["pattern"])
        parts.append(
            f"SELECT video_id, title_channel, "
            f"'{cat}' AS drop_category, "
            f"regexp_extract(title_channel, '{pat}', 0) AS drop_reason "
            f"FROM raw_text "
            f"WHERE regexp_matches(title_channel, '{pat}', 'i')"
        )
    return " UNION ALL ".join(parts)


def run_l1_l2(
    input_path: str,
    output_dir: str,
    *,
    stem: str = "商业服务_quality",
    l1_path: Path | None = None,
    l2_path: Path | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    drops = load_l1(l1_path)
    primary_order, routes = load_l2(l2_path)

    work = out / ".exo_service_cascade_text.duckdb"
    if work.exists():
        work.unlink()
    db = duckdb.connect(str(work))
    db.execute("SET memory_limit='6GB'")
    db.execute("SET threads=4")
    tmp = out / ".duckdb_tmp"
    tmp.mkdir(exist_ok=True)
    db.execute(f"SET temp_directory='{sql_escape(str(tmp))}'")

    n_total = load_raw_table(db, input_path)
    log(f"L1+L2 input rows: {n_total:,}")
    add_search_text(db)

    # --- L1 ---
    db.execute(f"CREATE TEMP TABLE l1_hits AS {_union_drop_sql(drops)}")
    db.execute("""
        CREATE TEMP TABLE l1_drop AS
        SELECT video_id,
               arg_min(drop_category, drop_category) AS drop_category,
               arg_min(drop_reason, drop_reason) AS drop_reason
        FROM l1_hits
        GROUP BY video_id
    """)
    n_l1 = db.execute("SELECT COUNT(*) FROM l1_drop").fetchone()[0]
    log(f"  L1 drop: {n_l1:,}")

    db.execute("""
        CREATE TEMP TABLE after_l1 AS
        SELECT r.video_id, r.title_channel
        FROM raw_text r
        ANTI JOIN l1_drop d USING (video_id)
    """)
    n_after = db.execute("SELECT COUNT(*) FROM after_l1").fetchone()[0]
    log(f"  after L1: {n_after:,}")

    # --- L2 flags（仅 id+文本，避免把全量宽表复制三遍）---
    flag_exprs: list[str] = []
    for r in routes:
        ind = r["industry"]
        pat = sql_escape(r["pattern"])
        flag_exprs.append(
            f"CASE WHEN regexp_matches(title_channel, '{pat}', 'i') "
            f"THEN true ELSE false END AS ind_{ind}"
        )
    db.execute(
        f"CREATE TEMP TABLE l2_flags AS SELECT video_id, "
        + ", ".join(flag_exprs)
        + " FROM after_l1"
    )

    any_parts = " OR ".join(f"ind_{r['industry']}" for r in routes)
    # primary: first in primary_order that is true
    primary_case = "CASE "
    for ind in primary_order:
        primary_case += f"WHEN ind_{ind} THEN '{ind}' "
    primary_case += "ELSE NULL END"

    industries_concat = " || ".join(
        f"CASE WHEN ind_{r['industry']} THEN '{r['industry']},' ELSE '' END" for r in routes
    )

    db.execute(f"""
        CREATE TEMP TABLE l2_tagged AS
        SELECT *,
               ({any_parts}) AS is_candidate,
               {primary_case} AS industry_primary,
               regexp_replace({industries_concat}, ',$', '') AS industries
        FROM l2_flags
    """)

    n_cand = db.execute(
        "SELECT COUNT(*) FROM l2_tagged WHERE is_candidate"
    ).fetchone()[0]
    n_unrouted = n_after - n_cand
    log(f"  L2 candidates: {n_cand:,} | unrouted: {n_unrouted:,}")

    date_tag = time.strftime("%m%d")
    keep_csv = out / f"{stem}_commercial_service_candidates_{date_tag}.csv"
    drop_csv = out / f"{stem}_l1_drop_{date_tag}.csv"
    unrouted_csv = out / f"{stem}_l2_unrouted_{date_tag}.csv"

    # join back full rows for outputs
    db.execute(f"""
        COPY (
          SELECT r.* EXCLUDE (search_text, title_channel),
                 d.drop_category AS l1_drop_category,
                 d.drop_reason AS l1_drop_reason
          FROM raw_text r
          JOIN l1_drop d USING (video_id)
        ) TO '{sql_escape(str(drop_csv))}' (FORMAT CSV, HEADER true)
    """)
    db.execute(f"""
        COPY (
          SELECT r.* EXCLUDE (search_text, title_channel),
                 t.industry_primary,
                 t.industries
          FROM raw_text r
          JOIN l2_tagged t USING (video_id)
          WHERE t.is_candidate
        ) TO '{sql_escape(str(keep_csv))}' (FORMAT CSV, HEADER true)
    """)
    db.execute(f"""
        COPY (
          SELECT r.* EXCLUDE (search_text, title_channel)
          FROM raw_text r
          JOIN l2_tagged t USING (video_id)
          WHERE NOT t.is_candidate
        ) TO '{sql_escape(str(unrouted_csv))}' (FORMAT CSV, HEADER true)
    """)

    by_l1 = {
        str(r[0]): int(r[1])
        for r in db.execute(
            "SELECT drop_category, COUNT(*) FROM l1_drop GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
    }
    by_ind = {
        str(r[0]): int(r[1])
        for r in db.execute(
            "SELECT industry_primary, COUNT(*) FROM l2_tagged "
            "WHERE is_candidate GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
    }

    summary = {
        "layer": "l1_l2",
        "input": input_path,
        "n_input": n_total,
        "n_l1_drop": n_l1,
        "n_after_l1": n_after,
        "n_candidates": n_cand,
        "n_unrouted": n_unrouted,
        "l1_by_category": by_l1,
        "l2_by_primary": by_ind,
        "candidates_csv": str(keep_csv),
        "l1_drop_csv": str(drop_csv),
        "unrouted_csv": str(unrouted_csv),
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "notes": [
            "candidates = commercial_service_candidates; not deliver",
            "unrouted is not L1 certain-noise",
            "L3 CLIP / L4 subcategory VLM next",
        ],
    }
    sum_path = out / "cascade_l1_l2_summary.json"
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  wrote {keep_csv.name} / {drop_csv.name} / {unrouted_csv.name}")
    log(f"  summary → {sum_path}")
    db.close()
    return summary


def classify_title_l1_l2(title: str, channel: str = "") -> dict[str, Any]:
    """单条调试：Python re，与 TOML 一致。"""
    text = f"{title or ''} {channel or ''}".strip()
    for d in load_l1():
        if re.search(d["pattern"], text, flags=re.I):
            return {"stage": "l1_drop", "drop_category": d["category"]}
    order, routes = load_l2()
    hit = [r["industry"] for r in routes if re.search(r["pattern"], text, flags=re.I)]
    # preserve primary_order
    hit_ordered = [i for i in order if i in hit]
    if not hit_ordered:
        return {"stage": "unrouted", "industries": []}
    return {
        "stage": "candidate",
        "industry_primary": hit_ordered[0],
        "industries": hit_ordered,
    }

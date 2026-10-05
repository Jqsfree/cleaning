#!/usr/bin/env python3
"""通用 certain-noise 文本黑名单清洗（channel_pass2 + title+channel pass2 / r2）。

百万行用落盘 DuckDB，避免 :memory: COPY 被 OOM/SIGTERM。
``channel_pass2`` 仅匹配 ``channel`` 列（金标纯 F 频道等）；未配置时跳过。

闸门层级（可选 ``[[rescue]]`` 豁免）::

    channel_pass2 / keyword_pass2  硬闸：整频道/整 playlist 拉黑，不可豁免
    r2                             硬闸：标题级、不可豁免（如游戏实况）
    pass2                          软闸：标题级；命中 [[rescue]] 的标题不参与本闸

``rescue`` 语义为「有本品类强信号就不算串台」，用于放行被通用规则误伤的品类内标题
（如 exo_dance 里被通用 workout 规则误伤的 ``FULL BODY DANCE WORKOUT``）；未配置时等价于 ``\\b\\B``（不豁免任何行）。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import duckdb

from core.log import log
from core.rules_loader import load_blacklist, compute_and_save_rule_stats
from core.sql_builder import (
    add_search_text,
    load_raw_table,
    sql_escape,
    write_summary_json,
)

_NEVER = r"\b\B"


def run_clean(
    *,
    category: str,
    rules_dir: Path,
    input_path: str,
    output_dir: str,
    stem: str = "",
    raw_name: str = "",
    run: str = "run01",
    **kwargs,
) -> dict:
    t0 = time.perf_counter()
    log(f"{category} 文本黑名单清洗（非农主题闸门 / certain-noise）...")

    os.makedirs(output_dir, exist_ok=True)
    rules = load_blacklist(rules_dir)
    channel_re = sql_escape(rules.get("channel_pass2") or _NEVER)
    keyword_re = sql_escape(rules.get("keyword_pass2") or _NEVER)
    pass2_re = sql_escape(rules["pass2"])
    r2_re = sql_escape(rules["r2"])
    rescue_re = sql_escape(rules.get("rescue") or _NEVER)
    has_channel = (rules.get("channel_pass2") or _NEVER) != _NEVER
    has_keyword = (rules.get("keyword_pass2") or _NEVER) != _NEVER
    has_rescue = (rules.get("rescue") or _NEVER) != _NEVER

    work = Path(output_dir) / f".{category}_clean.duckdb"
    if work.exists():
        work.unlink()
    db = duckdb.connect(str(work))
    mem = str(kwargs.get("memory_limit") or os.environ.get("DUCKDB_MEMORY_LIMIT") or "6GB")
    db.execute(f"SET memory_limit='{sql_escape(mem)}'")
    db.execute("SET threads=2")
    db.execute("SET preserve_insertion_order=false")
    tmp = Path(output_dir) / ".duckdb_tmp"
    tmp.mkdir(exist_ok=True)
    db.execute(f"SET temp_directory='{sql_escape(str(tmp))}'")
    n_total = load_raw_table(db, input_path)
    log(f"  rows: {n_total:,}")
    add_search_text(db)

    n_ch = 0
    if has_channel:
        db.execute(f"""
            CREATE TEMP TABLE step0_channel AS
            SELECT video_id, COALESCE(channel, '') AS title_channel,
                   'channel_blacklist' AS drop_step,
                   regexp_extract(COALESCE(channel, ''), '{channel_re}', 0) AS drop_reason
            FROM raw_text
            WHERE regexp_matches(COALESCE(channel, ''), '{channel_re}', 'i')
        """)
        n_ch = db.execute("SELECT COUNT(*) FROM step0_channel").fetchone()[0]
        log(f"  channel_pass2 drop: {n_ch:,}")
    else:
        db.execute("""
            CREATE TEMP TABLE step0_channel AS
            SELECT video_id, '' AS title_channel, '' AS drop_step, '' AS drop_reason
            FROM raw_text WHERE 1=0
        """)

    n_kw = 0
    if has_keyword:
        db.execute(f"""
            CREATE TEMP TABLE step0b_keyword AS
            SELECT r.video_id, COALESCE(r.keyword, '') AS title_channel,
                   'keyword_blacklist' AS drop_step,
                   regexp_extract(COALESCE(r.keyword, ''), '{keyword_re}', 0) AS drop_reason
            FROM raw_text r
            ANTI JOIN step0_channel c USING (video_id)
            WHERE regexp_matches(COALESCE(r.keyword, ''), '{keyword_re}', 'i')
        """)
        n_kw = db.execute("SELECT COUNT(*) FROM step0b_keyword").fetchone()[0]
        log(f"  keyword_pass2 drop: {n_kw:,}")
    else:
        db.execute("""
            CREATE TEMP TABLE step0b_keyword AS
            SELECT video_id, '' AS title_channel, '' AS drop_step, '' AS drop_reason
            FROM raw_text WHERE 1=0
        """)

    # pass2 为软闸：命中 [[rescue]] 的标题豁免（channel/keyword/r2 硬闸不受 rescue 影响）
    db.execute(f"""
        CREATE TEMP TABLE step1_soft AS
        SELECT r.video_id, r.title_channel,
               regexp_extract(r.title_channel, '{pass2_re}', 0) AS drop_reason,
               regexp_matches(r.title_channel, '{rescue_re}', 'i') AS rescued
        FROM raw_text r
        ANTI JOIN step0_channel c USING (video_id)
        ANTI JOIN step0b_keyword k USING (video_id)
        WHERE regexp_matches(r.title_channel, '{pass2_re}', 'i')
    """)
    n_resc = db.execute("SELECT COUNT(*) FROM step1_soft WHERE rescued").fetchone()[0]
    db.execute("""
        CREATE TEMP TABLE step1 AS
        SELECT video_id, title_channel, 'step1_blacklist' AS drop_step, drop_reason
        FROM step1_soft WHERE NOT rescued
    """)
    n_bl = db.execute("SELECT COUNT(*) FROM step1").fetchone()[0]
    log(f"  pass2 drop: {n_bl:,}" + (f" | rescue 放行: {n_resc:,}" if has_rescue else ""))

    db.execute(f"""
        CREATE TEMP TABLE step1b_r2 AS
        SELECT r.video_id, r.title_channel,
               'step1b_r2' AS drop_step,
               regexp_extract(r.title_channel, '{r2_re}', 0) AS drop_reason
        FROM raw_text r
        ANTI JOIN step0_channel c USING (video_id)
        ANTI JOIN step0b_keyword k USING (video_id)
        ANTI JOIN step1 s USING (video_id)
        WHERE regexp_matches(r.title_channel, '{r2_re}', 'i')
    """)
    n_r2 = db.execute("SELECT COUNT(*) FROM step1b_r2").fetchone()[0]
    log(f"  r2 drop:   {n_r2:,}")

    db.execute("""
        CREATE TEMP TABLE drop_ids AS
        SELECT video_id, drop_step, drop_reason FROM step0_channel
        UNION ALL
        SELECT video_id, drop_step, drop_reason FROM step0b_keyword
        UNION ALL
        SELECT video_id, drop_step, drop_reason FROM step1
        UNION ALL
        SELECT video_id, drop_step, drop_reason FROM step1b_r2
    """)
    n_drop = db.execute("SELECT COUNT(*) FROM drop_ids").fetchone()[0]
    n_keep = n_total - n_drop
    log(f"  keep: {n_keep:,} | drop: {n_drop:,}")

    section_map = {"pass2": "step1", "r2": "step1b_r2"}
    if has_channel:
        section_map = {"channel_pass2": "step0_channel", **section_map}
    if has_keyword:
        section_map = {"keyword_pass2": "step0b_keyword", **section_map}
    # channel/keyword stats use their own text col; pass2 uses title_channel
    rule_stats: dict = {}
    for section, table in section_map.items():
        rule_stats.update(
            compute_and_save_rule_stats(
                db, rules_dir,
                section_table_map={section: table},
                text_col="title_channel",
            )
        )

    base = raw_name if raw_name else (stem or category)
    date_tag = time.strftime("%m%d")
    out_keep = os.path.join(output_dir, f"{base}_clean_{date_tag}.csv")
    out_drop = os.path.join(output_dir, f"{base}_clean_drop_{date_tag}.csv")

    db.execute(
        f"COPY ("
        f"  SELECT r.* EXCLUDE (search_text, title_channel) FROM raw_text r"
        f"  ANTI JOIN drop_ids d USING (video_id)"
        f") TO '{sql_escape(out_keep)}' (FORMAT CSV, HEADER true)"
    )
    db.execute(
        f"COPY ("
        f"  SELECT r.* EXCLUDE (search_text, title_channel), d.drop_step, d.drop_reason"
        f"  FROM raw_text r JOIN drop_ids d USING (video_id)"
        f") TO '{sql_escape(out_drop)}' (FORMAT CSV, HEADER true)"
    )

    elapsed = time.perf_counter() - t0
    summary = {
        "engine": f"{category}-blacklist",
        "category": category,
        "input": os.path.abspath(input_path),
        "total_rows": n_total,
        "total_keep": n_keep,
        "total_drop": n_drop,
        "retention_pct": round(n_keep / max(n_total, 1) * 100, 1),
        "elapsed_sec": round(elapsed, 1),
        "steps": {
            "step0_channel": {"dropped": n_ch},
            "step0b_keyword": {"dropped": n_kw},
            "step1_pass2": {"dropped": n_bl, "rescued": n_resc},
            "step2_r2": {"dropped": n_r2},
        },
        "rule_stats": rule_stats,
        "keep_path": out_keep,
        "drop_path": out_drop,
        "run": run,
    }
    write_summary_json(output_dir, summary)
    db.close()
    try:
        work.unlink(missing_ok=True)
        for p in tmp.glob("*"):
            p.unlink(missing_ok=True)
        tmp.rmdir()
    except OSError:
        pass

    print(f"\n  总行数: {n_total:>12,}")
    print(f"  保留:   {n_keep:>12,} ({summary['retention_pct']}%)")
    print(f"  移除:   {n_drop:>12,}")
    print(f"  耗时:   {elapsed:>11.1f}s")
    print(f"  产物:   {output_dir}/")
    return summary

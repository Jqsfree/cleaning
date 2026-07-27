#!/usr/bin/env python3
"""
core/playlist.py -- Pass 1: DuckDB playlist 命中率分析（纯 SQL，无 UDF）

用 DuckDB SQL 计算每个 playlist 的体育命中率。
Pass 1 不需要精确评分，用快速启发式检查即可。
"""

import time
from pathlib import Path
import duckdb


# Pass 1 快速启发式正则 — 用于 playlist 命中率估算。
# 这不对应 TOML 规则；它是一个轻量级检查，旨在以低成本标记
# 体育内容占比低的 playlist。模式在 entities.toml 和
# whitelist.toml 中有镜像，但不共享代码 — 使两者保持同步。
_PASS1_FAST_PATTERN = (
    r'match|game|race|final|championship|tournament|league|cup|vs|versus|'
    r'highlights|live|stream|broadcast|replay|coverage|commentary|'
    r'nba|nfl|mlb|nhl|fifa|atp|wta|ufc|ncaa|olympic|world.cup|grand.prix'
)


def analyze(input_path: str, min_samples: int = 5, min_hit_rate: float = 0.10,
            chunksize: int = 0) -> set:
    t0 = time.perf_counter()
    print("Pass 1: analyzing playlist hit rates (DuckDB, pure SQL)...")

    ext = Path(input_path).suffix.lower()
    reader = "read_parquet" if ext == ".parquet" else "read_csv_auto"
    reader_opts = (f"('{input_path}')" if ext == ".parquet"
                   else f"('{input_path}', header=true, all_varchar=true, sample_size=-1, ignore_errors=true)")

    db = duckdb.connect(":memory:")

    # 纯 SQL 启发式：标题含赛事关键词 或 keyword 在标题中出现
    # 参见模块文档字符串 _PASS1_FAST_PATTERN。
    db.execute(f"""
        CREATE TEMP TABLE playlist_final AS
        SELECT source_ref,
               COUNT(*) AS total,
               SUM(CASE
                   WHEN regexp_matches(lower(title), '{_PASS1_FAST_PATTERN}')
                   THEN 1
                   WHEN lower(keyword) != '' AND lower(title) LIKE '%' || lower(keyword) || '%'
                   THEN 1
                   ELSE 0
               END) AS hits
        FROM {reader}{reader_opts}
        WHERE source_ref IS NOT NULL AND source_ref != ''
        GROUP BY source_ref
    """)

    try:
        result = db.execute(f"""
            SELECT source_ref FROM playlist_final
            WHERE total >= {min_samples}
              AND CAST(hits AS FLOAT) / total < {min_hit_rate}
        """).fetchall()
    except duckdb.Error as e:
        print(f"  [WARN] playlist analysis failed: {e}")
        db.close()
        return set()

    polluted = {row[0] for row in result}

    elapsed = time.perf_counter() - t0
    total_pl = db.execute("SELECT COUNT(*) FROM playlist_final").fetchone()[0]
    print(f"  playlists: {total_pl:,}, polluted: {len(polluted):,} ({elapsed:.1f}s)")

    db.close()
    return polluted

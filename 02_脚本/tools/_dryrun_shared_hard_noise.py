#!/usr/bin/env python3
"""共享硬噪声 dry-run（不含 vlog）。只统计，不写 keep。"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb

ROOT = Path("/home/jqs/projects/clean_DATASET")

TABLES = [
    ("exo_service", ROOT / "data/runs/exo_service/machine_0813/05_clean/run05/商业服务_merged_clean_0814.csv"),
    ("exo_agriculture", ROOT / "data/runs/exo_agriculture/machine_0814/01_quality/农业采集_quality_0814.csv"),
    ("parent_child_lt50", ROOT / "data/runs/parent_child/machine_0814_lt50/01_quality/亲子互动_<50%_quality_0814.csv"),
    ("parent_child_ge50", ROOT / "data/runs/parent_child/machine_0814_ge50/01_quality/亲子互动_≥50%_quality_0814.csv"),
    ("exo_fitness", ROOT / "data/runs/exo_fitness/machine_0814/01_quality/健身训练_quality_0814.csv"),
    ("exo_livestock", ROOT / "data/runs/exo_livestock/machine_0814/01_quality/渔业牧业_quality_0814.csv"),
    ("exo_factory", ROOT / "data/runs/exo_factory/machine_0814/01_quality/工厂生产_quality_0814.csv"),
    ("unbox_lt50", ROOT / "data/runs/unbox/machine_0814_lt50/01_quality/商品开箱_<50%_quality_0814.csv"),
    ("unbox_ge50", ROOT / "data/runs/unbox/machine_0814_ge50/01_quality/商品开箱_≥50%_quality_0814.csv"),
    ("exo_medical", ROOT / "data/runs/exo_medical/machine_0813/05_clean/run03/exo医疗场景_e15c3ad7_records_clean_0814.csv"),
]

# 跨 PDF 都不要：非真人 / 成片 MV / 硬新闻 / 儿歌成片
PATS = {
    "non_real_game": r"(?i)(\bgameplay\b|lets?\s*play|\bminecraft\b|\bfortnite\b|\broblox\b|\bgta\s*[45v]\b)",
    "non_real_anime": r"(?i)(anime\s*episode|cartoon\s*episode|3d\s*animation|\banime\b|\bcartoon\b)",
    "non_real_ai": r"(?i)(ai\s*generated|#ai\b|\bvtuber\b|virtual\s*youtuber|midjourney|\bsora\b|deepfake)",
    "music_mv": r"(?i)(official\s*(music\s*)?(video|audio|mv)|lyrics\s*video|music\s*video)",
    "news_hard": r"(?i)(\bbbc\s*news\b|\babc\s*news\b|\bctv\s*news\b|\bndtv\b|breaking\s*news|eyewitness\s*news)",
    "kids_show": r"(?i)(\bcocomelon\b|peppa\s*pig|nursery\s*rhymes?|alphabet\s*song|mother\s*goose\s*club)",
}


def main() -> None:
    con = duckdb.connect()
    print(f"{'batch':22} {'n':>10} {'any':>8} {'pct':>6}  " + " ".join(f"{k:>14}" for k in PATS))
    for name, path in TABLES:
        if not path.exists():
            print(f"{name:22} MISSING {path}")
            continue
        esc = str(path).replace("'", "''")
        con.execute(
            f"""
            CREATE OR REPLACE TABLE t AS
            SELECT
              coalesce(title,'') AS title,
              coalesce(channel,'') AS channel,
              coalesce(title,'') || ' ' || coalesce(channel,'') AS tc,
              try_cast(duration_seconds AS DOUBLE) AS dur
            FROM read_csv_auto('{esc}', header=true, ignore_errors=true, sample_size=20000)
            """
        )
        n = con.execute("SELECT count(*) FROM t").fetchone()[0]
        any_re = "|".join(f"(?:{p})" for p in PATS.values())
        any_n, any_h = con.execute(
            f"""
            SELECT count(*),
                   sum(CASE WHEN dur BETWEEN 1 AND 86400 THEN dur ELSE 0 END)/3600.0
            FROM t WHERE regexp_matches(tc, '{any_re.replace("'", "''")}')
            """
        ).fetchone()
        cols = []
        for key, pat in PATS.items():
            c = con.execute(
                f"SELECT count(*) FROM t WHERE regexp_matches(tc, '{pat.replace(chr(39), chr(39)+chr(39))}')"
            ).fetchone()[0]
            cols.append(f"{c:14,}")
        pct = 100.0 * any_n / n if n else 0
        print(f"{name:22} {n:10,} {any_n:8,} {pct:5.2f}%  " + " ".join(cols))
        print(f"{'':22} {'hours':>10} {any_h or 0:8.0f}")


if __name__ == "__main__":
    main()

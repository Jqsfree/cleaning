#!/usr/bin/env python3
"""
phase5_clean_beauty.py — 美妆规则清洗
使用 rules/beauty/ 下的规则，纯 SQL 过滤（pass2 + r2 → keep/drop）

用法:
  python3 phase5_clean_beauty.py data/runs/beauty/001_baseline/xxx_raw.parquet -o data/runs/beauty/005_clean/run01
"""

import sys, os, time, json, argparse, tomllib
from pathlib import Path
import duckdb

RULES_DIR = Path(__file__).resolve().parent / "rules" / "beauty"


def _load_regex_list(section: str) -> str:
    bl_path = RULES_DIR / "blacklist.toml"
    if not bl_path.exists():
        return r"(?!x)x"
    bl = tomllib.loads(bl_path.read_text("utf-8"))
    patterns = [item["pattern"] for item in bl.get(section, [])]
    return "|".join(patterns) if patterns else r"(?!x)x"


def _load_strong_pattern() -> str:
    wl_path = RULES_DIR / "whitelist.toml"
    if not wl_path.exists():
        return r"(?!x)x"
    wl = tomllib.loads(wl_path.read_text("utf-8"))
    return wl.get("strong_beauty_title_pattern", r"(?!x)x")


def _sql_escape(s: str) -> str:
    return s.replace("'", "''")


def main():
    parser = argparse.ArgumentParser(description="美妆规则清洗")
    parser.add_argument("input", help="baseline parquet 文件")
    parser.add_argument("-o", "--output-dir", default="data/runs/beauty/005_clean/run01")
    parser.add_argument("-r", "--run", default="run01")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    raw_stem = os.path.splitext(os.path.basename(args.input))[0]
    stem = raw_stem.replace("_raw", "") if "_raw" in raw_stem else raw_stem
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.perf_counter()
    print("美妆规则清洗 (pass2 + r2 黑名单)...")

    pass2_re = _sql_escape(_load_regex_list("pass2"))
    r2_re = _sql_escape(_load_regex_list("r2"))
    strong_re = _sql_escape(_load_strong_pattern())

    db = duckdb.connect(":memory:")

    # Step 0: load
    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".parquet":
        reader = f"read_parquet('{args.input}')"
    else:
        reader = f"read_csv_auto('{args.input}', header=true, all_varchar=true, sample_size=-1, ignore_errors=true)"

    db.execute(f"CREATE TEMP TABLE raw AS SELECT * FROM {reader}")
    n_total = db.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
    print(f"  rows: {n_total:,}")

    # Build search text (keyword -neg tags)
    db.execute("""
        CREATE TEMP TABLE raw_text AS
        SELECT *,
               COALESCE(title,'') || ' ' || COALESCE(channel,'') || ' ' ||
               regexp_replace(COALESCE(keyword,''), '((^|\\s)-[a-zA-Z0-9*?]+)+$', '') AS search_text,
               COALESCE(title,'') || ' ' || COALESCE(channel,'') AS title_channel
        FROM raw
    """)

    # Step 1: pass2 blacklist
    db.execute(f"""
        CREATE TEMP TABLE step1 AS
        SELECT *, 'step1_blacklist' AS drop_step,
               regexp_extract(search_text, '{pass2_re}') AS drop_reason
        FROM raw_text
        WHERE regexp_matches(search_text, '{pass2_re}', 'i')
    """)
    n_bl = db.execute("SELECT COUNT(*) FROM step1").fetchone()[0]
    print(f"  pass2 drop: {n_bl:,}")

    db.execute(f"""
        CREATE TEMP TABLE after_bl AS
        SELECT * FROM raw_text
        WHERE NOT regexp_matches(search_text, '{pass2_re}', 'i')
    """)

    # Step 2: r2 blacklist
    db.execute(f"""
        CREATE TEMP TABLE step2 AS
        SELECT *, 'step2_r2' AS drop_step,
               regexp_extract(search_text, '{r2_re}') AS drop_reason
        FROM after_bl
        WHERE regexp_matches(search_text, '{r2_re}', 'i')
    """)
    n_r2 = db.execute("SELECT COUNT(*) FROM step2").fetchone()[0]
    print(f"  r2 drop:   {n_r2:,}")

    # Survivors
    db.execute(f"""
        CREATE TEMP TABLE after_r2 AS
        SELECT * FROM after_bl
        WHERE NOT regexp_matches(search_text, '{r2_re}', 'i')
    """)
    n_surviving = db.execute("SELECT COUNT(*) FROM after_r2").fetchone()[0]
    print(f"  幸存:     {n_surviving:,}")

    # All dropped = pass2 + r2
    db.execute("""
        CREATE TEMP TABLE dropped AS
        SELECT * EXCLUDE (search_text, title_channel, drop_step, drop_reason),
               drop_step, drop_reason
        FROM step1
        UNION ALL
        SELECT * EXCLUDE (search_text, title_channel, drop_step, drop_reason),
               drop_step, drop_reason
        FROM step2
    """)
    n_drop = db.execute("SELECT COUNT(*) FROM dropped").fetchone()[0]

    # Output
    base = stem
    out_keep = os.path.join(output_dir, f"{base}_{args.run}_keep.parquet")
    out_drop = os.path.join(output_dir, f"{base}_{args.run}_drop.parquet")

    # Keep: strip aux columns
    all_cols = [c[0] for c in db.execute("SELECT * FROM dropped LIMIT 0").description]
    aux = {"search_text", "title_channel", "drop_step", "drop_reason"}
    keep_cols = [c for c in all_cols if c not in aux]
    keep_col_str = ", ".join(f'"{c}"' for c in keep_cols)

    db.execute(f"COPY (SELECT {keep_col_str} FROM after_r2) TO '{out_keep}' (FORMAT PARQUET)")

    # Drop: keep drop_step + drop_reason
    drop_aux = {"search_text", "title_channel"}
    drop_cols = [c for c in all_cols if c not in drop_aux]
    drop_col_str = ", ".join(f'"{c}"' for c in drop_cols)
    db.execute(f"COPY (SELECT {drop_col_str} FROM dropped) TO '{out_drop}' (FORMAT PARQUET)")

    elapsed = time.perf_counter() - t0

    n_keep = n_surviving
    total_drop = n_drop
    print(f"  keep: {n_keep:,} | drop: {total_drop:,} ({total_drop/max(n_total,1)*100:.1f}%)")

    # Summary
    summary = {
        "engine": "beauty-blacklist",
        "input": os.path.abspath(args.input),
        "total_rows": n_total,
        "total_keep": n_keep,
        "total_drop": total_drop,
        "retention_pct": round(n_keep / max(n_total, 1) * 100, 1),
        "elapsed_sec": round(elapsed, 1),
        "steps": {
            "step1_pass2_blacklist": {"dropped": n_bl},
            "step2_r2_blacklist": {"dropped": n_r2},
        },
    }

    with open(os.path.join(output_dir, "clean_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  output: {output_dir}/ ({elapsed:.1f}s)")
    print()
    print(f"  keep:  {out_keep}")
    print(f"  drop:  {out_drop}")
    db.close()


if __name__ == "__main__":
    main()

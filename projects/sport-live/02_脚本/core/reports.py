#!/usr/bin/env python3
"""
core/reports.py — 管道报告生成

从清洗结果生成 pipeline_report.md + rule_hits.csv + entity_hits.csv +
keep_sample.parquet + drop_sample.parquet。
"""

import os, time
from pathlib import Path
import duckdb


def generate(input_path: str, output_dir: str, stem: str = "clean",
             summary: dict = None, fmt: str = "parquet", sample_n: int = 200,
             raw_name: str = "", run: str = "run01"):
    """生成完整报告和审计样本。"""
    os.makedirs(output_dir, exist_ok=True)

    db = duckdb.connect(":memory:")

    # 读取 all 和 dropped 表
    base = raw_name if raw_name else stem
    all_path = os.path.join(output_dir, f"{base}_{run}_keep.{fmt}")
    drop_path = os.path.join(output_dir, f"{base}_{run}_drop.{fmt}")

    # 审计样本 — 仅在文件存在且有数据时生成
    sample_path = os.path.join(output_dir, f"{base}_{run}_keep_sample.parquet")
    try:
        n_keep = db.execute(
            f"SELECT COUNT(*) FROM read_parquet('{all_path}')"
        ).fetchone()[0] if os.path.exists(all_path) else 0
        if n_keep > 0:
            actual_n = min(sample_n, n_keep)
            db.execute(f"""
                CREATE TEMP TABLE keep_data AS
                SELECT * FROM read_parquet('{all_path}')
                USING SAMPLE {actual_n} ROWS
            """)
            db.execute(f"COPY keep_data TO '{sample_path}' (FORMAT PARQUET)")
            print(f"  keep_sample: {actual_n} rows")
        else:
            print(f"  keep_sample: skipped (keep_all is empty)")
    except (OSError, IOError, duckdb.IOException) as e:
        print(f"  keep_sample: skipped (I/O error: {e})")
    except duckdb.Error as e:
        print(f"  keep_sample: skipped (DuckDB error: {e})")

    sample_path = os.path.join(output_dir, f"{base}_{run}_drop_sample.parquet")
    try:
        n_drop = db.execute(
            f"SELECT COUNT(*) FROM read_parquet('{drop_path}')"
        ).fetchone()[0] if os.path.exists(drop_path) else 0
        if n_drop > 0:
            actual_n = min(sample_n, n_drop)
            db.execute(f"""
                CREATE TEMP TABLE drop_data AS
                SELECT * FROM read_parquet('{drop_path}')
                USING SAMPLE {actual_n} ROWS
            """)
            db.execute(f"COPY drop_data TO '{sample_path}' (FORMAT PARQUET)")
            print(f"  drop_sample: {actual_n} rows")
        else:
            print(f"  drop_sample: skipped (clean_dropped is empty)")
    except (OSError, IOError, duckdb.IOException) as e:
        print(f"  drop_sample: skipped (I/O error: {e})")
    except duckdb.Error as e:
        print(f"  drop_sample: skipped (DuckDB error: {e})")

    db.close()

    # Markdown 报告
    total = summary.get("total_rows", 0)
    keep = summary.get("total_keep", 0)
    drop = summary.get("total_drop", 0)
    high = summary.get("total_keep_high", 0)
    med = summary.get("total_keep_medium", 0)
    pct = keep / max(total, 1) * 100

    lines = [
        f"# Pipeline Report — {stem}",
        f"",
        f"**Engine:** DuckDB + Parquet",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Overview",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total input | {total:,} |",
        f"| Keep | {keep:,} ({pct:.1f}%) |",
        f"| — high | {high:,} |",
        f"| — medium | {med:,} |",
        f"| Drop | {drop:,} |",
        f"",
    ]

    steps = summary.get("steps", {})
    if steps:
        lines.append("## Drop Breakdown")
        lines.append("")
        lines.append("| Step | Count |")
        lines.append("|------|-------|")
        for step, info in sorted(steps.items()):
            if isinstance(info, dict):
                lines.append(f"| {step} | {info.get('dropped', 0):,} |")
        lines.append("")

    report_path = os.path.join(output_dir, "pipeline_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  report: {report_path}")

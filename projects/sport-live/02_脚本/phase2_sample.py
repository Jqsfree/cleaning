#!/usr/bin/env python3
"""
phase2_sample.py -- SOP Phase 2: 随机抽样质检

从 baseline.parquet 随机抽取样本供标注。

输出:
  {output_dir}/audit_sample_v1.parquet
  {output_dir}/audit_stats.csv

用法:
  python3 phase2_sample.py runs/001_baseline/baseline.parquet -o runs/002_audit/
  python3 phase2_sample.py baseline.parquet -o runs/002_audit/ -n 1000 --seed 42
"""

import sys, os, time, argparse, textwrap, csv
from pathlib import Path
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.sop import load_sop, print_banner, write_run_log
from core.progress import update, mark_done


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    sop_text = load_sop()
    if sop_text:
        print(sop_text[:600])
        print("...\n")
    print_banner(2)

    parser = argparse.ArgumentParser(
        description="SOP Phase 2: 随机抽样质检",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例:
              python3 phase2_sample.py runs/001_baseline/baseline.parquet -o runs/002_audit/
        """),
    )
    parser.add_argument("input", help="baseline.parquet")
    parser.add_argument("-o", "--output-dir", default="data/runs/002_audit",
                        help="输出目录 (默认: runs/002_audit)")
    parser.add_argument("-n", "--sample-size", type=int, default=1000,
                        help="抽样数量 (默认: 1000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-p", "--progress", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    out_dir = args.output_dir.rstrip("/")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    con = duckdb.connect()
    if args.progress:
        con.execute("SET enable_progress_bar = true")

    n_total = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{args.input}')"
    ).fetchone()[0]
    log(f"输入: {args.input} ({n_total:,} 行)")

    sample_n = min(args.sample_size, n_total)
    log(f"抽样: {sample_n}/{n_total:,} (seed={args.seed}) ...")

    con.execute(f"""
        CREATE TEMP TABLE sample_data AS
        SELECT * FROM read_parquet('{args.input}')
        USING SAMPLE {sample_n} ROWS
    """)
    n_samples = con.execute("SELECT COUNT(*) FROM sample_data").fetchone()[0]

    # Extract input stem for naming
    raw_stem = os.path.splitext(os.path.basename(args.input))[0]
    # Try to derive sample type from input path or output dir
    sample_type = "unknown"
    if "_keep." in raw_stem or "keep" in os.path.basename(out_dir).lower():
        sample_type = "keep"
    elif "_drop." in raw_stem or "drop" in os.path.basename(out_dir).lower():
        sample_type = "drop"
    out_parquet = os.path.join(out_dir, f"{raw_stem}_{sample_type}_qc.parquet" if sample_type != "unknown" else "audit_sample_v1.parquet")
    con.execute(f"COPY sample_data TO '{out_parquet}' (FORMAT PARQUET)")

    # keyword distribution
    kw_dist = con.execute("""
        SELECT keyword, COUNT(*) AS cnt FROM sample_data
        GROUP BY keyword ORDER BY cnt DESC LIMIT 20
    """).fetchall()

    col_info = con.execute("DESCRIBE sample_data").fetchall()
    title_col = next((c[0] for c in col_info if "title" in c[0].lower()), None)
    title_stats = ("N/A", "N/A", "N/A")
    if title_col:
        title_stats = con.execute(f"""
            SELECT MIN(LENGTH("{title_col}")), MAX(LENGTH("{title_col}")), AVG(LENGTH("{title_col}"))
            FROM sample_data
        """).fetchone()

    con.close()

    out_stats = os.path.join(out_dir, "audit_stats.csv")
    with open(out_stats, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["keyword", "count"])
        for kw, cnt in kw_dist:
            w.writerow([kw, cnt])

    elapsed = time.perf_counter() - t0
    print()
    print("=" * 62)
    print(f"  Phase 2 -- 抽样 完成")
    print("=" * 62)
    print(f"  样本:       {n_samples:,} / {n_total:,}")
    print(f"  耗时:       {elapsed:.1f}s")
    print(f"  产物:       {out_dir}/")
    print(f"              {os.path.basename(out_parquet)}")
    print(f"              audit_stats.csv")
    print("=" * 62)
    print()
    print(f"  → 标注 {os.path.basename(out_parquet)}，添加列:")

    mark_done(out_dir, 2, samples=n_samples, total=n_total, elapsed_sec=round(elapsed,1))
    write_run_log(2, args.input, out_dir,
                  stats={"total_rows": n_total, "sample_size": n_samples,
                         "seed": args.seed, "elapsed_sec": round(elapsed, 1)})

    print("    audit_label    = T / F / U")
    print("    audit_category = football / gaming / music / ...")
    print()
    print("  → 标注完成后:")
    print(f"    python3 phase3_analyze.py {out_parquet} -o runs/003_analysis/")


if __name__ == "__main__":
    main()

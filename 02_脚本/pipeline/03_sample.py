#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
"""
phase2_sample.py -- SOP Phase 2: 抽样 QC

支持:
- 基于统计学公式计算样本量: n = Z² * p(1-p) / e²
- 分层抽样（按 keyword）
- 简单随机抽样（默认）

用法:
  python3 phase2_sample.py baseline.parquet -o runs/002_audit/           # 95%置信, 5%误差, 自动计算样本量
  python3 phase2_sample.py baseline.parquet -o runs/002_audit/ -n 500    # 手动指定样本量
  python3 phase2_sample.py baseline.parquet -o runs/002_audit/ --margin 0.03  # 3% 误差 → 1067 样本
  python3 phase2_sample.py baseline.parquet -o runs/002_audit/ --stratify      # 分层抽样
"""

import sys, os, time, argparse, textwrap, csv
from pathlib import Path
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.sop import print_banner, write_run_log
from core.progress import mark_done
from core.io import duckdb_reader, strip_stem


# Z 值表
Z_TABLE = {90: 1.645, 95: 1.96, 99: 2.576}


def calc_sample_size(n_total: int, confidence: int = 95, margin: float = 0.05, p: float = 0.5) -> int:
    """
    n = Z² * p(1-p) / e²
    对有限总体做修正: n_adj = n / (1 + (n-1)/N)
    """
    z = Z_TABLE.get(confidence, 1.96)
    n_inf = (z ** 2) * p * (1 - p) / (margin ** 2)
    n_adj = n_inf / (1 + (n_inf - 1) / n_total) if n_total > 0 else n_inf
    return max(1, round(n_adj))


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    print_banner("sample")

    parser = argparse.ArgumentParser(
        description="抽样质检（支持公式计算 + 分层）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例:
              python3 03_sample.py baseline.parquet -o runs/002_audit/
              python3 03_sample.py keep.parquet -o runs/005_clean/run01/keep_qc/ --margin 0.03
              python3 03_sample.py drop.parquet -o runs/005_clean/run01/drop_qc/ --stratify
        """),
    )
    parser.add_argument("input", help="baseline.parquet 或 keep/drop.parquet")
    parser.add_argument("-o", "--output-dir", default="data/runs/002_audit",
                        help="输出目录")
    parser.add_argument("-n", "--sample-size", type=int, default=None,
                        help="手动指定样本量（不指定则用公式计算）")
    parser.add_argument("--confidence", type=int, default=95, choices=[90, 95, 99],
                        help="置信度 (默认: 95)")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="误差范围 (默认: 0.05 即 ±5%%)")
    parser.add_argument("--p", type=float, default=0.5,
                        help="预估比例 p (默认: 0.5 即最保守估计)")
    parser.add_argument("--stratify", action="store_true",
                        help="启用分层抽样（按 keyword 分层）")
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
        f"SELECT COUNT(*) FROM {duckdb_reader(args.input)}"
    ).fetchone()[0]
    log(f"输入: {args.input} ({n_total:,} 行)")

    # ── 样本量计算 ──
    if args.sample_size:
        sample_n = min(args.sample_size, n_total)
        log(f"样本量: 手动指定 = {sample_n}")
    else:
        n_calc = calc_sample_size(n_total, args.confidence, args.margin, args.p)
        sample_n = min(n_calc, n_total)
        z = Z_TABLE[args.confidence]
        log(f"样本量: 公式计算 = {n_calc} "
            f"(Z={z}, p={args.p}, e={args.margin}, N={n_total:,})")
        log(f"实际抽取: {sample_n}/{n_total:,}")

    # ── 抽样 ──
    if args.stratify:
        log(f"分层抽样: 按 keyword 比例分配 (seed={args.seed}) ...")

        # 计算每层的数量
        strata = con.execute(f"""
            WITH kw_counts AS (
                SELECT keyword, COUNT(*) AS total
                FROM {duckdb_reader(args.input)}
                GROUP BY keyword
            ),
            total AS (SELECT SUM(total) AS n FROM kw_counts)
            SELECT keyword, total,
                   ROUND(total * 1.0 / (SELECT n FROM total) * {sample_n}) AS alloc
            FROM kw_counts
            WHERE ROUND(total * 1.0 / (SELECT n FROM total) * {sample_n}) >= 1
            ORDER BY total DESC
        """).fetchall()

        log(f"  分层数: {len(strata)}")

        # 每层独立抽样，INSERT INTO 合并（全程 DuckDB，不经过 pandas）
        con.execute(f"""
            CREATE TEMP TABLE sample_data AS
            SELECT * FROM {duckdb_reader(args.input)} LIMIT 0
        """)

        for keyword, kw_total, alloc in strata:
            n_layer = min(alloc, kw_total)
            if n_layer == 0:
                continue
            safe_kw = keyword.replace("'", "''")
            con.execute(f"""
                INSERT INTO sample_data
                SELECT * FROM (
                    SELECT * FROM {duckdb_reader(args.input)}
                    WHERE keyword = '{safe_kw}'
                ) USING SAMPLE {n_layer} ROWS
            """)

    else:
        log(f"简单随机抽样 (seed={args.seed}) ...")
        con.execute(f"""
            CREATE TEMP TABLE sample_data AS
            SELECT * FROM {duckdb_reader(args.input)}
            USING SAMPLE {sample_n} ROWS
        """)

    n_samples = con.execute("SELECT COUNT(*) FROM sample_data").fetchone()[0]

    # ── 输出命名 ──
    dir_base = os.path.basename(out_dir.rstrip('/'))
    if 'keep' in dir_base.lower():
        sample_type = 'keep'
    elif 'drop' in dir_base.lower():
        sample_type = 'drop'
    else:
        sample_type = 'qc'

    raw_stem = os.path.splitext(os.path.basename(args.input))[0]
    clean_stem = strip_stem(raw_stem)
    date_tag = time.strftime("%m%d")
    out_name = f"{clean_stem}_sample_{date_tag}.parquet"
    out_parquet = os.path.join(out_dir, out_name)
    con.execute(f"COPY sample_data TO '{out_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    out_csv = out_parquet.replace(".parquet", ".csv")
    con.execute(f"COPY sample_data TO '{out_csv}' (FORMAT CSV, HEADER true)")

    # keyword distribution
    kw_dist = con.execute("""
        SELECT keyword, COUNT(*) AS cnt FROM sample_data
        GROUP BY keyword ORDER BY cnt DESC LIMIT 20
    """).fetchall()

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
    print(f"  总体:       {n_total:>12,}")
    print(f"  样本:       {n_samples:>12,}")
    print(f"  置信度:     {args.confidence}%")
    print(f"  误差:       ±{args.margin*100:.0f}%")
    print(f"  方式:       {'分层(stratified)' if args.stratify else '简单随机(SRS)'}")
    print(f"  耗时:       {elapsed:.1f}s")
    print(f"  产物:       {out_dir}/")
    print(f"              {os.path.basename(out_parquet)}")
    print(f"              {os.path.basename(out_csv)}")
    print(f"              audit_stats.csv")
    print("=" * 62)
    print()
    print(f"  → 标注 {os.path.basename(out_parquet)}，添加列:")

    mark_done(out_dir, "sample", samples=n_samples, total=n_total,
              confidence=args.confidence, margin=args.margin,
              stratify=args.stratify, elapsed_sec=round(elapsed, 1))
    write_run_log("sample", args.input, out_dir,
                  stats={"total_rows": n_total, "sample_size": n_samples,
                         "confidence": args.confidence, "margin": args.margin,
                         "stratify": args.stratify, "seed": args.seed,
                         "elapsed_sec": round(elapsed, 1),
                         "sample_parquet": out_parquet,
                         "sample_csv": out_csv},
                  command=f"03_sample.py {args.input} -o {out_dir}")

    print("    audit_label    = T / F / U")
    print("    audit_category = language / cartoon / gaming / ...")
    print()
    print("  → 标注完成后:")
    print(f"    python3 04_analyze.py {out_parquet} -o runs/003_analysis/")


if __name__ == "__main__":
    main()

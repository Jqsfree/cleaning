#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
"""
06_dedup.py — 新批次 vs 旧批次去重

从 baseline 中移除旧批次已交付的 video_id。

用法:
  # 单个旧交付
  02_脚本/pipeline/06_dedup.py baseline.parquet -d old/07_deliver/0724_deliver_ge720.csv -o deduped.parquet

  # 批次根（自动找 07_deliver/）
  02_脚本/pipeline/06_dedup.py baseline.parquet --old-run data/runs/film_tv/human_0724/ -o deduped.parquet
"""

import sys, os, argparse, glob
from pathlib import Path
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.progress import mark_done
from core.sop import write_run_log


def find_deliver_csvs(source: str) -> list:
    """从目录或文件路径解析交付 CSV 列表（兼容新旧布局）。"""
    if os.path.isfile(source):
        return [source]
    if not os.path.isdir(source):
        return []

    patterns = [
        # 新布局
        os.path.join(source, "07_deliver", "*_deliver*.csv"),
        os.path.join(source, "07_deliver", "*.csv"),
        # 旧布局
        os.path.join(source, "deliver", "*_keep_final.csv"),
        os.path.join(source, "deliver", "*_deliver*.csv"),
        os.path.join(source, "*_keep_final.csv"),
        os.path.join(source, "*_deliver*.csv"),
    ]
    found: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            if not os.path.isfile(p):
                continue
            if p in seen:
                continue
            name = os.path.basename(p).lower()
            if name.startswith("."):
                continue
            seen.add(p)
            found.append(p)
        if found and ("07_deliver" in pat or "deliver" in pat):
            # 优先返回该布局下的命中，避免根目录杂 CSV
            if any("/07_deliver/" in f or "/deliver/" in f for f in found):
                return [f for f in found if "/07_deliver/" in f or "/deliver/" in f]
    return found


def main():
    parser = argparse.ArgumentParser(description="新批次 vs 旧批次去重")
    parser.add_argument("input", help="新批次 baseline.parquet")
    parser.add_argument("-d", "--deliveries", nargs="+", default=[],
                        help="旧批次交付 CSV 或目录（可多个）")
    parser.add_argument("--old-run", nargs="+", default=[],
                        help="旧批次 run 目录（自动找 07_deliver/）")
    parser.add_argument("-o", "--output", required=True,
                        help="输出去重后的 parquet")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] baseline 不存在: {args.input}")
        sys.exit(1)

    old_csvs = []
    for d in args.deliveries:
        old_csvs.extend(find_deliver_csvs(d))
    for r in args.old_run:
        old_csvs.extend(find_deliver_csvs(r))

    # 去重路径
    seen = set()
    uniq = []
    for c in old_csvs:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    old_csvs = uniq

    if not old_csvs:
        print("[ERROR] 未找到旧交付。用 -d 或 --old-run 指定（支持 07_deliver/*_deliver*.csv）。")
        sys.exit(1)

    print(f"旧交付: {len(old_csvs)} 个")
    for c in old_csvs:
        print(f"  {c}")

    con = duckdb.connect()
    n_bl = con.execute(f"SELECT COUNT(*) FROM read_parquet('{args.input}')").fetchone()[0]
    print(f"\nbaseline: {n_bl:,}")

    parts = []
    for c in old_csvs:
        parts.append(f"SELECT video_id FROM read_csv_auto('{c}', header=true, all_varchar=true, sample_size=-1)")
    sql = " UNION ALL ".join(parts)
    con.execute(f"CREATE TEMP TABLE old_ids AS SELECT DISTINCT video_id FROM ({sql}) t")
    n_old = con.execute("SELECT COUNT(*) FROM old_ids").fetchone()[0]
    print(f"旧 ID 去重后: {n_old:,}")

    con.execute(f"""
        CREATE TEMP TABLE deduped AS
        SELECT b.* FROM read_parquet('{args.input}') b
        WHERE b.video_id NOT IN (SELECT video_id FROM old_ids)
    """)
    n_out = con.execute("SELECT COUNT(*) FROM deduped").fetchone()[0]
    print(f"baseline: {n_bl:,} → 去重后: {n_out:,} (移除 {n_bl-n_out:,})")

    if not args.dry_run:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        con.execute(f"COPY (SELECT * FROM deduped) TO '{args.output}' (FORMAT PARQUET)")
        print(f"已写出: {args.output}")
    else:
        print("[dry-run]")

    con.close()

    if not args.dry_run:
        out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
        mark_done(out_dir, "dedup",
                  baseline=n_bl, old_ids=n_old, output=n_out,
                  removed=n_bl - n_out, output_path=args.output)
        write_run_log(
            "dedup", args.input, out_dir,
            stats={
                "baseline": n_bl,
                "old_ids": n_old,
                "output_rows": n_out,
                "removed": n_bl - n_out,
                "output_path": args.output,
                "old_deliveries": len(old_csvs),
            },
            command=f"06_dedup.py {args.input} -o {args.output}",
        )


if __name__ == "__main__":
    main()

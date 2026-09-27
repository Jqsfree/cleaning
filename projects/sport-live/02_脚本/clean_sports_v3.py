#!/usr/bin/env python3
"""
clean_sports_v3.py -- SOP Phase 5: DuckDB + Parquet 规则清洗

用法:
  python3 clean_sports_v3.py baseline.parquet -o data/runs/005_clean/
  python3 clean_sports_v3.py baseline.parquet -o data/runs/005_clean/ --run run02
  python3 clean_sports_v3.py baseline.parquet -o data/runs/005_clean/ --chunksize 500000 -p
"""

import sys, os, time, argparse, textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.sop import load_sop, print_banner, write_run_log
from core.progress import update, mark_done
from core.rules_manager import load_rules, get_rules_summary
from core.playlist import analyze
from core.cleaner import clean
from core.reports import generate


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    sop_text = load_sop()
    if sop_text: print(sop_text[:600] + "...\n")
    print_banner(5)

    parser = argparse.ArgumentParser(description="SOP Phase 5: 规则清洗 (DuckDB+Parquet)")
    parser.add_argument("input", help="baseline.parquet 或 CSV")
    parser.add_argument("-o", "--output-dir", default="data/runs/005_clean")
    parser.add_argument("--run", default="run01", help="迭代轮次")
    parser.add_argument("--chunksize", type=int, default=0)
    parser.add_argument("--sample-n", type=int, default=200)
    parser.add_argument("-p", "--progress", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] {args.input}"); sys.exit(1)

    out_dir = os.path.join(args.output_dir.rstrip("/"), args.run)
    os.makedirs(out_dir, exist_ok=True)

    # 规则：data/runs/{dataset}/rules/ > 主规则
    # 数据集名从输出路径提取，转为绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parts = os.path.normpath(os.path.abspath(args.output_dir)).split(os.sep)
    ds_rules_dir = None
    for i, p in enumerate(parts):
        if p == "runs" and i + 1 < len(parts):
            ds_rules_dir = os.sep + os.path.join(*parts[:i+2], "rules")
            break
    if ds_rules_dir is None:
        ds_rules_dir = os.path.join(args.output_dir.rstrip('/'), "rules")

    # 优先数据集专属规则，不存在则用主规则
    rules_path = os.path.join(ds_rules_dir, "blacklist.toml")
    if os.path.exists(rules_path):
        log("规则: 数据集专属")
        rules = load_rules(rules_path)
        from core import scoring
        scoring.set_rules_dir(ds_rules_dir)
    else:
        log("规则: 主规则 (数据集专属规则不存在)")
        rules = load_rules()
    log(f"  {get_rules_summary(rules)}")

    t0 = time.perf_counter()
    print(f"Phase 5: {args.input} → {out_dir}/")

    update(out_dir, 5, status="running", stage="pass1")
    polluted = analyze(args.input, chunksize=args.chunksize)

    update(out_dir, 5, status="running", stage="pass2")
    raw_stem = os.path.splitext(os.path.basename(args.input))[0]
    # Remove _raw suffix if present
    if raw_stem.endswith("_raw"):
        raw_stem = raw_stem[:-4]
    summary = clean(args.input, polluted, "clean", out_dir, fmt="parquet", raw_name=raw_stem, run=args.run)

    generate(args.input, out_dir, "clean", summary, fmt="parquet", sample_n=args.sample_n, raw_name=raw_stem, run=args.run)

    # _done
    retention = summary["total_keep"] / max(summary["total_rows"], 1)
    input_stem = Path(args.input).stem
    if 0.65 <= retention <= 0.85:
        import shutil
        src_done = os.path.join(out_dir, f"{raw_stem}_{args.run}_keep.parquet")
        dst_done = os.path.join(out_dir, f"{input_stem}_done.parquet")
        if os.path.exists(src_done):
            shutil.copy2(src_done, dst_done)
        log(f"可落盘: {input_stem}_done.parquet ({retention*100:.1f}%)")
    else:
        log(f"通过率 {retention*100:.1f}% 不在 65-85%, 跳过 _done")

    elapsed = time.perf_counter() - t0
    mark_done(out_dir, 5, keep=summary["total_keep"], drop=summary["total_drop"],
              retention_pct=round(retention*100, 1), elapsed_sec=round(elapsed, 1))
    write_run_log(5, args.input, out_dir,
                  stats={"total": summary["total_rows"], "keep": summary["total_keep"],
                         "drop": summary["total_drop"], "retention_pct": round(retention*100, 1),
                         "elapsed_sec": round(elapsed, 1)})

    print(f"\nPhase 5 done ({elapsed:.1f}s) | keep={summary['total_keep']:,} | {retention*100:.1f}%")


if __name__ == "__main__":
    main()

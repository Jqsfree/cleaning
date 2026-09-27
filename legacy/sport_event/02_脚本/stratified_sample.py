#!/usr/bin/env python3
"""
stratified_sample.py
按 keyword 分层抽样。

两种模式:

1. 统计公式模式（默认）：每层用有限总体修正公式独立计算样本量
   n = Z²·p(1-p)·N / (N·e² + Z²·p(1-p))
   适合：需要统计推断（估计 precision 置信区间）

2. 总量控制模式（--max-total）：按各层占比分配总预算
   n_i = N_i / N_total * max_total
   适合：限定总样本量，按比例观察各层

用法:
  # 公式模式（每层独立计算）
  python3 stratified_sample.py input.csv

  # 总量控制：总共只抽 2000 条
  python3 stratified_sample.py input.csv --max-total 2000

  # 总量控制 + 调整上下限
  python3 stratified_sample.py input.csv --max-total 2000 --min-per-layer 5 --max-per-layer 200
"""

import sys
import math
import argparse
import time
import os

try:
    import duckdb
except ImportError:
    print("[ERROR] pip install duckdb")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("[ERROR] pip install pandas")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="按 keyword 分层抽样")
    p.add_argument("input", nargs="?", default="filtered_records.csv",
                   help="输入 CSV（默认：filtered_records.csv）")
    p.add_argument("--output", "-o", default="sample_for_qc.csv",
                   help="输出 CSV（默认：sample_for_qc.csv）")
    p.add_argument("--keyword-col", default="keyword",
                   help="分层列名（默认：keyword）")

    # 公式模式参数
    p.add_argument("--confidence", type=float, default=0.95,
                   choices=[0.90, 0.95, 0.99],
                   help="置信度：0.90 / 0.95 / 0.99（默认：0.95）")
    p.add_argument("--p", type=float, default=0.5,
                   help="预估通过率 p（默认：0.5）")
    p.add_argument("--error", type=float, default=0.05,
                   help="允许误差 e（默认：0.05 即 ±5%%）")

    # 总量控制模式
    p.add_argument("--max-total", type=int, default=0,
                   help="总样本量上限（设置后启用总量控制模式。0=禁用）")

    # 通用参数
    p.add_argument("--min-per-layer", type=int, default=5,
                   help="每层最小抽样量（默认：5）")
    p.add_argument("--max-per-layer", type=int, default=200,
                   help="每层最大抽样量（默认：200）")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子（默认：42）")
    p.add_argument("--encoding", default="utf-8-sig",
                   help="输出编码（默认：utf-8-sig）")
    return p.parse_args()


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def allocate_proportional(layer_sizes: list, max_total: int,
                          min_n: int, max_n: int) -> list:
    """
    按层占比分配总预算，约束在 [min_n, max_n] 之间。

    layer_sizes: [(kw, N), ...]  按 N 降序排列
    返回: [(kw, N, n_allocated), ...]
    """
    total_N = sum(N for _, N in layer_sizes)
    if total_N == 0:
        return [(kw, N, 0) for kw, N in layer_sizes]

    # 首轮按比例分配
    alloc = []
    for kw, N in layer_sizes:
        n = max(1, round(N / total_N * max_total))
        n = max(min_n, min(max_n, n, N))
        alloc.append([kw, N, n])

    # 调整总量逼近 max_total
    current_total = sum(a[2] for a in alloc)
    max_iter = 100

    while current_total != max_total and max_iter > 0:
        max_iter -= 1

        if current_total > max_total:
            # 需要削减：从超过 min_n 的大层减起
            excess = current_total - max_total
            for a in sorted(alloc, key=lambda x: -(x[2] - min_n)):
                if excess <= 0:
                    break
                slack = a[2] - min_n
                if slack > 0:
                    cut = min(slack, excess)
                    a[2] -= cut
                    excess -= cut
        else:
            # 需要增加：往还没到 max_n 的层加
            deficit = max_total - current_total
            for a in sorted(alloc, key=lambda x: -(max_n - x[2])):
                if deficit <= 0:
                    break
                room = min(max_n, a[1]) - a[2]
                if room > 0:
                    add = min(room, deficit)
                    a[2] += add
                    deficit -= add

        current_total = sum(a[2] for a in alloc)

    return alloc


def process(args):
    start = time.perf_counter()

    Z_MAP = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    Z = Z_MAP[args.confidence]
    p = args.p
    e = args.error
    kw_col = args.keyword_col
    use_total_cap = args.max_total > 0

    n_theory_inf = math.ceil((Z ** 2 * p * (1 - p)) / (e ** 2))

    print(f"{'='*52}")
    print(f"  分层抽样参数")
    print(f"{'='*52}")
    if use_total_cap:
        print(f"  模式           ：总量控制（按占比分配）")
        print(f"  总样本量上限   ：{args.max_total:,} 条")
    else:
        print(f"  模式           ：统计公式（每层独立计算）")
        print(f"  置信度         ：{args.confidence*100:.0f}%  (Z = {Z})")
        print(f"  预估通过率     ：p = {p}")
        print(f"  允许误差       ：±{e*100:.0f}%")
        print(f"  无限总体理论 n ：{n_theory_inf} 条/层")
    print(f"  每层下限       ：{args.min_per_layer} 条")
    print(f"  每层上限       ：{args.max_per_layer} 条")
    print(f"  随机种子       ：{args.seed}")
    print(f"{'='*52}\n")

    # 支持 glob pattern + 逗号分隔
    import glob as _glob
    _input_files = []
    for part in args.input.split(","):
        part = part.strip()
        if "*" in part or "?" in part:
            _input_files.extend(sorted(_glob.glob(part)))
        elif os.path.exists(part):
            _input_files.append(part)
    if not _input_files:
        log(f"[ERROR] 找不到文件：{args.input}")
        sys.exit(1)

    con = duckdb.connect()
    con.execute("SET temp_directory='/home/jqs/tiyu/duckdb_spill'")

    # ── Step 1：加载 ─────────────────────────────────────
    log(f"加载数据：{len(_input_files)} 个文件 ...")
    for f in _input_files:
        log(f"  {f}")
    t_load = time.perf_counter()

    con.execute("""
        CREATE TEMP TABLE _full_data AS
        SELECT *
        FROM read_csv_auto($1, header=true, ignore_errors=true)
    """, [_input_files])
    total_raw = con.execute("SELECT COUNT(*) FROM _full_data").fetchone()[0]
    log(f"  已加载 {total_raw:,} 行，耗时 {time.perf_counter() - t_load:.1f}s")

    # ── Step 2：统计各层 ──────────────────────────────────
    log(f"统计分层（列：{kw_col}）...")
    layer_stats = con.execute(f"""
        SELECT
            COALESCE(CAST("{kw_col}" AS VARCHAR), '__NULL__') AS kw,
            COUNT(*) AS total
        FROM _full_data
        GROUP BY kw
        ORDER BY total DESC
    """).df()

    n_layers = len(layer_stats)
    log(f"发现 {n_layers} 个分层\n")

    # ── Step 3：计算每层分配量 ────────────────────────────
    layer_sizes = [(str(r["kw"]), int(r["total"])) for _, r in layer_stats.iterrows()]

    if use_total_cap:
        alloc = allocate_proportional(layer_sizes, args.max_total,
                                      args.min_per_layer, args.max_per_layer)
    else:
        # 公式模式
        alloc = []
        for kw, N in layer_sizes:
            n_finite = math.ceil(
                (Z**2 * p * (1-p)) * N / (N * e**2 + Z**2 * p * (1-p))
            )
            n_actual = max(args.min_per_layer, min(args.max_per_layer, n_finite))
            n_actual = min(n_actual, N)
            alloc.append([kw, N, n_actual])

    # ── Step 4：打印 + 抽样 ───────────────────────────────
    col_w = 48
    total_target = sum(a[2] for a in alloc)
    print(f"  预期总抽样: {total_target:,} 条\n")
    print(f"  {'keyword':<{col_w}} {'总量':>10} {'分配':>7} {'占比':>8}")
    print(f"  {'─'*col_w} {'─'*10} {'─'*7} {'─'*8}")

    frames = []
    total_sampled = 0

    for kw, N, n_actual in alloc:
        if n_actual <= 0:
            continue

        rate = n_actual / N * 100
        kw_display = kw if len(kw) <= col_w else kw[:col_w-3] + "..."
        print(f"  {kw_display:<{col_w}} {N:>10,} {n_actual:>7,} {rate:>7.1f}%")

        if kw == "__NULL__":
            where_clause = f'"{kw_col}" IS NULL'
            params = []
        else:
            where_clause = f'CAST("{kw_col}" AS VARCHAR) = $1'
            params = [kw]

        sql = f"""
            SELECT *
            FROM _full_data
            WHERE {where_clause}
            ORDER BY random()
            LIMIT {n_actual}
        """
        sample = con.execute(sql, params).df()

        actual_got = len(sample)
        if actual_got < n_actual:
            log(f"  [WARN] {kw_display!r}：目标 {n_actual} 条，实际只有 {actual_got} 条")

        total_sampled += actual_got
        frames.append(sample)

    # ── Step 5：合并 & 写出 ───────────────────────────────
    print()
    log("合并抽样结果...")
    df_sample = pd.concat(frames, ignore_index=True)
    df_sample = df_sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df_sample.to_csv(args.output, index=False, encoding=args.encoding)
    out_mb = os.path.getsize(args.output) / 1024 / 1024

    elapsed = time.perf_counter() - start
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60
    overall_rate = total_sampled / total_raw * 100 if total_raw else 0

    print()
    print(f"{'='*52}")
    print(f"  抽样结果汇总")
    print(f"{'='*52}")
    print(f"  运行时长        ：{h:02d}h {m:02d}m {s:05.2f}s")
    print(f"  原始总量        ：{total_raw:>14,} 条")
    print(f"  分层数          ：{n_layers:>14,} 层")
    print(f"  抽样总量        ：{total_sampled:>14,} 条")
    print(f"  整体抽样率      ：{overall_rate:>13.2f}%")
    print(f"  输出文件大小    ：{out_mb:>13.1f} MB")
    print(f"  输出路径        ：{os.path.abspath(args.output)}")
    print(f"{'='*52}")

    if use_total_cap:
        print(f"\n  模式：总量控制。每层按占比分配，总样本量控制在 ~{args.max_total:,} 条。")
    else:
        print(f"\n  模式：统计公式。每层独立计算，可做 precision 置信区间估计。")


if __name__ == "__main__":
    args = parse_args()
    process(args)

#!/usr/bin/env python3
"""
tools/batch_ops/sample_qc.py — 抽样 QC（目录阶段：02_sample/）

由 ``pipeline/03_sample.py`` 迁入（跨品类通用，故落在 tools/batch_ops）。
CLI 参数保持不变，orchestrate 等既有调用方无需改命令。

支持:
- 基于统计学公式计算样本量: n = Z² * p(1-p) / e²
- 分层抽样（按 keyword，配额用最大余额法，Σ配额 == 目标样本量）
- 简单随机抽样（默认）；--seed 经 DuckDB REPEATABLE 复现
- 落盘 margin_effective：实际样本量对应的误差（分层丢小层 / 手动 -n 时 ≠ 请求误差）
- 分层模式额外落盘 audit_strata.csv（各层 pool_rows/alloc/weight），供加权 pass_rate
- 输出命名：``{品类}_{MMDD}_sample.{csv,parquet}``（品类由 -o 的批次路径推断，
  可用 --name-tag 覆盖）

用法:
  PYTHONPATH=02_脚本 .venv/bin/python3 02_脚本/tools/batch_ops/sample_qc.py \\
      keep.csv -o data/runs/exo_service/machine_0813/02_sample/ -n 385
  PYTHONPATH=02_脚本 .venv/bin/python3 02_脚本/tools/batch_ops/sample_qc.py \\
      keep.csv -o data/runs/exo_service/machine_0813/02_sample/ --stratify
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import duckdb

_SCRIPT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SCRIPT))

from core.batch_layout import infer_category, require_output_dir, warn_outside_batch  # noqa: E402
from core.io import duckdb_reader, strip_stem, warn_csv_row_skew  # noqa: E402
from core.log import log as core_log  # noqa: E402
from core.progress import mark_done  # noqa: E402
from core.run_manifest import maybe_update_stage  # noqa: E402
from core.sop import print_banner, write_run_log  # noqa: E402

PROG = "tools/batch_ops/sample_qc.py"

# Z 值表
Z_TABLE = {90: 1.645, 95: 1.96, 99: 2.576}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def calc_sample_size(
    n_total: int,
    confidence: int = 95,
    margin: float = 0.05,
    p: float = 0.5,
) -> int:
    """
    n = Z² * p(1-p) / e²
    对有限总体做修正: n_adj = n / (1 + (n-1)/N)
    """
    z = Z_TABLE.get(confidence, 1.96)
    n_inf = (z ** 2) * p * (1 - p) / (margin ** 2)
    n_adj = n_inf / (1 + (n_inf - 1) / n_total) if n_total > 0 else n_inf
    return max(1, round(n_adj))


def effective_margin(
    n_samples: int,
    n_total: int,
    confidence: int = 95,
    p: float = 0.5,
) -> float:
    """实际样本量 n 对应的有限总体误差 e = Z·√(p(1-p)/n)·√((N-n)/(N-1))。"""
    if n_samples <= 1 or n_total <= 1:
        return float("nan")
    z = Z_TABLE.get(confidence, 1.96)
    e = z * ((p * (1 - p) / n_samples) ** 0.5)
    return e * (((n_total - n_samples) / (n_total - 1)) ** 0.5)


def _stratified_alloc(con: duckdb.DuckDBPyConnection, reader: str, sample_n: int) -> list[tuple]:
    """按 keyword 分层，用最大余额法分配配额（Σalloc == sample_n）。

    floor 分配后把余量按小数部分从大到小补足，保证 Σalloc == n_target。
    旧实现用 ROUND + alloc>=1 过滤，逐层取整丢零头会把 n_target 缩水
    （如 270 → 180），且各层抽样率被拉偏；此处修正。
    """
    return con.execute(f"""
        WITH kw_counts AS (
            SELECT keyword, COUNT(*) AS total
            FROM {reader}
            WHERE keyword IS NOT NULL
            GROUP BY keyword
        ),
        pool AS (SELECT SUM(total) AS n FROM kw_counts),
        quota AS (
            SELECT keyword,
                   total,
                   (CAST(total AS HUGEINT) * {sample_n}) // CAST((SELECT n FROM pool) AS HUGEINT)
                       AS floor_alloc,
                   (CAST(total AS HUGEINT) * {sample_n}) %  CAST((SELECT n FROM pool) AS HUGEINT)
                       AS remainder
            FROM kw_counts
        ),
        ranked AS (
            SELECT keyword, total, floor_alloc,
                   ROW_NUMBER() OVER (ORDER BY remainder DESC, total DESC, keyword ASC) AS rn,
                   {sample_n} - (SELECT SUM(floor_alloc) FROM quota) AS n_leftover
            FROM quota
        )
        SELECT keyword, total,
               floor_alloc + CASE WHEN rn <= n_leftover THEN 1 ELSE 0 END AS alloc
        FROM ranked
        WHERE floor_alloc + CASE WHEN rn <= n_leftover THEN 1 ELSE 0 END >= 1
        ORDER BY total DESC, keyword ASC
    """).fetchall()


def build_sample(
    input_path: str,
    out_dir: str,
    *,
    sample_size: int | None = None,
    confidence: int = 95,
    margin: float = 0.05,
    prop: float = 0.5,
    stratify: bool = False,
    seed: int = 42,
    name_tag: str | None = None,
    progress: bool = False,
    log_fn=None,
) -> dict[str, Any]:
    """执行抽样并落盘，返回统计 dict（纯数据操作，不做控制台汇总打印）。

    产物：
      - ``{name_tag}_sample_{MMDD}.parquet`` / ``.csv``
      - ``audit_stats.csv``（样本 keyword 分布）
      - ``audit_strata.csv``（仅分层模式；各层 pool_rows/alloc/weight）
    """
    logger = log_fn or log
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    con = duckdb.connect()
    if progress:
        con.execute("SET enable_progress_bar = true")

    reader = duckdb_reader(input_path)
    n_total = con.execute(f"SELECT COUNT(*) FROM {reader}").fetchone()[0]
    warn_csv_row_skew(input_path, n_total, log_fn=core_log)
    logger(f"输入: {input_path} ({n_total:,} 行)")

    # ── 样本量计算 ──
    if sample_size:
        sample_n = min(sample_size, n_total)
        logger(f"样本量: 手动指定 = {sample_n}")
    else:
        n_calc = calc_sample_size(n_total, confidence, margin, prop)
        sample_n = min(n_calc, n_total)
        logger(f"样本量: 公式计算 = {n_calc} "
                f"(Z={Z_TABLE[confidence]}, p={prop}, e={margin}, N={n_total:,})")
        logger(f"实际抽取: {sample_n}/{n_total:,}")

    seed = int(seed)
    strata_csv = ""
    strata_rows_covered = n_total          # 仅分层模式会改写；非分层 = 全池
    strata_coverage = 1.0

    # ── 抽样（reservoir + REPEATABLE） ──
    if stratify:
        logger(f"分层抽样: 按 keyword 比例分配, 最大余额法 (seed={seed}) ...")
        strata = _stratified_alloc(con, reader, sample_n)

        n_strata_total = con.execute(
            f"SELECT COUNT(DISTINCT keyword) FROM {reader} WHERE keyword IS NOT NULL"
        ).fetchone()[0]
        alloc_sum = sum(int(a) for _, _, a in strata)
        logger(f"  分层数: {len(strata)} (池内 keyword 共 {n_strata_total:,} 个，"
               f"配额为 0 的 {n_strata_total - len(strata):,} 个小层不参与抽样)")
        logger(f"  配额合计: {alloc_sum} / 目标 {sample_n}")

        # 覆盖度体检：floor 配额为 0 的小层会被整层丢弃 → 加权估计只能覆盖「参与分层」的行。
        # 池内 keyword 越碎片化（层均行数 << N/n），覆盖度越低；此时应改用 SRS。
        strata_rows_covered = sum(int(t) for _, t, _ in strata)
        n_rows_kw = con.execute(
            f"SELECT COUNT(*) FROM {reader} WHERE keyword IS NOT NULL"
        ).fetchone()[0]
        strata_coverage = strata_rows_covered / max(n_rows_kw, 1)
        logger(f"  分层覆盖: {strata_rows_covered:,} / {n_rows_kw:,} 行 "
               f"({strata_coverage * 100:.1f}%)；未覆盖 {n_rows_kw - strata_rows_covered:,} 行"
               f"（{(1 - strata_coverage) * 100:.1f}%）在样本中无任何代表")
        if strata_coverage < 0.95:
            core_log(
                f"[WARN] 分层覆盖仅 {strata_coverage * 100:.1f}%（< 95%）：加权 pass_rate 只能外推"
                f"「参与分层的 {strata_rows_covered:,} 行」这一子总体，未覆盖的 "
                f"{n_rows_kw - strata_rows_covered:,} 行需另行假设。层均行数 "
                f"{n_rows_kw / max(n_strata_total, 1):.0f} 越小越容易触发；"
                f"建议改为 SRS（去掉 --stratify）以对全池无偏。"
            )
        assert alloc_sum == min(sample_n, n_total), \
            f"配额合计 {alloc_sum} ≠ 目标 {min(sample_n, n_total)}"

        con.execute(f"CREATE TEMP TABLE sample_data AS SELECT * FROM {reader} LIMIT 0")

        for layer_i, (keyword, kw_total, alloc) in enumerate(strata):
            n_layer = min(int(alloc), int(kw_total))
            if n_layer == 0 or keyword is None:
                continue
            safe_kw = keyword.replace("'", "''")
            con.execute(f"""
                INSERT INTO sample_data
                SELECT * FROM (
                    SELECT * FROM {reader} WHERE keyword = '{safe_kw}'
                ) USING SAMPLE reservoir({n_layer}) REPEATABLE ({seed + layer_i})
            """)

        # 分层配额明细：各层抽样率不等，算 pass_rate 时用 weight 做加权估计
        # （未加权均值会被小层抬高——层抽样率 ∝ 1/层行数）
        strata_csv = os.path.join(out_dir, "audit_strata.csv")
        with open(strata_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["keyword", "pool_rows", "alloc", "weight"])
            for keyword, kw_total, alloc in strata:
                n_layer = min(int(alloc), int(kw_total))
                if n_layer == 0:
                    continue
                w.writerow([keyword, int(kw_total), n_layer,
                            round(int(kw_total) / n_layer, 4)])
        logger(f"  分层配额明细: {strata_csv}")

    else:
        logger(f"简单随机抽样 (seed={seed}) ...")
        con.execute(f"""
            CREATE TEMP TABLE sample_data AS
            SELECT * FROM {reader}
            USING SAMPLE reservoir({sample_n}) REPEATABLE ({seed})
        """)

    n_samples = con.execute("SELECT COUNT(*) FROM sample_data").fetchone()[0]

    # ── 实际误差（分层丢层/手动 -n 都会让实际样本量 ≠ 目标） ──
    n_target = min(sample_n, n_total)
    margin_eff = effective_margin(n_samples, n_total, confidence, prop)
    if abs(margin_eff - margin) > 0.005:
        core_log(f"[WARN] 实际样本量 {n_samples:,} (目标 {n_target:,}) 对应误差 "
                 f"±{margin_eff*100:.2f}%，与请求的 ±{margin*100:.0f}% 不符；"
                 f"run_log/README 请以 margin_effective 为准")

    # ── 输出命名：{品类}_{MMDD}_sample.{ext} ──
    clean_stem = strip_stem(os.path.splitext(os.path.basename(input_path))[0])
    tag = name_tag or infer_category(out_dir) or clean_stem
    date_tag = time.strftime("%m%d")
    out_parquet = os.path.join(out_dir, f"{tag}_sample_{date_tag}.parquet")
    con.execute(f"COPY sample_data TO '{out_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    out_csv = out_parquet.replace(".parquet", ".csv")
    con.execute(f"COPY sample_data TO '{out_csv}' (FORMAT CSV, HEADER true)")

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
    return {
        "total_rows": n_total,
        "sample_size": n_samples,
        "sample_size_target": n_target,
        "confidence": confidence,
        "margin": margin,
        "margin_effective": round(margin_eff, 4),
        "name_tag": tag,
        "stratify": stratify,
        "seed": seed,
        "elapsed_sec": round(elapsed, 1),
        "sample_parquet": out_parquet,
        "sample_csv": out_csv,
        "audit_stats_csv": out_stats,
        "audit_strata_csv": strata_csv,
        "strata_rows_covered": strata_rows_covered,
        "strata_coverage_pct": round(100 * strata_coverage, 2),
        "out_dir": out_dir,
    }


def _print_summary(stats: dict[str, Any]) -> None:
    pct = f"±{stats['margin_effective']*100:.2f}%"
    print()
    print("=" * 62)
    print("  抽样 完成（目录阶段 02_sample）")
    print("=" * 62)
    print(f"  总体:       {stats['total_rows']:>12,}")
    print(f"  样本:       {stats['sample_size']:>12,}  (目标 {stats['sample_size_target']:,})")
    print(f"  置信度:     {stats['confidence']}%")
    print(f"  误差:       {pct:>12}  (请求 ±{stats['margin']*100:.0f}%)")
    print(f"  seed:       {stats['seed']}")
    print(f"  方式:       {'分层(stratified)' if stats['stratify'] else '简单随机(SRS)'}")
    print(f"  耗时:       {stats['elapsed_sec']:.1f}s")
    print(f"  产物:       {stats['out_dir']}/")
    print(f"              {os.path.basename(stats['sample_parquet'])}")
    print(f"              {os.path.basename(stats['sample_csv'])}")
    print("              audit_stats.csv")
    if stats["stratify"]:
        print("              audit_strata.csv  (分层权重，加权 pass_rate 用)")
    print("=" * 62)
    print()
    print(f"  → 标注 {os.path.basename(stats['sample_parquet'])}，添加列:")
    print("    audit_label    = T / F / U")
    print("    audit_category = language / cartoon / gaming / ...")
    print()
    print("  → 标注完成后:")
    print(f"    02_脚本/pipeline/04_analyze.py {stats['sample_parquet']} -o <batch>/03_qc/analysis/")


def main() -> int:
    print_banner("sample")
    parser = argparse.ArgumentParser(
        description="抽样质检（支持公式计算 + 分层）；落盘目录建议 02_sample/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f"""\
            示例:
              02_脚本/{PROG} keep.csv -o data/runs/exo_service/machine_0813/02_sample/ -n 385
              02_脚本/{PROG} keep.csv -o …/02_sample/ --margin 0.03 --seed 42
              02_脚本/{PROG} keep.csv -o …/02_sample/ --stratify
            注意: 比例参数用 --prop，勿用 -p（旧名仍兼容但隐藏）。
        """),
    )
    parser.add_argument("input", help="quality keep CSV/Parquet")
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="输出目录（须 …/{source}_{batch}/02_sample/）",
    )
    parser.add_argument("-n", "--sample-size", type=int, default=None,
                        help="手动指定样本量（不指定则用公式计算）")
    parser.add_argument("--confidence", type=int, default=95, choices=[90, 95, 99],
                        help="置信度 (默认: 95)")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="误差范围 (默认: 0.05 即 ±5%%)")
    parser.add_argument("--prop", type=float, default=0.5,
                        help="预估比例 p (默认: 0.5 即最保守估计)；旧名 --p 仍可用")
    parser.add_argument("--p", type=float, dest="prop", help=argparse.SUPPRESS)
    parser.add_argument("--stratify", action="store_true",
                        help="启用分层抽样（按 keyword 分层）")
    parser.add_argument("--seed", type=int, default=42,
                        help="DuckDB SAMPLE REPEATABLE 种子")
    parser.add_argument("--name-tag", default=None,
                        help="输出文件名前缀（默认: 从批次路径推断品类，如 exo_service；"
                             "推断不到则用输入文件名 stem）")
    parser.add_argument("--progress", action="store_true",
                        help="启用 DuckDB progress bar")
    parser.add_argument("--no-log", action="store_true",
                        help="不写 项目记录.md / run_log.md（冒烟测试用）")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        return 1

    out_dir = require_output_dir(args.output_dir)
    warn_outside_batch(out_dir, log_fn=core_log)

    stats = build_sample(
        args.input, out_dir,
        sample_size=args.sample_size,
        confidence=args.confidence,
        margin=args.margin,
        prop=args.prop,
        stratify=args.stratify,
        seed=args.seed,
        name_tag=args.name_tag,
        progress=args.progress,
    )

    _print_summary(stats)

    mark_done(out_dir, "sample",
              samples=stats["sample_size"], total=stats["total_rows"],
              confidence=stats["confidence"], margin=stats["margin"],
              margin_effective=stats["margin_effective"],
              stratify=stats["stratify"], seed=stats["seed"],
              elapsed_sec=stats["elapsed_sec"])
    if args.no_log:
        return 0

    write_run_log(
        "sample", args.input, out_dir,
        stats=stats,
        command=f"{PROG} {args.input} -o {out_dir}"
                + (" --stratify" if args.stratify else "")
                + (f" -n {args.sample_size}" if args.sample_size else ""),
    )
    if maybe_update_stage(
        out_dir,
        "sample",
        paths={"parquet": stats["sample_parquet"], "csv": stats["sample_csv"]},
        stats={
            "total_rows": stats["total_rows"],
            "sample_size": stats["sample_size"],
            "seed": stats["seed"],
        },
    ):
        log("manifest 已更新 stage=sample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

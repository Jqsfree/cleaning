#!/usr/bin/env python3
"""
tools/batch_ops/merge_csvs.py — 多 CSV 合并去重 → Bronze

用途：
  同一品类常有多次导出、且汇总表与子场景表互相重叠。本脚本把散落的 CSV
  合并成一个去重后的 Bronze 文件落 ``raw/{category}/``，供
  ``pipeline/01_quality.py`` 初筛（初筛仍会再做一次批内去重，属幂等）。

去重口径：
  按 ``--dedup-key``（默认 ``video_id``）保留第一条，排序键为「文件名, rowid」，
  保证同一输入必得同一输出（可复现）。历史外部合并是「先到先得」，此处显式排序。

命名规范：
  ``{中文品类名}_merged_{batch}.csv``（中文名见 ``core.category_labels``）；
  ``--stem`` 可覆盖。产物同时落 ``合并记录.md`` 并追加 ``项目记录.md``。

用法:
  PYTHONPATH=02_脚本 .venv/bin/python3 02_脚本/tools/batch_ops/merge_csvs.py \\
      /Users/muse/data/exo/商业服务合集 -o raw/exo_service --batch 0813
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import duckdb

_SCRIPT = Path(__file__).resolve().parents[2]
_REPO = _SCRIPT.parent
sys.path.insert(0, str(_SCRIPT))

from core.batch_layout import infer_category  # noqa: E402
from core.category_labels import category_label  # noqa: E402
from core.log import log as core_log  # noqa: E402
from core.sop import print_banner, write_run_log  # noqa: E402
from core.sql_builder import sql_escape  # noqa: E402

OP_NAME = "merged"
_MACOSX_DIR = "__MACOSX"
_MACOSX_PREFIX = "._"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def discover_csvs(
    inputs: Sequence[Path],
    *,
    log_fn: Callable[[str], None] | None = None,
) -> list[Path]:
    """展开输入为 CSV 列表（目录递归取 ``*.csv``），排除 macOS 资源叉。

    - 目录：``glob('**/*.csv')``，跳过 ``__MACOSX/`` 与 ``._*``
      （AirDrop/压缩包常见，内容是 AppleDouble 元数据，非数据）
    - 文件：原样保留（不判扩展名，允许显式传入非 .csv 以便兼容）

    返回按路径排序的绝对路径列表；重复项去重。
    """
    logger = log_fn or _log
    found: list[Path] = []
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_dir():
            hits = [
                Path(h)
                for h in glob.glob(str(p / "**" / "*.csv"), recursive=True)
                if _MACOSX_DIR not in Path(h).parts
                and not Path(h).name.startswith(_MACOSX_PREFIX)
            ]
            logger(f"目录 {p} → {len(hits)} 个 CSV")
            found.extend(hits)
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(f"输入不存在: {p}")

    # 去重（保留顺序无关，最终排序）
    uniq = sorted({str(p.resolve()): p.resolve() for p in found}.values())
    if not uniq:
        raise FileNotFoundError("未找到任何 CSV 输入")
    return uniq


def source_label(inputs: Sequence[Path]) -> str:
    """源名称兜底：目录输入取目录名（如 ``商业服务合集``），文件输入取父目录名。"""
    for raw in inputs:
        p = Path(raw).expanduser()
        name = p.name if p.is_dir() else p.parent.name
        if name:
            return name
    return ""


def resolve_stem(
    out_dir: Path,
    *,
    stem: str | None = None,
    batch: str = "",
    fallback: str = "",
) -> str:
    """输出文件名前缀：``{中文品类名}_merged_{batch}``。

    ``--stem`` 优先；推断不出品类时退回 ``fallback``（源合集名），
    再不行才用裸 ``merged``，避免出现 ``merged_merged_0915`` 这类叠词。
    """
    if stem:
        return stem
    label = category_label(infer_category(out_dir)) or fallback
    parts = [p for p in (label, OP_NAME, batch) if p]
    return "_".join(parts) or OP_NAME


def _csv_list_sql(paths: Sequence[Path]) -> str:
    """DuckDB 多文件 reader 的路径列表字面量。"""
    return "[" + ", ".join(f"'{sql_escape(str(p))}'" for p in paths) + "]"


def merge_csvs(
    csvs: Sequence[Path],
    out_csv: Path,
    *,
    dedup_key: str = "video_id",
    keep_source_filename: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """合并 + 去重，写 ``out_csv``，返回统计 dict（含 per_file 明细）。

    ``union_by_name=true`` 使各文件列不完全一致时按列名对齐（缺列补 NULL），
    避免因多导出的 schema 漂移直接失败。
    """
    logger = log_fn or _log
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    work = out_csv.parent / ".merge.duckdb"
    tmp = out_csv.parent / ".duckdb_tmp"
    if work.exists():
        work.unlink()
    tmp.mkdir(exist_ok=True)

    db = duckdb.connect(str(work))
    db.execute("SET memory_limit='4GB'")
    db.execute(f"SET temp_directory='{sql_escape(str(tmp))}'")

    t0 = time.perf_counter()
    try:
        reader = (
            f"read_csv_auto({_csv_list_sql(csvs)}, header=true, all_varchar=true, "
            f"sample_size=-1, ignore_errors=true, union_by_name=true, filename=true)"
        )
        db.execute(f"CREATE TABLE raw AS SELECT * FROM {reader}")
        n_raw = db.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
        cols = [c[0] for c in db.execute("SELECT * FROM raw LIMIT 0").description]
        logger(f"载入 {len(csvs)} 个文件: {n_raw:,} 行 / {len(cols)} 列")

        if dedup_key not in cols:
            raise ValueError(f"缺少去重键列 {dedup_key!r}；实际列: {cols}")

        # 确定性排序：文件名次第 → rowid 兜底（同文件内同 key 重复也能复现）
        db.execute(f"""
            CREATE TABLE ranked AS
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY {dedup_key} ORDER BY filename, rowid
            ) AS _rn
            FROM raw
        """)

        per_file = db.execute("""
            SELECT regexp_extract(filename, '[^/]+$') AS source_file,
                   COUNT(*)                            AS rows_in,
                   SUM(CASE WHEN _rn = 1 THEN 1 ELSE 0 END) AS rows_kept
            FROM ranked
            GROUP BY filename
            ORDER BY rows_in DESC, source_file ASC
        """).fetchall()

        n_kept = int(sum(int(r[2]) for r in per_file))
        n_removed = n_raw - n_kept

        # ORDER BY 必须显式给全序：DuckDB 并行扫描的行序不保证稳定，
        # 不排序会让「同一输入」产出字节不同的文件（回归：test_merge_is_deterministic）。
        # 按文件名次第 + 去重键排序，既确定又可追溯到源文件。
        exclude = "(_rn)" if keep_source_filename else "(_rn, filename)"
        db.execute(
            f"COPY (SELECT * EXCLUDE {exclude} FROM ranked WHERE _rn = 1 "
            f"ORDER BY filename, {dedup_key}) TO '{sql_escape(str(out_csv))}' "
            f"(FORMAT CSV, HEADER true)"
        )

        # 总时长按「产物」口径统计（去重后），对齐历史合并记录
        dur = 0.0
        if "duration_seconds" in cols:
            dur = float(db.execute(
                "SELECT COALESCE(SUM(TRY_CAST(duration_seconds AS DOUBLE)), 0) "
                "FROM ranked WHERE _rn = 1"
            ).fetchone()[0])

        n_dup_keys = db.execute(
            f"SELECT COUNT(*) FROM (SELECT {dedup_key} FROM raw "
            f"GROUP BY {dedup_key} HAVING COUNT(*) > 1)"
        ).fetchone()[0]
    finally:
        db.close()
        for p in [work, work.with_suffix(".duckdb.wal")]:
            p.unlink(missing_ok=True)
        try:
            for f in tmp.glob("*"):
                f.unlink(missing_ok=True)
            tmp.rmdir()
        except OSError:
            pass

    elapsed = time.perf_counter() - t0
    logger(
        f"合并完成: {n_raw:,} → {n_kept:,} (移除 {n_removed:,})  |  {elapsed:.1f}s"
    )
    return {
        "files": [str(p) for p in csvs],
        "n_files": len(csvs),
        "total_rows": n_raw,
        "keep": n_kept,
        "removed": n_removed,
        "removed_pct": round(n_removed / max(1, n_raw) * 100, 2),
        "n_duplicate_keys": n_dup_keys,
        "total_duration_hours": round(float(dur) / 3600, 1),
        # 产物实际列（read_csv_auto 的 filename 辅助列默认不落盘）
        "columns": [
            c for c in cols
            if c != "filename" or keep_source_filename
        ],
        "per_file": [
            {
                "source_file": r[0],
                "rows_in": int(r[1]),
                "rows_kept": int(r[2]),
                "rows_removed": int(r[1]) - int(r[2]),
            }
            for r in per_file
        ],
        "out_csv": str(out_csv),
        "elapsed_sec": round(elapsed, 1),
        "dedup_key": dedup_key,
    }


def render_merge_record(summary: dict[str, Any], *, out_dir: Path, batch: str) -> str:
    """生成 ``合并记录.md`` 正文（对齐历史合并记录表格口径）。"""
    cat = infer_category(out_dir) or "(未识别)"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    src_dirs = sorted({str(Path(f).parent) for f in summary["files"]})
    lines = [
        f"# 合并记录 — {cat} / {batch}",
        "",
        f"- 合并时间：{ts}",
        f"- 源目录（{len(src_dirs)} 个）：",
    ]
    lines += [f"  - `{d}`" for d in src_dirs]
    lines += [
        f"- 源 CSV 文件数：{summary['n_files']}",
        f"- 去重键：`{summary['dedup_key']}`（按文件名次第保留首条，可复现）",
        f"- 原始行数：{summary['total_rows']:,}，去重后：{summary['keep']:,}，"
        f"去重移除：{summary['removed']:,}（{summary['removed_pct']}%）",
        f"- 涉及重复键：{summary['n_duplicate_keys']:,} 个",
        f"- 总时长（去重后）：{summary['total_duration_hours']:,} 小时",
        f"- 输出：`{summary['out_csv']}`",
        f"- 列数：{len(summary['columns'])}",
        "",
        "## 源文件明细",
        "",
        "| # | 源文件 | 原始行 | 保留行 | 移除 |",
        "|---|--------|--------|--------|------|",
    ]
    for i, r in enumerate(summary["per_file"], 1):
        lines.append(
            f"| {i} | `{r['source_file']}` | {r['rows_in']:,} | "
            f"{r['rows_kept']:,} | {r['rows_removed']:,} |"
        )
    lines += [
        "",
        "## 备注",
        "",
        f"- 去重后每行 ``{summary['dedup_key']}`` 唯一；保留行归属见上表「保留行」列。",
        "- 汇总表（如 `*场景总*`、`*覆盖*`）与子场景表大面积重叠，重复主要来自此。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="多 CSV 合并去重 → Bronze（raw/{category}/）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  02_脚本/tools/batch_ops/merge_csvs.py /Users/muse/data/exo/商业服务合集 \\\n"
            "      -o raw/exo_service --batch 0813\n"
        ),
    )
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="源目录（递归找 *.csv）或显式 CSV 文件")
    parser.add_argument("-o", "--out-dir", required=True, type=Path,
                        help="输出目录，约定 raw/{category}/")
    parser.add_argument("--batch", default=time.strftime("%m%d"),
                        help="批次号（默认今天 MMDD）")
    parser.add_argument("--stem", default=None,
                        help="输出文件名前缀（默认 {中文品类名}_merged_{batch}）")
    parser.add_argument("--dedup-key", default="video_id", help="去重键列（默认 video_id）")
    parser.add_argument("--keep-source-filename", action="store_true",
                        help="输出保留 filename 列（默认丢弃，保持源 schema）")
    parser.add_argument("--no-log", action="store_true",
                        help="不写 项目记录.md / run_log.md（冒烟测试用）")
    args = parser.parse_args()

    try:
        csvs = discover_csvs(args.inputs)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", flush=True)
        return 1

    print_banner("merge", category=infer_category(args.out_dir))
    if infer_category(args.out_dir) is None:
        core_log(
            f"输出目录推断不出品类（约定 raw/{{category}}/，如 raw/exo_service/）: "
            f"{args.out_dir}",
            level="WARN",
        )

    stem = resolve_stem(
        args.out_dir, stem=args.stem, batch=args.batch,
        fallback=source_label(args.inputs),
    )
    out_csv = args.out_dir / f"{stem}.csv"

    try:
        summary = merge_csvs(
            csvs, out_csv,
            dedup_key=args.dedup_key,
            keep_source_filename=args.keep_source_filename,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}", flush=True)
        return 1

    record_path = args.out_dir / "合并记录.md"
    record_path.write_text(
        render_merge_record(summary, out_dir=args.out_dir, batch=args.batch),
        encoding="utf-8",
    )

    print()
    print("=" * 62)
    print(f"  合并 完成（Bronze）")
    print("=" * 62)
    print(f"  源文件:     {summary['n_files']:>12,}")
    print(f"  原始行:     {summary['total_rows']:>12,}")
    print(f"  去重后:     {summary['keep']:>12,}  (移除 {summary['removed']:,})")
    print(f"  总时长:     {summary['total_duration_hours']:>12,.1f} h")
    print(f"  耗时:       {summary['elapsed_sec']:>12,.1f} s")
    print(f"  产物:       {out_csv}")
    print(f"              {record_path}")
    print("=" * 62)
    print()

    if args.no_log:
        return 0

    write_run_log(
        "merge",
        " + ".join(str(p) for p in args.inputs),
        str(args.out_dir),
        stats={
            "n_files": summary["n_files"],
            "total_rows": summary["total_rows"],
            "keep": summary["keep"],
            "removed": summary["removed"],
            "total_duration_hours": summary["total_duration_hours"],
            "out_csv": summary["out_csv"],
            "record": str(record_path),
        },
        command=(
            f"tools/batch_ops/merge_csvs.py {' '.join(str(p) for p in args.inputs)} "
            f"-o {args.out_dir} --batch {args.batch}"
        ),
        category=infer_category(args.out_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

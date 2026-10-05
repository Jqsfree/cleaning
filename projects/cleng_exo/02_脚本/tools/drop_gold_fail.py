#!/usr/bin/env python3
"""按人工金标 F 从 keep 丢弃 video_id（增量，不全量 MiniLM 打分）。

口径:
  - 只丢 qc_result 确定为 F 的 video_id（T/U 不动）
  - 多批金标合并：同 video_id 若任一批为 F 则丢（冲突时 F 优先）
  - 用于「新人标入库后立刻回滤 keep」；重训 MiniLM 另跑 experiments/

用法:
  PYTHONPATH=02_脚本 .venv/bin/python3 02_脚本/tools/drop_gold_fail.py \\
    data/runs/exo_cook/machine_0916/06_tools/text_gov_v05/烹饪教学_merged_0916_ml_keep_0918.csv \\
    --qc-snapshot ~/Downloads/humen-烹饪餐饮/*.csv \\
    -o data/runs/exo_cook/machine_0916/06_tools/text_gov_v05/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

_SCRIPT = Path(__file__).resolve().parent
_REPO = _SCRIPT.parent.parent
sys.path.insert(0, str(_SCRIPT.parent))

from core.sql_builder import sql_escape  # noqa: E402


def _norm_lab(x: object) -> str:
    s = str(x or "").strip().upper()
    if s.startswith("T"):
        return "T"
    if s.startswith("F"):
        return "F"
    return ""


def load_fail_ids(qc_paths: list[Path]) -> pd.DataFrame:
    """合并金标；返回去重后的 F 行（video_id, batches）。"""
    rows: list[pd.DataFrame] = []
    for p in qc_paths:
        df = pd.read_csv(p, dtype=str)
        if "video_id" not in df.columns or "qc_result" not in df.columns:
            raise ValueError(f"缺少 video_id/qc_result: {p}")
        lab = df["qc_result"].map(_norm_lab)
        sub = df.loc[lab == "F", ["video_id"]].copy()
        sub["video_id"] = sub["video_id"].astype(str).str.strip()
        sub = sub[sub["video_id"] != ""]
        sub["batch_file"] = p.name
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["video_id", "batches", "n_batches"])
    all_f = pd.concat(rows, ignore_index=True)
    g = (
        all_f.groupby("video_id", as_index=False)
        .agg(batches=("batch_file", lambda s: "|".join(sorted(set(s)))), n_batches=("batch_file", "nunique"))
    )
    return g


def drop_fail_from_keep(
    keep_csv: Path,
    fail_ids: pd.DataFrame,
    *,
    out_dir: Path,
    stem: str | None = None,
) -> dict:
    """ANTI JOIN 丢 F；写 keep/drop + summary。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = stem or keep_csv.stem
    keep_out = out_dir / f"{base}_gold_f_filt.csv"
    drop_out = out_dir / f"{base}_gold_f_drop.csv"
    fail_list = out_dir / f"{base}_gold_f_ids.csv"
    fail_ids.to_csv(fail_list, index=False)

    con = duckdb.connect()
    con.execute("SET threads=2")
    fail_path = sql_escape(str(fail_list.resolve()))
    keep_path = sql_escape(str(keep_csv.resolve()))
    keep_out_s = sql_escape(str(keep_out.resolve()))
    drop_out_s = sql_escape(str(drop_out.resolve()))

    n0, h0 = con.execute(
        f"SELECT count(*), COALESCE(SUM(TRY_CAST(duration_seconds AS DOUBLE)),0)/3600 "
        f"FROM read_csv_auto('{keep_path}', header=true, all_varchar=true)"
    ).fetchone()
    con.execute(
        f"CREATE TEMP TABLE gold_f AS SELECT CAST(video_id AS VARCHAR) video_id "
        f"FROM read_csv_auto('{fail_path}', header=true, all_varchar=true)"
    )
    con.execute(
        f"COPY (SELECT r.* FROM read_csv_auto('{keep_path}', header=true, all_varchar=true) r "
        f"ANTI JOIN gold_f g USING (video_id)) TO '{keep_out_s}' (HEADER, DELIMITER ',')"
    )
    con.execute(
        f"COPY (SELECT r.* FROM read_csv_auto('{keep_path}', header=true, all_varchar=true) r "
        f"INNER JOIN gold_f g USING (video_id)) TO '{drop_out_s}' (HEADER, DELIMITER ',')"
    )
    n1, h1 = con.execute(
        f"SELECT count(*), COALESCE(SUM(TRY_CAST(duration_seconds AS DOUBLE)),0)/3600 "
        f"FROM read_csv_auto('{keep_out_s}', header=true, all_varchar=true)"
    ).fetchone()
    nd, hd = con.execute(
        f"SELECT count(*), COALESCE(SUM(TRY_CAST(duration_seconds AS DOUBLE)),0)/3600 "
        f"FROM read_csv_auto('{drop_out_s}', header=true, all_varchar=true)"
    ).fetchone()

    summary = {
        "keep_in": str(keep_csv),
        "n_gold_f_ids": int(len(fail_ids)),
        "before": {"n": int(n0), "hours": round(float(h0), 1)},
        "after": {"n": int(n1), "hours": round(float(h1), 1)},
        "dropped": {"n": int(nd), "hours": round(float(hd), 1)},
        "keep_out": str(keep_out),
        "drop_out": str(drop_out),
        "fail_list": str(fail_list),
        "note": "仅丢人工 F video_id；未全量 MiniLM 打分",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / f"{base}_gold_f_filt_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="按金标 F 增量丢 keep（不全量打分）")
    ap.add_argument("keep", type=Path, help="当前 keep CSV")
    ap.add_argument(
        "--qc-snapshot",
        action="append",
        type=Path,
        dest="qc_snapshots",
        required=True,
        help="人工 qc_result.csv（可多次）",
    )
    ap.add_argument("-o", "--output-dir", type=Path, required=True)
    ap.add_argument("--stem", default=None, help="输出文件名前缀（默认=keep stem）")
    ap.add_argument("--replace-keep", action="store_true", help="用滤后结果覆盖输入 keep")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    qc = [p.expanduser().resolve() for p in args.qc_snapshots]
    missing = [str(p) for p in qc if not p.is_file()]
    if missing:
        print(f"[ERROR] 金标不存在: {missing}")
        return 1
    if not args.keep.is_file():
        print(f"[ERROR] keep 不存在: {args.keep}")
        return 1

    fail = load_fail_ids(qc)
    print(f"[info] 金标 F 去重 video_id: {len(fail):,}")
    summary = drop_fail_from_keep(
        args.keep.resolve(), fail, out_dir=args.output_dir.resolve(), stem=args.stem
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.replace_keep:
        import shutil

        shutil.copy2(summary["keep_out"], args.keep)
        print(f"[info] 已覆盖 keep → {args.keep}")

    if not args.no_log:
        try:
            from core.sop import write_run_log

            write_run_log(
                stage="tools/drop_gold_fail",
                input_path=str(args.keep),
                output_dir=str(args.output_dir),
                stats=summary,
                command="drop_gold_fail.py",
                category="exo_cook",
            )
        except Exception as e:  # noqa: BLE001 — 留痕失败不阻断
            print(f"[WARN] write_run_log: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

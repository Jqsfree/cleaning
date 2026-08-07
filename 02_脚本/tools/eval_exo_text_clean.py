#!/usr/bin/env python3
"""
tools/eval_exo_text_clean.py — 评估 exo 标题规则清洗效果（无 LLM）

对比 quality / keep / drop / 抽样：
  - 时长（小时 / 万小时）
  - 农采正信号 vs certain-noise 残留（标题正则代理）
  - drop 归因分布、可能误杀样例
  - keep/抽样残留噪声样例（人工扫一眼）

用法:
  02_脚本/tools/eval_exo_text_clean.py -o data/runs/exo/machine_0807/ --run title02
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.batch_layout import require_output_dir
from core.rules_loader import load_blacklist_individual

# 报告用代理（≠ 生产 blacklist；用于 keep 残留 / drop 纯度粗估）
AGRI_RE = (
    r"harvest|harvesting|picking|farm|farming|agriculture|vegetable|fruit|"
    r"tomato|potato|onion|cabbage|carrot|corn|apple|grape|strawberry|"
    r"greenhouse|orchard|spinach|cucumber|eggplant|workers|manual"
)
NOISE_RE = (
    r"gameplay|minecraft|roblox|fortnite|let.?s\s*play|farming\s*simulator|"
    r"recipe|cooking|interview|podcast|talk\s*show|tedx|ted\s*talk|"
    r"makeup\s*tutorial|skincare|online\s*course|webinar|masterclass|"
    r"tutorial\s*video|piano\s*lesson|guitar\s*lesson|full\s*match|"
    r"\bnba\b|\bnfl\b|ufc|street\s*fight|official\s*music|music\s*video|"
    r"official\s*trailer|anime|cartoon"
)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _reader(path: Path) -> str:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return f"read_parquet('{path}')"
    return f"read_csv_auto('{path}', header=true, all_varchar=true)"


def _find_one(dir_path: Path, patterns: list[str], *, kind: str = "any") -> Path | None:
    if not dir_path.is_dir():
        return None
    for pat in patterns:
        hits = sorted(dir_path.glob(pat))
        if kind == "keep":
            hits = [h for h in hits if "_drop" not in h.name.lower()]
        elif kind == "drop":
            hits = [h for h in hits if "_drop" in h.name.lower()]
        elif kind == "quality":
            hits = [h for h in hits if "quality_drop" not in h.name.lower() and "_drop" not in h.name.lower()]
        if hits:
            return hits[-1]
    return None


def resolve_paths(batch: Path, run: str) -> dict[str, Path]:
    quality_dir = batch / "01_quality"
    clean_dir = batch / "05_clean" / f"run_{run}"
    sample_dir = batch / "02_sample" / f"run_{run}"
    quality = _find_one(quality_dir, ["*quality*.csv", "*quality*.parquet"], kind="quality")
    keep = _find_one(clean_dir, ["*_keep.parquet", "*_keep.csv"], kind="keep")
    drop = _find_one(clean_dir, ["*_drop.parquet", "*_drop.csv"], kind="drop")
    sample = _find_one(sample_dir, ["*sample*.csv", "*sample*.parquet"], kind="any")
    missing = [k for k, v in {
        "quality": quality, "keep": keep, "drop": drop,
    }.items() if v is None]
    if missing:
        raise SystemExit(f"[ERROR] 缺少文件: {missing} under {batch} run={run}")
    return {"quality": quality, "keep": keep, "drop": drop, "sample": sample}


def duration_stats(con: duckdb.DuckDBPyConnection, path: Path) -> dict:
    r = con.execute(f"""
        SELECT
          COUNT(*) AS n,
          SUM(TRY_CAST(duration_seconds AS DOUBLE)) / 3600.0 AS hrs,
          AVG(TRY_CAST(duration_seconds AS DOUBLE)) AS avg_s,
          MEDIAN(TRY_CAST(duration_seconds AS DOUBLE)) AS med_s
        FROM {_reader(path)}
        WHERE TRY_CAST(duration_seconds AS DOUBLE) IS NOT NULL
          AND TRY_CAST(duration_seconds AS DOUBLE) > 0
    """).fetchone()
    return {
        "n": int(r[0]),
        "hours": round(float(r[1] or 0), 1),
        "wan_hours": round(float(r[1] or 0) / 10000, 2),
        "avg_sec": round(float(r[2] or 0), 1),
        "median_sec": round(float(r[3] or 0), 1),
    }


def signal_stats(con: duckdb.DuckDBPyConnection, path: Path) -> dict:
    r = con.execute(f"""
        SELECT
          COUNT(*) AS n,
          SUM(CASE WHEN regexp_matches(lower(COALESCE(title,'')), '{AGRI_RE}', 'i')
                   THEN 1 ELSE 0 END) AS agri,
          SUM(CASE WHEN regexp_matches(lower(COALESCE(title,'')), '{NOISE_RE}', 'i')
                   THEN 1 ELSE 0 END) AS noise
        FROM {_reader(path)}
    """).fetchone()
    n = max(int(r[0]), 1)
    return {
        "n": int(r[0]),
        "agri_rate": round(int(r[1]) / n, 4),
        "noise_rate": round(int(r[2]) / n, 4),
        "agri_n": int(r[1]),
        "noise_n": int(r[2]),
    }


def drop_breakdown(con: duckdb.DuckDBPyConnection, path: Path) -> list[dict]:
    rows = con.execute(f"""
        SELECT drop_step, COUNT(*) AS c,
               SUM(TRY_CAST(duration_seconds AS DOUBLE))/3600.0 AS hrs
        FROM {_reader(path)}
        GROUP BY 1 ORDER BY c DESC
    """).fetchall()
    return [
        {"drop_step": a, "n": int(b), "hours": round(float(c or 0), 1)}
        for a, b, c in rows
    ]


def top_reasons(con: duckdb.DuckDBPyConnection, path: Path, limit: int = 20) -> list[dict]:
    rows = con.execute(f"""
        SELECT lower(COALESCE(drop_reason,'')) AS r, COUNT(*) AS c
        FROM {_reader(path)}
        GROUP BY 1 ORDER BY c DESC LIMIT {limit}
    """).fetchall()
    return [{"reason": a or "(empty)", "n": int(b)} for a, b in rows]


def sample_titles(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    where: str,
    limit: int = 15,
    seed: int = 42,
) -> list[str]:
    rows = con.execute(f"""
        SELECT COALESCE(title,'') AS title
        FROM {_reader(path)}
        WHERE {where}
        ORDER BY hash(COALESCE(video_id, title) || '{seed}')
        LIMIT {limit}
    """).fetchall()
    return [r[0][:140] for r in rows]


def verdict(quality_sig: dict, keep_sig: dict, drop_sig: dict) -> list[str]:
    notes = []
    noise_reduction = quality_sig["noise_rate"] - keep_sig["noise_rate"]
    notes.append(
        f"certain-noise 标题残留: {quality_sig['noise_rate']:.1%} → "
        f"{keep_sig['noise_rate']:.1%}（↓{noise_reduction:.1%}）"
    )
    if keep_sig["noise_rate"] <= 0.015:
        notes.append("规则层对已知硬噪声清除较好（残留 ≤1.5%）。")
    else:
        notes.append("规则层残留噪声仍偏高，建议扩 blacklist 或抽样 text QC。")
    if drop_sig["noise_rate"] >= 0.5:
        notes.append(
            f"drop 集噪声代理命中 {drop_sig['noise_rate']:.0%}，丢弃归因大体可解释。"
        )
    else:
        notes.append("drop 集噪声代理命中偏低，可能误杀偏多，需抽查 drop。")
    if keep_sig["agri_rate"] < 0.45:
        notes.append(
            f"keep 农采标题词仅 {keep_sig['agri_rate']:.0%}："
            "文本规则清噪声有效，但目标纯度仍低，需 text QC / 画面，不能当交付合格率。"
        )
    else:
        notes.append("keep 农采标题词占比较高，可进入 text QC 校准。")
    return notes


def write_report(out: Path, data: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# exo 文本清洗评估 — {data['run']}",
        "",
        f"- batch: `{data['batch']}`",
        f"- ts: {data['ts']}",
        "",
        "## 时长",
        "",
        "| frame | rows | hours | 万小时 | median_s |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("quality", "keep", "drop", "sample"):
        d = data["duration"].get(name)
        if not d:
            continue
        lines.append(
            f"| {name} | {d['n']:,} | {d['hours']:,.1f} | {d['wan_hours']:.2f} | {d['median_sec']:.0f} |"
        )
    lines += ["", "## 标题信号（代理正则）", ""]
    lines.append("| frame | agri_rate | noise_rate |")
    lines.append("|---|---:|---:|")
    for name in ("quality", "keep", "drop", "sample"):
        s = data["signal"].get(name)
        if not s:
            continue
        lines.append(f"| {name} | {s['agri_rate']:.1%} | {s['noise_rate']:.1%} |")
    lines += ["", "## Drop 归因", ""]
    for row in data["drop_steps"]:
        lines.append(f"- {row['drop_step']}: {row['n']:,} ({row['hours']:,.1f} h)")
    lines += ["", "### Top drop_reason", ""]
    for row in data["top_reasons"]:
        lines.append(f"- {row['n']:,}  `{row['reason']}`")
    lines += ["", "## 结论", ""]
    for n in data["verdict"]:
        lines.append(f"- {n}")
    lines += ["", "## Keep 残留噪声样例", ""]
    for t in data["examples"]["keep_noise"]:
        lines.append(f"- {t}")
    if not data["examples"]["keep_noise"]:
        lines.append("- （代理正则未命中）")
    lines += ["", "## Drop 中偏农标题样例（可能误杀/边界）", ""]
    for t in data["examples"]["drop_agri"]:
        lines.append(f"- {t}")
    lines += ["", "## 抽样标题扫一眼", ""]
    for t in data["examples"]["sample"]:
        lines.append(f"- {t}")
    path = out / "text_clean_eval.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "text_clean_eval.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="评估 exo 文本规则清洗")
    ap.add_argument("-o", "--output", required=True, help="批次根目录")
    ap.add_argument("--run", default="title02")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    batch = Path(require_output_dir(args.output))
    paths = resolve_paths(batch, args.run)
    con = duckdb.connect()

    duration = {}
    signal = {}
    for name, path in paths.items():
        if path is None:
            continue
        duration[name] = duration_stats(con, path)
        signal[name] = signal_stats(con, path)
        _log(
            f"{name}: {duration[name]['wan_hours']:.2f} 万小时 | "
            f"agri={signal[name]['agri_rate']:.1%} noise={signal[name]['noise_rate']:.1%}"
        )

    drop_steps = drop_breakdown(con, paths["drop"])
    reasons = top_reasons(con, paths["drop"])
    examples = {
        "keep_noise": sample_titles(
            con, paths["keep"],
            f"regexp_matches(lower(COALESCE(title,'')), '{NOISE_RE}', 'i')",
            seed=args.seed,
        ),
        "drop_agri": sample_titles(
            con, paths["drop"],
            f"regexp_matches(lower(COALESCE(title,'')), '{AGRI_RE}', 'i')",
            seed=args.seed,
        ),
        "sample": sample_titles(
            con, paths["sample"], "TRUE", limit=20, seed=args.seed
        ) if paths.get("sample") else [],
    }
    v = verdict(signal["quality"], signal["keep"], signal["drop"])

    # 规则覆盖：blacklist 类别数
    bl = load_blacklist_individual(Path("02_脚本/categories/exo/rules"))
    data = {
        "batch": str(batch),
        "run": args.run,
        "paths": {k: str(v) if v else None for k, v in paths.items()},
        "ts": datetime.now(timezone.utc).isoformat(),
        "blacklist_title_categories": [r["category"] for r in bl.get("title_pass2", [])],
        "duration": duration,
        "signal": signal,
        "drop_steps": drop_steps,
        "top_reasons": reasons,
        "examples": examples,
        "verdict": v,
    }
    out = batch / "03_qc" / "analysis" / args.run
    report = write_report(out, data)
    _log(f"报告: {report}")
    for line in v:
        _log(f"结论: {line}")


if __name__ == "__main__":
    main()

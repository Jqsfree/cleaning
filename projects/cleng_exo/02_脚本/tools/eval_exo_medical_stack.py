#!/usr/bin/env python3
"""离线评估 exo_medical 过滤栈：封面 CV + 题材 veto + 人体框。

在人工锚点（R1+R2）上报告 remain 合格率，不写批次数据。

  02_脚本/tools/eval_exo_medical_stack.py \\
    --human-r1 /home/jqs/tmp/exo医疗_614991c7_qc_result.csv \\
    --human-r2 /home/jqs/tmp/exo医疗2_c86e393b_qc_result.csv \\
    -o data/runs/exo_medical/machine_0813/06_tools/stack_eval/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from core.rules_loader import load_blacklist  # noqa: E402
from core.sql_builder import sql_escape  # noqa: E402

DEFAULT_COVER_SCORED = (
    _SCRIPT_DIR.parent
    / "data/runs/exo_medical/machine_0813/06_tools/cover_text"
    / "exo医疗场景_e15c3ad7_records_clean_0814_cover_text_scored.csv"
)
RULES_DIR = _SCRIPT_DIR / "categories/exo_medical/rules"


def load_human(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        df["human_source"] = p.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError("未找到人工标注文件")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates("video_id", keep="last")
    out["label"] = out["qc_result"].astype(str).str.strip().str.upper()
    out["is_pass"] = out["label"].eq("T")
    return out


def build_search_text(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ("title", "channel", "keyword", "description"):
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str))
        else:
            parts.append(pd.Series("", index=df.index))
    text = parts[0]
    for p in parts[1:]:
        text = text + " " + p
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def apply_blacklist(df: pd.DataFrame, rules_dir: Path) -> pd.Series:
    rules = load_blacklist(rules_dir)
    pass2 = sql_escape(rules["pass2"])
    r2 = sql_escape(rules["r2"])
    work = df.copy()
    work["search_text"] = build_search_text(work)
    con = duckdb.connect()
    con.register("raw", work[["video_id", "search_text"]])
    hits = con.execute(f"""
        SELECT video_id,
               regexp_matches(search_text, '{pass2}', 'i') AS pass2_hit,
               regexp_matches(search_text, '{r2}', 'i') AS r2_hit
        FROM raw
    """).df()
    return hits["pass2_hit"] | hits["r2_hit"]


def eval_remain(
    df: pd.DataFrame,
    mask_drop: pd.Series,
    name: str,
) -> dict:
    remain = ~mask_drop
    n = int(len(df))
    n_drop = int(mask_drop.sum())
    n_rem = int(remain.sum())
    tp = int((mask_drop & ~df["is_pass"]).sum())
    fp = int((mask_drop & df["is_pass"]).sum())
    pass_rate_all = float(df["is_pass"].mean())
    pass_rate_rem = float(df.loc[remain, "is_pass"].mean()) if n_rem else 0.0
    return {
        "stack": name,
        "n": n,
        "n_drop": n_drop,
        "n_remain": n_rem,
        "catch_f": tp,
        "kill_t": fp,
        "drop_precision": tp / (n_drop + 1e-9),
        "pass_rate_all": round(pass_rate_all, 4),
        "pass_rate_remain": round(pass_rate_rem, 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="exo_medical 过滤栈离线评估")
    p.add_argument("--human-r1", type=Path, required=True)
    p.add_argument("--human-r2", type=Path, required=True)
    p.add_argument("--cover-scored", type=Path, default=DEFAULT_COVER_SCORED)
    p.add_argument("--person-features", type=Path, default=None)
    p.add_argument("--cover-threshold", type=float, default=0.8577)
    p.add_argument("--min-person-area", type=float, default=0.08)
    p.add_argument("-o", "--output-dir", required=True)
    args = p.parse_args()

    hum = load_human([args.human_r1, args.human_r2])
    scored = pd.read_csv(args.cover_scored, low_memory=False)
    cols = ["video_id", "p_text_heavy", "cover_text_action"]
    cols = [c for c in cols if c in scored.columns]
    df = hum.merge(scored[cols].drop_duplicates("video_id"), on="video_id", how="left")

    if args.person_features and args.person_features.exists():
        pf = pd.read_parquet(args.person_features)
        keep_cols = [
            c for c in pf.columns
            if c in {
                "video_id", "thumb_person_action", "thumb_person_reason",
                "thumb_person_max_area",
            }
        ]
        df = df.merge(pf[keep_cols].drop_duplicates("video_id"), on="video_id", how="left")
    else:
        df["thumb_person_action"] = "keep_error"
        df["thumb_person_max_area"] = float("nan")

    p_text = pd.to_numeric(df.get("p_text_heavy"), errors="coerce")
    cover_drop = p_text >= args.cover_threshold
    if "cover_text_action" in df.columns:
        cover_drop = cover_drop | df["cover_text_action"].eq("highconf_drop")

    topic_drop = apply_blacklist(df, RULES_DIR)
    person_drop = df["thumb_person_action"].eq("highconf_drop")
    if df["thumb_person_max_area"].notna().any():
        person_drop = person_drop | (
            pd.to_numeric(df["thumb_person_max_area"], errors="coerce")
            < args.min_person_area
        )

    stacks = [
        ("baseline", pd.Series(False, index=df.index)),
        ("cover_cv", cover_drop.fillna(False)),
        ("topic_veto", topic_drop.fillna(False)),
        ("person", person_drop.fillna(False)),
        ("cover+topic", cover_drop.fillna(False) | topic_drop.fillna(False)),
        (
            "cover+topic+person",
            cover_drop.fillna(False) | topic_drop.fillna(False) | person_drop.fillna(False),
        ),
    ]

    rows = [eval_remain(df, drop, name) for name, drop in stacks]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "n_human_unique": len(df),
        "n_pass": int(df["is_pass"].sum()),
        "n_fail": int((~df["is_pass"]).sum()),
        "baseline_pass_rate": float(df["is_pass"].mean()),
        "cover_threshold": args.cover_threshold,
        "min_person_area": args.min_person_area,
        "has_person_features": bool(args.person_features and args.person_features.exists()),
        "stacks": rows,
    }
    (out_dir / "stack_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(out_dir / "stack_eval.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

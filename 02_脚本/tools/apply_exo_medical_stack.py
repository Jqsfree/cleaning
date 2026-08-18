#!/usr/bin/env python3
"""合并 exo_medical 前置过滤：cover keep → 题材 blacklist → 人体框。

  02_脚本/tools/apply_exo_medical_stack.py \\
    data/runs/exo_medical/machine_0813/06_tools/cover_text/*_cover_text_keep.csv \\
    --person-features data/runs/exo_medical/machine_0813/06_tools/thumb_person/thumb_person_features.parquet \\
    -o data/runs/exo_medical/machine_0813/06_tools/stack_remain/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from core.rules_loader import load_blacklist  # noqa: E402
from core.sql_builder import sql_escape  # noqa: E402

RULES_DIR = _SCRIPT_DIR / "categories/exo_medical/rules"


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def build_search_text(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ("title", "channel", "keyword", "description"):
        parts.append(df.get(col, pd.Series("", index=df.index)).fillna("").astype(str))
    text = parts[0]
    for p in parts[1:]:
        text = text + " " + p
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def apply_blacklist_flags(df: pd.DataFrame) -> pd.Series:
    rules = load_blacklist(RULES_DIR)
    pass2 = sql_escape(rules["pass2"])
    r2 = sql_escape(rules["r2"])
    work = df.copy()
    work["search_text"] = build_search_text(work)
    con = duckdb.connect()
    con.register("raw", work[["video_id", "search_text"]])
    hits = con.execute(f"""
        SELECT video_id,
               regexp_matches(search_text, '{pass2}', 'i') AS topic_drop
        FROM raw
    """).df()
    merged = df[["video_id"]].merge(hits, on="video_id", how="left")
    return merged["topic_drop"].fillna(False)


def main() -> None:
    p = argparse.ArgumentParser(description="exo_medical 合并前置过滤栈")
    p.add_argument("input", help="cover_text_keep CSV")
    p.add_argument("-o", "--output-dir", required=True)
    p.add_argument("--person-features", type=Path, required=True)
    p.add_argument("--min-person-area", type=float, default=0.08)
    args = p.parse_args()

    df = read_table(Path(args.input))
    pf = pd.read_parquet(args.person_features)
    df = df.merge(
        pf.drop_duplicates("video_id"),
        on="video_id",
        how="left",
        suffixes=("", "_person"),
    )

    df["topic_drop"] = apply_blacklist_flags(df)
    area = pd.to_numeric(df.get("thumb_person_max_area"), errors="coerce")
    df["person_drop"] = (
        df.get("thumb_person_action", pd.Series("", index=df.index)).eq("highconf_drop")
        | (area < args.min_person_area)
    )
    df["stack_drop"] = df["topic_drop"] | df["person_drop"]
    df["stack_action"] = df["stack_drop"].map({True: "drop", False: "keep"})

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem.replace("_cover_text_keep", "")
    drop = df[df["stack_drop"]].copy()
    remain = df[~df["stack_drop"]].copy()
    remain_path = out / f"{stem}_stack_remain.csv"
    drop_path = out / f"{stem}_stack_drop.csv"
    remain.to_csv(remain_path, index=False)
    drop.to_csv(drop_path, index=False)

    summary = {
        "input": str(Path(args.input).resolve()),
        "n_input": int(len(df)),
        "n_drop": int(len(drop)),
        "n_remain": int(len(remain)),
        "topic_drop": int(df["topic_drop"].sum()),
        "person_drop": int(df["person_drop"].sum()),
        "min_person_area": args.min_person_area,
        "remain_path": str(remain_path.resolve()),
        "drop_path": str(drop_path.resolve()),
    }
    (out / "apply.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

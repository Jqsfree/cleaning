#!/usr/bin/env python3
"""构建 exo_service MiniLM 弱监督训练集（品类二元，不按行业分层）。

弱 F = L1 drop；弱 T = after_L1 池上「像商业服务」标题正则（SRS 下采样）。

用法:
  PYTHONPATH=02_脚本 .venv/bin/python3 02_脚本/tools/build_exo_service_weak_labels.py \\
    --l1-drop …/商业服务_quality_l1_drop_0915.csv \\
    --pool …/after_l1.csv \\
    -o work/exo_service_minilm_v0_0915/weak_train_category_v0.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

# 品类级：像商业服务（店内/柜台劳动），非行业枚举
WEAK_T_RE = re.compile(
    r"(理发|美发|剪发|洗剪吹|烫发|染发|发型师|造型师|美甲|纹绣|嫁接睫毛|"
    r"美容院|美容护理|传菜|后厨|吧台|咖啡师|服务员|收银|理货|保洁|家政|"
    r"客房服务|酒店前台|洗车|钣金|喷漆|汽修|宠物美容|给顾客|门店服务|商业服务|"
    r"\bbarber\b|barbershop|haircut|hairstyl|hair\s*salon|"
    r"manicure|pedicure|nail\s*(tech|salon)|facial|"
    r"\bbarista\b|\bwaiter\b|\bwaitress\b|back\s*of\s*house|"
    r"\bcashier\b|checkout|housekeep|janitor|front\s*desk|"
    r"car\s*wash|auto\s*repair|body\s*shop|pet\s*groom)",
    re.I,
)

WEAK_T_EXCLUDE_RE = re.compile(
    r"(植发|毛发移植|hair\s*transplant|fue\b|fut\b|"
    r"教程|妆教|how\s*to|routine|种草|好物推荐|开箱|"
    r"护肤routine|卸妆护肤|diy|短剧|podcast|interview|tedx?|"
    r"保养品|叶黄素|益生菌|ceo|marketing\s*tips)",
    re.I,
)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_weak(
    l1_drop: Path,
    pool: Path,
    *,
    max_f: int,
    max_t: int,
    seed: int,
) -> pd.DataFrame:
    con = duckdb.connect()
    f_df = con.execute(
        """
        SELECT video_id, title, channel, duration_seconds,
               'l1_drop' AS weak_source,
               'F' AS qc_result
        FROM read_csv_auto(?, header=true, sample_size=-1, all_varchar=true)
        WHERE video_id IS NOT NULL AND title IS NOT NULL
        """,
        [str(l1_drop)],
    ).fetchdf()

    p_df = con.execute(
        """
        SELECT video_id, title, channel, duration_seconds,
               'after_l1_regex' AS weak_source
        FROM read_csv_auto(?, header=true, sample_size=-1, all_varchar=true)
        WHERE video_id IS NOT NULL AND title IS NOT NULL
        """,
        [str(pool)],
    ).fetchdf()

    text = (
        p_df["title"].fillna("").astype(str)
        + " "
        + p_df["channel"].fillna("").astype(str)
    )
    hit = text.map(lambda s: bool(WEAK_T_RE.search(s)))
    excl = text.map(lambda s: bool(WEAK_T_EXCLUDE_RE.search(s)))
    t_df = p_df.loc[hit & ~excl].copy()
    t_df["qc_result"] = "T"

    _log(f"弱 F 池: {len(f_df):,}  after_L1: {len(p_df):,}  弱 T 命中: {len(t_df):,}")

    if len(t_df) > max_t:
        t_df = t_df.sample(n=max_t, random_state=seed)
        _log(f"弱 T SRS → {len(t_df):,}")
    if len(f_df) > max_f:
        f_df = f_df.sample(n=max_f, random_state=seed)
        _log(f"弱 F SRS → {len(f_df):,}")

    cols = ["video_id", "title", "channel", "duration_seconds", "weak_source", "qc_result"]
    out = pd.concat([f_df[cols], t_df[cols]], ignore_index=True)
    return out.drop_duplicates("video_id", keep="first").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 exo_service MiniLM 弱监督（品类二元）")
    ap.add_argument("--l1-drop", type=Path, required=True)
    ap.add_argument(
        "--pool", type=Path, required=True,
        help="after_L1 池 CSV（candidates∪unrouted 或专用 after_l1）",
    )
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--max-f", type=int, default=10000)
    ap.add_argument("--max-t", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for p in (args.l1_drop, args.pool):
        if not p.is_file():
            print(f"[ERROR] 缺少: {p}", file=sys.stderr)
            return 1

    df = build_weak(
        args.l1_drop, args.pool,
        max_f=args.max_f, max_t=args.max_t, seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    n_t = int((df["qc_result"] == "T").sum())
    n_f = int((df["qc_result"] == "F").sum())
    _log(f"写出 {args.output}  n={len(df):,} T={n_t:,} F={n_f:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

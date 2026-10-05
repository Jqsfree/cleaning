#!/usr/bin/env python3
"""对 exo_service G4（或任意 CSV）应用文本 certain-noise DROP + 采集词闸门。

用法:
  PYTHONPATH=02_脚本 .venv/bin/python3 02_脚本/tools/apply_exo_service_text_noise.py \\
    data/runs/exo_service/machine_0813/06_tools/text_semantic/after_l1_ml_keep_g4_0916.csv \\
    -o data/runs/exo_service/machine_0813/06_tools/text_semantic/g4_text_0916/ \\
    --eval-labels data/runs/exo_service/machine_0813/03_qc/g4_human_0916/labeled.csv \\
    --halt-on-high-t-hurt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

_SCRIPT = Path(__file__).resolve().parent.parent
_REPO = _SCRIPT.parent
_RULES = _SCRIPT / "categories" / "exo_service" / "rules"
sys.path.insert(0, str(_SCRIPT))

from core.log import log  # noqa: E402


def load_drop_rules(path: Path) -> list[dict[str, str]]:
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    rows = cfg.get("drop") or []
    out = []
    for r in rows:
        re.compile(r["pattern"])
        out.append({"category": r["category"], "pattern": r["pattern"]})
    if not out:
        raise ValueError(f"无 drop 规则: {path}")
    return out


def load_keyword_blacklist(path: Path) -> set[str]:
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    return {str(k).strip() for k in (cfg.get("keywords") or []) if str(k).strip()}


def load_eval_labels(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, low_memory=False)
    if "human_label" in raw.columns:
        lab = raw["human_label"].str.lower().map({"pass": "T", "fail": "F"})
    elif "qc_result" in raw.columns:
        qc = raw["qc_result"].astype(str).str.strip().str.upper()
        lab = qc.map(lambda x: "T" if x.startswith("T") else ("F" if x.startswith("F") else None))
    else:
        raise ValueError("eval-labels 需含 human_label 或 qc_result")
    out = raw.assign(gold=lab).dropna(subset=["gold"]).copy()
    out["video_id"] = out["video_id"].astype(str).str.strip()
    return out[["video_id", "gold"]].drop_duplicates("video_id", keep="last")


def match_drop(
    frame: pd.DataFrame,
    drops: list[dict[str, str]],
    keywords: set[str],
) -> pd.Series:
    """Return drop_category series (NA = keep). First match wins."""
    text = (
        frame.get("title", pd.Series("", index=frame.index)).fillna("").astype(str)
        + " "
        + frame.get("channel", pd.Series("", index=frame.index)).fillna("").astype(str)
    )
    cat = pd.Series(pd.NA, index=frame.index, dtype=object)
    if "keyword" in frame.columns and keywords:
        kw = frame["keyword"].fillna("").astype(str).str.strip()
        hit = kw.isin(keywords) & cat.isna()
        cat = cat.mask(hit, "keyword_blacklist")
    for d in drops:
        rx = re.compile(d["pattern"], flags=re.I)
        hit = text.map(lambda s, r=rx: bool(r.search(s))) & cat.isna()
        cat = cat.mask(hit, d["category"])
    return cat


def hours(df: pd.DataFrame) -> float:
    if "duration_seconds" not in df.columns:
        return 0.0
    return float(pd.to_numeric(df["duration_seconds"], errors="coerce").fillna(0).sum() / 3600)


def overturn(drop_ids: set[str], labels: pd.DataFrame) -> dict[str, Any]:
    m = labels.copy()
    m["dropped"] = m["video_id"].isin(drop_ids)
    n_t = int((m["gold"] == "T").sum())
    n_f = int((m["gold"] == "F").sum())
    t_hurt = int(((m["gold"] == "T") & m["dropped"]).sum())
    f_caught = int(((m["gold"] == "F") & m["dropped"]).sum())
    return {
        "n_eval": len(m),
        "n_t": n_t,
        "n_f": n_f,
        "t_hurt": t_hurt,
        "t_hurt_rate": t_hurt / max(n_t, 1),
        "f_caught": f_caught,
        "f_recall": f_caught / max(n_f, 1),
        "pass_rate_before": n_t / max(n_t + n_f, 1),
        "pass_rate_after": (n_t - t_hurt) / max(n_t - t_hurt + n_f - f_caught, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="exo_service 文本噪声 DROP（G4 等）")
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True, help="输出目录")
    ap.add_argument(
        "--rules",
        type=Path,
        default=_RULES / "cascade_text_noise.toml",
    )
    ap.add_argument(
        "--keyword-blacklist",
        type=Path,
        default=_RULES / "keyword_blacklist.toml",
    )
    ap.add_argument("--stem", default=None)
    ap.add_argument("--eval-labels", type=Path, default=None)
    ap.add_argument("--halt-on-high-t-hurt", action="store_true")
    ap.add_argument("--max-t-hurt-rate", type=float, default=0.10)
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"[ERROR] 缺少输入: {args.input}", file=sys.stderr)
        return 1

    drops = load_drop_rules(args.rules)
    kws = load_keyword_blacklist(args.keyword_blacklist) if args.keyword_blacklist.is_file() else set()
    log(f"rules={args.rules.name} drops={len(drops)} keyword_bl={len(kws)}")

    df = pd.read_csv(args.input, dtype=str, low_memory=False)
    log(f"input={args.input} n={len(df):,}")
    drop_cat = match_drop(df, drops, kws)
    out = df.copy()
    out["text_drop_category"] = drop_cat
    out["text_auto_drop"] = drop_cat.notna()

    keep_df = out.loc[~out["text_auto_drop"]].copy()
    drop_df = out.loc[out["text_auto_drop"]].copy()

    args.output.mkdir(parents=True, exist_ok=True)
    date_tag = time.strftime("%m%d")
    stem = args.stem or args.input.stem
    keep_path = args.output / f"{stem}_text_keep_{date_tag}.csv"
    drop_path = args.output / f"{stem}_text_drop_{date_tag}.csv"
    keep_df.to_csv(keep_path, index=False)
    drop_df.to_csv(drop_path, index=False)

    by_cat = (
        drop_df["text_drop_category"].value_counts(dropna=False).astype(int).to_dict()
        if len(drop_df) else {}
    )
    summary: dict[str, Any] = {
        "input": str(args.input),
        "n_input": len(df),
        "hours_input": round(hours(df), 1),
        "n_keep": len(keep_df),
        "hours_keep": round(hours(keep_df), 1),
        "n_drop": len(drop_df),
        "hours_drop": round(hours(drop_df), 1),
        "drop_by_category": by_cat,
        "keep_path": str(keep_path),
        "drop_path": str(drop_path),
        "rules": str(args.rules),
        "keyword_blacklist": str(args.keyword_blacklist) if kws else None,
    }

    if args.eval_labels and args.eval_labels.is_file():
        labels = load_eval_labels(args.eval_labels)
        drop_ids = set(drop_df["video_id"].astype(str).str.strip())
        ov = overturn(drop_ids, labels)
        summary["overturn"] = ov
        log(
            f"eval: t_hurt={ov['t_hurt']}/{ov['n_t']} ({ov['t_hurt_rate']:.1%})  "
            f"f_recall={ov['f_caught']}/{ov['n_f']} ({ov['f_recall']:.1%})  "
            f"pass {ov['pass_rate_before']:.1%}→{ov['pass_rate_after']:.1%}"
        )
        if args.halt_on_high_t_hurt and ov["t_hurt_rate"] > args.max_t_hurt_rate:
            print(
                f"[HALT] t_hurt_rate {ov['t_hurt_rate']:.1%} > {args.max_t_hurt_rate:.1%}",
                file=sys.stderr,
            )
            (args.output / "text_noise_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return 2

    sum_path = args.output / "text_noise_summary.json"
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"keep={keep_path.name} ({summary['n_keep']:,}, {summary['hours_keep']:,.1f}h)")
    log(f"drop={drop_path.name} ({summary['n_drop']:,}, {summary['hours_drop']:,.1f}h)")
    log(f"summary → {sum_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
tools/filter_exemplar_neg.py — 人工 F 负例原型 → margin 二滤

用法:
  02_脚本/tools/filter_exemplar_neg.py \\
    --labels /home/jqs/tmp/纯直播机采_e26e1746_qc_result.csv \\
    --pool data/runs/live_sell/machine_0805/06_tools/纯直播机采_0805_exemplar_keep_high_0805.csv \\
    --pos-sim data/runs/live_sell/machine_0805/06_tools/纯直播机采_0805_records_quality_0805_exemplar_sim_0805.csv \\
    --pos-bank data/assets/exemplars/yt_live_scene/ \\
    -o data/runs/live_sell/machine_0805/06_tools/ \\
    --neg-bank data/assets/exemplars/yt_live_scene_neg_human_f/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.exemplar_sim import (  # noqa: E402
    ClipEncoder,
    build_bank_from_video_ids,
    load_bank,
    score_max_sim_to_bank,
)
from core.run_manifest import maybe_update_stage  # noqa: E402


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p, dtype=str, low_memory=False)


def _norm_bool(s: pd.Series) -> pd.Series:
    v = s.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype=object)
    out[v.isin({"true", "1", "t", "yes", "pass"})] = True
    out[v.isin({"false", "0", "f", "no", "fail"})] = False
    return out


def _calibrate_margin(
    labeled: pd.DataFrame,
    *,
    objective: str = "pass_rate",
    min_t_recall: float = 0.90,
    min_keep_labels: int = 8,
    min_sim: float | None = None,
) -> dict:
    """标定 margin 阈值。

    objective=pass_rate（默认）：允许误杀 T，最大化标签通过率（keep 内 T/(T+F)）。
    objective=t_recall：优先保住 ≥min_t_recall 的 T，再尽量丢 F。
    """
    lab = labeled.dropna(subset=["margin", "qc_bool"]).copy()
    if min_sim is not None:
        lab = lab[lab["sim_score"].fillna(-1) >= float(min_sim)].copy()
    t = lab[lab["qc_bool"] == True]  # noqa: E712
    f = lab[lab["qc_bool"] == False]  # noqa: E712
    base = {
        "t_n": int(len(t)),
        "f_n": int(len(f)),
        "base_pass_rate": float(len(t) / (len(t) + len(f))) if (len(t) + len(f)) else None,
        "min_sim": min_sim,
        "objective": objective,
    }
    if t.empty or f.empty or lab.empty:
        thr = float(lab["margin"].median()) if not lab.empty else 0.0
        return {**base, "threshold": thr, "method": "median_fallback"}

    if objective == "t_recall":
        qs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        rows = []
        for q in qs:
            thr = float(t["margin"].quantile(q))
            t_keep = float((t["margin"] >= thr).mean())
            f_drop = float((f["margin"] < thr).mean())
            rows.append({"q": q, "threshold": thr, "t_recall": t_keep, "f_drop": f_drop})
        ok = [r for r in rows if r["t_recall"] >= min_t_recall]
        if ok:
            best = max(ok, key=lambda r: (r["f_drop"], -r["threshold"]))
            method = f"t_q_min_recall_{min_t_recall}"
        else:
            best = max(rows, key=lambda r: (r["t_recall"], r["f_drop"]))
            method = "best_effort_t_recall"
        return {
            **base,
            "threshold": best["threshold"],
            "method": method,
            "t_recall": best["t_recall"],
            "f_drop": best["f_drop"],
            "grid": rows,
            "pos_mean_t": float(t["sim_score"].mean()),
            "pos_mean_f": float(f["sim_score"].mean()),
            "neg_mean_t": float(t["neg_sim"].mean()),
            "neg_mean_f": float(f["neg_sim"].mean()),
            "margin_mean_t": float(t["margin"].mean()),
            "margin_mean_f": float(f["margin"].mean()),
        }

    # pass_rate：扫 margin 阈值，最大化 keep 内通过率；要求至少 min_keep_labels
    margins = np.sort(lab["margin"].dropna().unique())
    if len(margins) > 120:
        # 均匀下采样候选点
        idx = np.linspace(0, len(margins) - 1, 120).astype(int)
        cand_thrs = margins[idx]
    else:
        cand_thrs = margins

    rows = []
    for thr in cand_thrs:
        keep = lab[lab["margin"] >= float(thr)]
        tk = int((keep["qc_bool"] == True).sum())  # noqa: E712
        fk = int((keep["qc_bool"] == False).sum())  # noqa: E712
        n = tk + fk
        if n <= 0:
            continue
        rows.append({
            "threshold": float(thr),
            "n_keep_labels": n,
            "t_kept": tk,
            "f_kept": fk,
            "pass_rate": tk / n,
            "t_recall": tk / len(t) if len(t) else 0.0,
            "f_drop": 1.0 - (fk / len(f) if len(f) else 0.0),
        })

    eligible = [r for r in rows if r["n_keep_labels"] >= int(min_keep_labels)]
    pool = eligible or rows
    best = max(pool, key=lambda r: (r["pass_rate"], r["n_keep_labels"], r["threshold"]))
    return {
        **base,
        "threshold": best["threshold"],
        "method": f"max_pass_rate_min_keep_{min_keep_labels}" if eligible else "max_pass_rate_no_min_keep",
        "t_recall": best["t_recall"],
        "f_drop": best["f_drop"],
        "pass_rate": best["pass_rate"],
        "n_keep_labels": best["n_keep_labels"],
        "grid_top": sorted(rows, key=lambda r: (-r["pass_rate"], -r["n_keep_labels"]))[:15],
        "pos_mean_t": float(t["sim_score"].mean()),
        "pos_mean_f": float(f["sim_score"].mean()),
        "neg_mean_t": float(t["neg_sim"].mean()),
        "neg_mean_f": float(f["neg_sim"].mean()),
        "margin_mean_t": float(t["margin"].mean()),
        "margin_mean_f": float(f["margin"].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="人工负例 margin 二滤")
    ap.add_argument("--labels", required=True, help="人工 QC CSV（需 video_id, qc_result）")
    ap.add_argument("--pool", required=True, help="待滤池（如 exemplar keep high）")
    ap.add_argument("--pos-sim", required=True, help="正例相似度全量 CSV（video_id,sim_score,band）")
    ap.add_argument("--pos-bank", default="data/assets/exemplars/yt_live_scene/")
    ap.add_argument("--neg-bank", default="data/assets/exemplars/yt_live_scene_neg_human_f/")
    ap.add_argument("-o", "--output-dir", required=True)
    ap.add_argument("--cache-dir", default="qc_thumb_cache/exemplar_sim")
    ap.add_argument("--rebuild-neg-bank", action="store_true")
    ap.add_argument(
        "--objective", choices=["pass_rate", "t_recall"], default="pass_rate",
        help="pass_rate=抬通过率可误杀（默认）；t_recall=少误杀",
    )
    ap.add_argument("--min-t-recall", type=float, default=0.90, help="仅 objective=t_recall")
    ap.add_argument("--min-keep-labels", type=int, default=8, help="pass_rate 标定最少保留标签数")
    ap.add_argument(
        "--min-sim", type=float, default=0.70,
        help="额外正例相似度下限（与 margin 双门；默认 0.70 偏纯度）",
    )
    ap.add_argument("--margin-threshold", type=float, default=None, help="手动 margin 阈值；默认用标签标定")
    ap.add_argument("--batch-rows", type=int, default=5000)
    ap.add_argument("--thumb-workers", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = time.strftime("%m%d")

    # --- labels ---
    lab = _read(args.labels)
    if "video_id" not in lab.columns or "qc_result" not in lab.columns:
        print("[ERROR] labels 需要 video_id + qc_result")
        sys.exit(2)
    lab = lab.copy()
    lab["video_id"] = lab["video_id"].astype(str).str.strip()
    lab["qc_bool"] = _norm_bool(lab["qc_result"])
    neg_ids = lab.loc[lab["qc_bool"] == False, "video_id"].tolist()  # noqa: E712
    pos_ids = lab.loc[lab["qc_bool"] == True, "video_id"].tolist()  # noqa: E712
    print(f"labels: n={len(lab)}  T={len(pos_ids)}  F={len(neg_ids)}  U/null={lab['qc_bool'].isna().sum()}")

    # provenance copy
    lab_copy = out_dir / f"neg_labels_human_{date_tag}.csv"
    if Path(args.labels).resolve() != lab_copy.resolve():
        shutil.copy2(args.labels, lab_copy)

    # --- neg bank ---
    neg_bank = Path(args.neg_bank)
    if args.rebuild_neg_bank or not (neg_bank / "prototypes.npy").is_file():
        print(f"build neg bank → {neg_bank}  n_F={len(neg_ids)}")
        meta = build_bank_from_video_ids(
            neg_ids,
            neg_bank,
            cache_dir=args.cache_dir,
            thumb_workers=args.thumb_workers,
            batch_size=args.batch_size,
            note="人工 QC qc_result=false 缩略图负例；用于 margin 二滤",
            labels={vid: "F" for vid in neg_ids},
        )
        print(f"neg bank ok  n={meta['n_exemplars']}")
    else:
        print(f"reuse neg bank {neg_bank}")

    proto_neg, man_neg, meta_neg = load_bank(neg_bank)
    neg_exemplar_ids = man_neg["exemplar_id"].astype(str).tolist()
    pos_meta = json.loads(Path(args.pos_bank, "meta.json").read_text(encoding="utf-8"))
    encoder = ClipEncoder(
        meta_neg.get("model") or pos_meta.get("model", "ViT-B-32"),
        meta_neg.get("pretrained") or pos_meta.get("pretrained", "openai"),
    )

    # --- pool + pos sim ---
    pool = _read(args.pool)
    pool["video_id"] = pool["video_id"].astype(str).str.strip()
    pos_sim = _read(args.pos_sim)[["video_id", "sim_score", "band"]].copy()
    pos_sim["video_id"] = pos_sim["video_id"].astype(str).str.strip()
    pos_sim["sim_score"] = pd.to_numeric(pos_sim["sim_score"], errors="coerce")

    stem = Path(args.pool).stem
    ckpt = out_dir / f"{stem}_neg_sim.ckpt.csv"
    scored_path = out_dir / f"{stem}_neg_margin_{date_tag}.csv"
    keep_path = out_dir / f"{stem}_neg_margin_keep_{date_tag}.csv"
    drop_path = out_dir / f"{stem}_neg_margin_drop_{date_tag}.csv"
    sum_path = out_dir / f"{stem}_neg_margin_{date_tag}_summary.json"

    done: set[str] = set()
    parts: list[pd.DataFrame] = []
    if ckpt.is_file() and not args.overwrite:
        prev = pd.read_csv(ckpt, dtype={"video_id": str})
        parts.append(prev)
        done = set(prev["video_id"].astype(str).str.strip())
        print(f"[续跑] neg ckpt already={len(done)}")

    pending = [v for v in pool["video_id"].tolist() if v not in done]
    print(f"pool={len(pool)}  pending_neg_score={len(pending)}")

    t0 = time.time()
    for start in range(0, len(pending), args.batch_rows):
        chunk = pending[start : start + args.batch_rows]
        print(f"=== neg batch {start // args.batch_rows + 1}  rows={len(chunk)} ===", flush=True)
        scored = score_max_sim_to_bank(
            chunk,
            proto_neg,
            neg_exemplar_ids,
            encoder,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            thumb_workers=args.thumb_workers,
            exclude_self=True,
        )
        parts.append(scored)
        pd.concat(parts, ignore_index=True).to_csv(ckpt, index=False)
        elapsed = time.time() - t0
        done_n = sum(len(p) for p in parts)
        rate = done_n / elapsed if elapsed > 0 else 0
        print(f"  ckpt done={done_n}  {rate:.1f} rows/s", flush=True)

    neg_all = pd.concat(parts, ignore_index=True)
    neg_all["video_id"] = neg_all["video_id"].astype(str).str.strip()
    neg_all["neg_sim"] = pd.to_numeric(neg_all["neg_sim"], errors="coerce")

    merged = pool.merge(pos_sim, on="video_id", how="left", suffixes=("", "_pos"))
    if "sim_score_pos" in merged.columns and "sim_score" not in merged.columns:
        merged = merged.rename(columns={"sim_score_pos": "sim_score"})
    # pool may already have sim_score
    if "sim_score" in pool.columns:
        merged["sim_score"] = pd.to_numeric(merged["sim_score"], errors="coerce")
    else:
        merged["sim_score"] = pd.to_numeric(merged.get("sim_score"), errors="coerce")
    merged = merged.merge(neg_all[["video_id", "neg_sim", "nearest_neg_id"]], on="video_id", how="left")
    merged["margin"] = merged["sim_score"] - merged["neg_sim"]

    # --- calibrate on labels (LOO neg already in score) ---
    lab_join = lab[["video_id", "qc_bool"]].merge(
        merged[["video_id", "sim_score", "neg_sim", "margin"]],
        on="video_id",
        how="inner",
    )
    # labels may be from high+mid partial; if not in high pool, score them ad-hoc
    missing = lab.loc[~lab["video_id"].isin(merged["video_id"]), "video_id"].tolist()
    if missing:
        print(f"labels not in pool: {len(missing)} → 单独打 neg_sim 用于标定")
        extra_neg = score_max_sim_to_bank(
            missing,
            proto_neg,
            neg_exemplar_ids,
            encoder,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            thumb_workers=args.thumb_workers,
            exclude_self=True,
        )
        extra = lab.loc[lab["video_id"].isin(missing), ["video_id", "qc_bool"]].merge(
            pos_sim, on="video_id", how="left",
        ).merge(extra_neg[["video_id", "neg_sim"]], on="video_id", how="left")
        extra["margin"] = pd.to_numeric(extra["sim_score"], errors="coerce") - pd.to_numeric(
            extra["neg_sim"], errors="coerce",
        )
        lab_join = pd.concat([lab_join, extra[["video_id", "qc_bool", "sim_score", "neg_sim", "margin"]]], ignore_index=True)

    cal = _calibrate_margin(
        lab_join,
        objective=args.objective,
        min_t_recall=args.min_t_recall,
        min_keep_labels=args.min_keep_labels,
        min_sim=args.min_sim,
    )
    thr = args.margin_threshold if args.margin_threshold is not None else float(cal["threshold"])
    min_sim = args.min_sim
    print(
        f"objective={args.objective}  margin_thr={thr:.4f}  min_sim={min_sim}  "
        f"method={cal.get('method')}  pass_rate={cal.get('pass_rate')}  "
        f"T_recall={cal.get('t_recall')} F_drop={cal.get('f_drop')}"
    )

    merged["pass_margin"] = merged["margin"] >= thr
    if min_sim is not None:
        merged["pass_margin"] = merged["pass_margin"] & (
            merged["sim_score"].fillna(-1) >= float(min_sim)
        )
    # missing margin → drop（偏纯度）
    merged.loc[merged["margin"].isna(), "pass_margin"] = False

    keep = merged.loc[merged["pass_margin"]].copy()
    drop = merged.loc[~merged["pass_margin"]].copy()
    merged.to_csv(scored_path, index=False)
    keep.to_csv(keep_path, index=False)
    drop.to_csv(drop_path, index=False)

    # labeled metrics at chosen thr (+ min_sim)
    lj = lab_join.dropna(subset=["margin", "qc_bool"]).copy()
    if min_sim is not None:
        lj["_pass"] = (lj["margin"] >= thr) & (lj["sim_score"].fillna(-1) >= float(min_sim))
    else:
        lj["_pass"] = lj["margin"] >= thr
    t = lj[lj["qc_bool"] == True]  # noqa: E712
    f = lj[lj["qc_bool"] == False]  # noqa: E712
    kept = lj[lj["_pass"]]
    metrics = {
        "t_n": int(len(t)),
        "f_n": int(len(f)),
        "t_kept": int(t["_pass"].sum()) if len(t) else 0,
        "f_kept": int(f["_pass"].sum()) if len(f) else 0,
        "t_recall": float(t["_pass"].mean()) if len(t) else None,
        "f_drop": float((~f["_pass"]).mean()) if len(f) else None,
        "implied_pass_rate_on_labels": (
            float((kept["qc_bool"] == True).mean()) if len(kept) else None  # noqa: E712
        ),
        "n_keep_labels": int(len(kept)),
    }

    summary = {
        "labels": str(Path(args.labels).resolve()),
        "labels_copy": str(lab_copy),
        "pool": str(Path(args.pool).resolve()),
        "neg_bank": str(neg_bank.resolve()),
        "n_neg_prototypes": len(neg_exemplar_ids),
        "n_pool": int(len(merged)),
        "n_keep": int(len(keep)),
        "n_drop": int(len(drop)),
        "objective": args.objective,
        "margin_threshold": thr,
        "min_sim": min_sim,
        "calibration": cal,
        "label_metrics_at_threshold": metrics,
        "scored": str(scored_path),
        "keep": str(keep_path),
        "drop": str(drop_path),
        "note": "margin=pos_sim-neg_sim；负例=人工 qc_result=false；exclude_self LOO",
    }
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "n_pool", "n_keep", "n_drop", "margin_threshold", "label_metrics_at_threshold",
    )}, ensure_ascii=False, indent=2))
    print(f"keep → {keep_path}")
    print(f"drop → {drop_path}")

    maybe_update_stage(
        out_dir,
        "exemplar_neg_margin",
        paths={
            "keep": str(keep_path),
            "drop": str(drop_path),
            "summary": str(sum_path),
        },
        stats={
            "n_keep": int(len(keep)),
            "n_drop": int(len(drop)),
            "margin_threshold": thr,
        },
    )


if __name__ == "__main__":
    main()

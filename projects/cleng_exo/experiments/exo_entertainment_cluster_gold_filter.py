#!/usr/bin/env python3
"""人标映射到 exo_entertainment MiniLM 簇 → 按纯度规则过滤 keep。

默认只按金标纯度丢簇（n_tf≥8 & pass<0.35；或 n_tf≥5 & pass=0）。
无人标时拒绝整簇丢（exit 2）。可选 --allow-tag-fallback（不推荐作交付）。

用法:
  # 金标已入库 03_qc/
  .venv/bin/python3 experiments/exo_entertainment_cluster_gold_filter.py \\
    --cluster-dir work/exo_entertainment_text_cluster_0918 \\
    --keep data/runs/exo_entertainment/machine_0818/05_clean/run01_v01/娱乐表演_merged_0818_clean_0917.csv \\
    --qc-dir data/runs/exo_entertainment/machine_0818/03_qc \\
    --out-keep data/runs/exo_entertainment/machine_0818/05_clean/run02_cluster_filt/娱乐表演_merged_0818_cluster_filt_0918.csv \\
    --device mps

  # 或显式金标 CSV（需含 video_id + human_label|qc_result + title/channel）
  .venv/bin/python3 experiments/exo_entertainment_cluster_gold_filter.py \\
    --cluster-dir work/exo_entertainment_text_cluster_0918 \\
    --gold /path/to/labeled.csv --gold /path/to/more.csv
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "02_脚本"))
sys.path.insert(0, str(PROJECT / "experiments"))
from core.dotenv_load import load_project_env  # noqa: E402

load_project_env()
import exo_cook_text_classifier as ck  # noqa: E402

DEFAULT_KEEP = (
    PROJECT
    / "data/runs/exo_entertainment/machine_0818/05_clean/run01_v01"
    / "娱乐表演_merged_0818_clean_0917.csv"
)
DEFAULT_QC = PROJECT / "data/runs/exo_entertainment/machine_0818/03_qc"
DEFAULT_CLUSTER = PROJECT / "work/exo_entertainment_text_cluster_0918"

# 仅 --allow-tag-fallback 时使用
NOISE_TAGS = {
    "news_broadcast",
    "anime_cartoon",
    "kids_nursery",
    "certain_noise_nonperf",
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _normalize_lab_series(s: pd.Series) -> np.ndarray:
    u = s.fillna("").astype(str).str.strip().str.upper()
    out = np.full(len(u), "U", dtype=object)
    out[u.str.startswith("T") | u.isin({"PASS", "1", "TRUE", "Y", "YES"})] = "T"
    out[u.str.startswith("F") | u.isin({"FAIL", "0", "FALSE", "N", "NO"})] = "F"
    # qc_result style T/... F/...
    return out


def load_gold_from_paths(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, dtype=str)
        lab_col = None
        for c in ("human_label", "qc_result", "label", "audit_label"):
            if c in df.columns:
                lab_col = c
                break
        if lab_col is None or "video_id" not in df.columns:
            raise ValueError(f"{path}: need video_id + human_label|qc_result")
        lab = _normalize_lab_series(df[lab_col])
        mask = np.isin(lab, ["T", "F"])
        sub = df.loc[mask, ["video_id"]].copy()
        sub["lab"] = lab[mask]
        sub["title"] = df.loc[mask, "title"].fillna("").astype(str) if "title" in df.columns else ""
        sub["channel"] = (
            df.loc[mask, "channel"].fillna("").astype(str) if "channel" in df.columns else ""
        )
        sub["batch"] = path.stem
        rows.append(sub)
        _log(f"gold {path.name}: TF={int(mask.sum())}")
    if not rows:
        return pd.DataFrame(columns=["video_id", "lab", "title", "channel", "batch"])
    out = pd.concat(rows, ignore_index=True).drop_duplicates("video_id", keep="last")
    return out.reset_index(drop=True)


def discover_qc_gold(qc_dir: Path) -> list[Path]:
    if not qc_dir.is_dir():
        return []
    cands = []
    for name in ("train_export.csv", "labeled.csv", "pass.csv", "fail.csv"):
        p = qc_dir / name
        if p.is_file():
            cands.append(p)
    # also nested
    for p in sorted(qc_dir.rglob("*labeled*.csv")):
        if p not in cands:
            cands.append(p)
    for p in sorted(qc_dir.rglob("train_export.csv")):
        if p not in cands:
            cands.append(p)
    return cands


def build_texts_df(df: pd.DataFrame) -> list[str]:
    title = df["title"].fillna("").astype(str)
    channel = df["channel"].fillna("").astype(str)
    return title.str.replace(r"\s+", " ", regex=True).str.strip().tolist()


def compute_centroids(emb: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    dim = emb.shape[1]
    cents = np.zeros((k, dim), dtype=np.float32)
    counts = np.zeros(k, dtype=np.int64)
    batch = 50000
    for start in range(0, len(labels), batch):
        end = min(start + batch, len(labels))
        X = np.asarray(emb[start:end], dtype=np.float32)
        lab = labels[start:end]
        for c in range(k):
            mask = lab == c
            if not mask.any():
                continue
            cents[c] += X[mask].sum(axis=0)
            counts[c] += int(mask.sum())
    for c in range(k):
        if counts[c] > 0:
            cents[c] /= counts[c]
            nrm = np.linalg.norm(cents[c]) + 1e-8
            cents[c] /= nrm
    return cents


def assign_nearest(vecs: np.ndarray, cents: np.ndarray) -> np.ndarray:
    sims = vecs @ cents.T
    return sims.argmax(axis=1).astype(np.int32)


def heuristic_tag(top_terms: str, top_channels: str, samples: list[str]) -> tuple[str, float]:
    blob = f"{top_terms} {top_channels} " + " ".join(samples[:20])
    blob_l = blob.lower()
    if re.search(r"\bnews\b|新聞|新闻|formosa\s*news|fox\s*news|cnn", blob_l):
        return "news_broadcast", 0.75
    if re.search(r"\banime\b|\bcartoon\b|动漫|動漫|babybus|videogyan|naruto", blob_l):
        return "anime_cartoon", 0.75
    if re.search(r"cocomelon|peppa|nursery|儿歌|paw\s*patrol|preschool", blob_l):
        return "kids_nursery", 0.7
    if re.search(r"gameplay|minecraft|roblox|asmr|podcast|mukbang|workout|skincare", blob_l):
        return "certain_noise_nonperf", 0.7
    if re.search(r"busk|street\s*(show|perform)|talent\s*show|dance\s*practice|acrobat", blob_l):
        return "live_performance", 0.65
    return "other", 0.4


def main() -> int:
    ap = argparse.ArgumentParser(description="exo_entertainment cluster gold purity filter")
    ap.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER)
    ap.add_argument("--keep", type=Path, default=DEFAULT_KEEP)
    ap.add_argument("--qc-dir", type=Path, default=DEFAULT_QC, help="自动找 labeled/train_export")
    ap.add_argument("--gold", type=Path, action="append", default=None, help="可重复；显式金标 CSV")
    ap.add_argument("--out-keep", type=Path, default=None, help="过滤后 keep CSV；默认 stem+_cluster_filt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--allow-tag-fallback",
        action="store_true",
        help="无人标可丢时用启发式/LLM 噪声标签丢簇（默认关闭）",
    )
    args = ap.parse_args()

    cdir = args.cluster_dir.resolve()
    keep_path = args.keep.resolve()
    if not (cdir / "embeddings.f16.npy").is_file() or not (cdir / "labels.parquet").is_file():
        raise SystemExit(f"cluster artifacts missing under {cdir}")
    if not keep_path.is_file():
        raise SystemExit(f"keep missing: {keep_path}")

    gold_paths: list[Path] = []
    if args.gold:
        gold_paths.extend(p.resolve() for p in args.gold)
    else:
        gold_paths = discover_qc_gold(args.qc_dir.resolve())

    if not gold_paths:
        pending = cdir / "AWAIT_GOLD.md"
        pending.write_text(
            f"""# 等待人标金标

- 聚类目录: `{cdir}`
- keep: `{keep_path}`
- 预期金标: `{args.qc_dir}/labeled.csv` 或 `train_export.csv`（`ingest_human_qc.py` 入库）
- 或: `--gold /path/to/labeled.csv`

入库后再跑本脚本；**禁止无人标整簇丢**。
""",
            encoding="utf-8",
        )
        _log(f"no gold found — wrote {pending}; exit 2")
        return 2

    gold = load_gold_from_paths(gold_paths)
    if len(gold) < 5:
        _log(f"gold TF too few: {len(gold)} — exit 2")
        return 2

    emb = np.load(cdir / "embeddings.f16.npy", mmap_mode="r")
    lab_df = pd.read_parquet(cdir / "labels.parquet")
    index = pd.read_parquet(cdir / "index.parquet")
    if "row" in index.columns:
        index = index.sort_values("row").reset_index(drop=True)
    labels = index.merge(lab_df, on="video_id", how="left")["cluster_id"].to_numpy()
    if len(labels) != emb.shape[0]:
        labels = lab_df["cluster_id"].to_numpy()
    k = int(labels.max()) + 1
    _log(f"centroids K={k} from {emb.shape[0]:,} vectors; gold TF={len(gold)}")
    cents = compute_centroids(emb, labels.astype(np.int32), k)
    np.save(cdir / "centroids.f32.npy", cents)

    keep_ids = set(pd.read_csv(keep_path, usecols=["video_id"], dtype=str)["video_id"].tolist())
    gold["in_keep"] = gold["video_id"].isin(keep_ids)

    # fill missing title/channel from keep index if needed
    if (gold["title"].astype(str).str.len() == 0).all() and "title" in index.columns:
        m = index.set_index("video_id")[["title", "channel"]]
        gold = gold.join(m, on="video_id", rsuffix="_idx")
        if "title_idx" in gold.columns:
            gold["title"] = gold["title"].where(gold["title"].astype(str).str.len() > 0, gold["title_idx"])
            gold["channel"] = gold["channel"].where(
                gold["channel"].astype(str).str.len() > 0, gold["channel_idx"]
            )

    encoder = ck.MiniLMEncoder(batch_size=args.batch_size, device=args.device)
    encoder.fit([])
    vecs = encoder.transform(build_texts_df(gold)).astype(np.float32)
    gold["cluster_id"] = assign_nearest(vecs, cents)
    gold_out = cdir / "gold_cluster_assign.parquet"
    gold[["video_id", "lab", "batch", "in_keep", "cluster_id", "title", "channel"]].to_parquet(
        gold_out, index=False
    )

    summary = pd.read_csv(cdir / "cluster_summary.csv")
    samples = pd.read_csv(cdir / "cluster_samples.csv", dtype=str)

    stats_rows = []
    for cid in range(k):
        sub = gold[gold.cluster_id == cid]
        nt = int((sub.lab == "T").sum())
        nf = int((sub.lab == "F").sum())
        n_tf = nt + nf
        in_k = int(sub.in_keep.sum())
        pass_rate = (nt / n_tf) if n_tf else None
        meta = summary[summary.cluster_id == cid]
        n_pool = int(meta.n.iloc[0]) if len(meta) else 0
        hours = float(meta.hours.iloc[0]) if len(meta) else 0.0
        top_terms = str(meta.top_terms.iloc[0]) if len(meta) else ""
        top_ch = str(meta.top_channels.iloc[0]) if len(meta) else ""
        samp_titles = samples.loc[samples.cluster_id.astype(int) == cid, "title"].fillna("").tolist()
        tag, conf = heuristic_tag(top_terms, top_ch, samp_titles)

        action = "keep"
        reason = ""
        if n_tf >= 8 and pass_rate is not None and pass_rate < 0.35:
            action = "drop_gold"
            reason = f"n_tf>=8 pass={pass_rate:.2f}<0.35"
        elif n_tf >= 5 and pass_rate == 0.0:
            action = "drop_gold"
            reason = f"n_tf>=5 pass=0"

        stats_rows.append(
            {
                "cluster_id": cid,
                "n_pool": n_pool,
                "hours": hours,
                "n_t": nt,
                "n_f": nf,
                "n_tf": n_tf,
                "pass": None if pass_rate is None else round(pass_rate, 4),
                "n_gold_in_keep": in_k,
                "heuristic_tag": tag,
                "heuristic_conf": conf,
                "action": action,
                "action_reason": reason,
                "top_terms": top_terms[:120],
            }
        )

    stats = pd.DataFrame(stats_rows)
    gold_drops = set(stats.loc[stats.action == "drop_gold", "cluster_id"].astype(int))

    if not gold_drops and args.allow_tag_fallback:
        _log("no gold-rule drops — applying tag fallback (--allow-tag-fallback)")
        for i, row in stats.iterrows():
            tag = row.heuristic_tag
            conf = float(row.heuristic_conf)
            pr = row["pass"]
            if tag not in NOISE_TAGS or conf < 0.65:
                continue
            if pr is not None and float(pr) > 0.50:
                continue
            if row.n_tf == 0 or (pr is not None and float(pr) <= 0.50):
                stats.at[i, "action"] = "drop_tag"
                stats.at[i, "action_reason"] = f"tag={tag} conf={conf:.2f} pass={pr}"
    elif not gold_drops:
        _log("no gold-rule drops — keep all clusters (no tag fallback)")

    stats_path = cdir / "cluster_gold_stats.csv"
    stats.to_csv(stats_path, index=False)
    drop_ids = set(stats.loc[stats.action.str.startswith("drop"), "cluster_id"].astype(int))
    _log(f"drop clusters: {sorted(drop_ids)} ({len(drop_ids)})")

    if args.out_keep:
        out_keep = args.out_keep.resolve()
    else:
        out_keep = keep_path.with_name(keep_path.stem + "_cluster_filt.csv")
    out_keep.parent.mkdir(parents=True, exist_ok=True)

    drop_vids = lab_df[lab_df.cluster_id.isin(drop_ids)]["video_id"].astype(str)
    drop_vid_path = cdir / "drop_cluster_video_ids.txt"
    drop_vid_path.write_text("\n".join(drop_vids.tolist()) + "\n")

    import duckdb

    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE drop_v(video_id VARCHAR)")
    if len(drop_vids):
        con.executemany("INSERT INTO drop_v VALUES (?)", [(v,) for v in drop_vids.tolist()])

    n0, h0 = con.execute(
        f"""
        SELECT COUNT(*), COALESCE(SUM(TRY_CAST(duration_seconds AS DOUBLE)),0)/3600.0
        FROM read_csv_auto('{keep_path}', header=true, all_varchar=true)
        """
    ).fetchone()

    if drop_ids:
        con.execute(
            f"""
            COPY (
              SELECT r.* FROM read_csv_auto('{keep_path}', header=true, all_varchar=true) r
              ANTI JOIN drop_v d USING (video_id)
            ) TO '{out_keep}' (HEADER, DELIMITER ',')
            """
        )
    else:
        shutil.copy2(keep_path, out_keep)

    n1, h1 = con.execute(
        f"""
        SELECT COUNT(*), COALESCE(SUM(TRY_CAST(duration_seconds AS DOUBLE)),0)/3600.0
        FROM read_csv_auto('{out_keep}', header=true, all_varchar=true)
        """
    ).fetchone()

    report = {
        "category": "exo_entertainment",
        "gold_n": int(len(gold)),
        "gold_paths": [str(p) for p in gold_paths],
        "gold_in_keep": int(gold.in_keep.sum()),
        "k": k,
        "drop_clusters": sorted(int(x) for x in drop_ids),
        "drop_by_action": stats.loc[stats.action.str.startswith("drop"), "action"]
        .value_counts()
        .to_dict(),
        "keep_before": {"n": int(n0), "hours": round(float(h0), 1)},
        "keep_after": {"n": int(n1), "hours": round(float(h1), 1)},
        "hours_dropped": round(float(h0) - float(h1), 1),
        "n_dropped": int(n0) - int(n1),
        "out_keep": str(out_keep),
        "allow_tag_fallback": bool(args.allow_tag_fallback),
        "note": "title MiniLM; drop only via gold purity unless --allow-tag-fallback",
    }
    (cdir / "gold_filter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

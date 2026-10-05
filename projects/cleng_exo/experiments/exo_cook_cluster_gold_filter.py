#!/usr/bin/env python3
"""人标映射到 exo_cook MiniLM 簇 → 诊断 + 按规则过滤 keep。

用法:
  .venv/bin/python3 experiments/exo_cook_cluster_gold_filter.py \\
    --cluster-dir work/exo_cook_text_cluster_0917 \\
    --keep data/runs/exo_cook/machine_0916/06_tools/text_gov_v03/烹饪教学_merged_0916_ml_keep_0917.csv \\
    --device mps
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

GOLD_PATHS = [
    ("/Users/muse/Downloads/humen-烹饪餐饮/烹饪餐饮v0.1_95aa1f5a_qc_result.csv", "v01"),
    ("/Users/muse/Downloads/humen-烹饪餐饮/烹饪餐饮v0.5_a42d3cf2_qc_result.csv", "v05"),
    ("/Users/muse/Downloads/humen-烹饪餐饮/烹饪餐饮v0.6_a74e9860_qc_result.csv", "v06"),
]

NOISE_TAGS = {"mukbang", "food_review_street", "certain_noise_noncook"}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_gold() -> pd.DataFrame:
    rows = []
    for path, batch in GOLD_PATHS:
        df = pd.read_csv(path, dtype=str)
        qc = df["qc_result"].fillna("").astype(str).str.strip()
        lab = np.where(
            qc.str.upper().str.startswith("T"),
            "T",
            np.where(qc.str.upper().str.startswith("F"), "F", "U"),
        )
        sub = df.loc[np.isin(lab, ["T", "F"]), ["video_id", "title", "channel"]].copy()
        sub["lab"] = lab[np.isin(lab, ["T", "F"])]
        sub["batch"] = batch
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True).drop_duplicates("video_id", keep="last")
    out["channel"] = out["channel"].fillna("").astype(str)
    out["title"] = out["title"].fillna("").astype(str)
    return out.reset_index(drop=True)


def build_texts_df(df: pd.DataFrame) -> list[str]:
    title = df["title"].fillna("").astype(str)
    channel = df["channel"].fillna("").astype(str)
    return title.str.replace(r"\s+", " ", regex=True).str.strip().tolist()


def compute_centroids(emb: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    dim = emb.shape[1]
    cents = np.zeros((k, dim), dtype=np.float32)
    counts = np.zeros(k, dtype=np.int64)
    # chunked accumulate
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
    # cosine since L2-normalized: argmax dot
    sims = vecs @ cents.T
    return sims.argmax(axis=1).astype(np.int32)


def heuristic_tag(top_terms: str, top_channels: str, samples: list[str]) -> tuple[str, float]:
    """Return (tag, confidence) without LLM."""
    blob = f"{top_terms} {top_channels} " + " ".join(samples[:20])
    blob_l = blob.lower()
    if re.search(r"mukbang|eating\s*show|asmr\s*eat|吃播", blob_l):
        return "mukbang", 0.75
    if re.search(
        r"street\s*food|food\s*review|twesty|india\s*eat|hanoi\s*food|crazy\s*foody|"
        r"探店|street\s*eat",
        blob_l,
    ):
        return "food_review_street", 0.7
    if re.search(
        r"sky\s*news|big\s*brother|gameplay|makeup|workout|bodybuilding|"
        r"reality\s*show|only\s*connect",
        blob_l,
    ):
        return "certain_noise_noncook", 0.8
    if re.search(r"culinary\s*(academy|institute|school)|cookery\s*class|baking\s*class", blob_l):
        return "culinary_school", 0.65
    if re.search(r"village|grandpa|countryside|rural|乡村", blob_l):
        return "village_labor", 0.6
    if re.search(r"how\s*to|recipe|easy|masala|biryani|kitchen", blob_l):
        return "recipe_demo", 0.55
    return "other", 0.4


def try_llm_tags(cluster_dir: Path, summary: pd.DataFrame, samples: pd.DataFrame) -> dict[int, dict]:
    """If DASHSCOPE_API_KEY set, tag clusters via qwen; else {}."""
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        _log("no DASHSCOPE_API_KEY — skip LLM tagging")
        return {}
    try:
        from openai import OpenAI
    except ImportError:
        _log("openai pkg missing — skip LLM")
        return {}

    # reuse text qc base if present
    base = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    client = OpenAI(api_key=key, base_url=base, timeout=60.0)
    tags = {}
    for _, row in summary.iterrows():
        cid = int(row.cluster_id)
        samp = samples[samples.cluster_id == cid].head(8)
        lines = [
            f"- {str(r.title)[:80]} | {str(r.channel)[:40]}"
            for _, r in samp.iterrows()
        ]
        prompt = (
            "你是烹饪/餐饮视频质检助手。根据下列同簇标题+频道样例，给该簇一个标签，"
            "只能选其一: village_labor, culinary_school, recipe_demo, street_food, "
            "mukbang, food_review_street, certain_noise_noncook, other。\n"
            "只输出 JSON: {\"tag\":\"...\",\"confidence\":0-1,\"reason\":\"一句话\"}\n\n"
            + "\n".join(lines)
        )
        try:
            resp = client.chat.completions.create(
                model=os.getenv("DASHSCOPE_MODEL", "qwen-plus"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            text = resp.choices[0].message.content or ""
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                continue
            obj = json.loads(m.group(0))
            tags[cid] = {
                "tag": str(obj.get("tag", "other")),
                "confidence": float(obj.get("confidence", 0.5)),
                "reason": str(obj.get("reason", ""))[:200],
                "source": "llm",
            }
        except Exception as e:
            _log(f"LLM tag fail cluster {cid}: {e}")
    (cluster_dir / "cluster_llm_tags.json").write_text(
        json.dumps(tags, ensure_ascii=False, indent=2) + "\n"
    )
    return tags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cluster-dir",
        type=Path,
        default=PROJECT / "work/exo_cook_text_cluster_0917",
    )
    ap.add_argument(
        "--keep",
        type=Path,
        default=PROJECT
        / "data/runs/exo_cook/machine_0916/06_tools/text_gov_v03"
        / "烹饪教学_merged_0916_ml_keep_0917.csv",
    )
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    cdir = args.cluster_dir.resolve()
    keep_path = args.keep.resolve()

    emb = np.load(cdir / "embeddings.f16.npy", mmap_mode="r")
    lab_df = pd.read_parquet(cdir / "labels.parquet")
    # align labels to embedding row order via index
    index = pd.read_parquet(cdir / "index.parquet")
    if "row" in index.columns:
        index = index.sort_values("row").reset_index(drop=True)
    labels = index.merge(lab_df, on="video_id", how="left")["cluster_id"].to_numpy()
    if len(labels) != emb.shape[0]:
        # fallback: lab_df order equals encode order
        labels = lab_df["cluster_id"].to_numpy()
    k = int(labels.max()) + 1
    _log(f"centroids K={k} from {emb.shape[0]:,} vectors")
    cents = compute_centroids(emb, labels.astype(np.int32), k)
    np.save(cdir / "centroids.f32.npy", cents)

    gold = load_gold()
    keep_ids = set(
        pd.read_csv(keep_path, usecols=["video_id"], dtype=str)["video_id"].tolist()
    )
    gold["in_keep"] = gold["video_id"].isin(keep_ids)
    _log(f"gold TF={len(gold)} in_keep={gold.in_keep.sum()}")

    encoder = ck.MiniLMEncoder(batch_size=args.batch_size, device=args.device)
    encoder.fit([])
    texts = build_texts_df(gold)
    vecs = encoder.transform(texts).astype(np.float32)
    # L2 already from encode
    assigned = assign_nearest(vecs, cents)
    gold["cluster_id"] = assigned
    gold_out = cdir / "gold_cluster_assign.parquet"
    gold[["video_id", "lab", "batch", "in_keep", "cluster_id", "title", "channel"]].to_parquet(
        gold_out, index=False
    )

    summary = pd.read_csv(cdir / "cluster_summary.csv")
    samples = pd.read_csv(cdir / "cluster_samples.csv", dtype=str)

    # per-cluster gold stats (all mapped)
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

        # gold drop rules
        action = "keep"
        reason = ""
        if n_tf >= 8 and pass_rate is not None and pass_rate < 0.35:
            action = "drop_gold"
            reason = f"n_tf>={8} pass={pass_rate:.2f}<0.35"
        elif n_tf >= 5 and pass_rate == 0.0:
            action = "drop_gold"
            reason = f"n_tf>={5} pass=0"

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
    llm_tags = try_llm_tags(cdir, summary, samples)
    if llm_tags:
        stats["llm_tag"] = stats.cluster_id.map(lambda c: llm_tags.get(int(c), {}).get("tag"))
        stats["llm_conf"] = stats.cluster_id.map(
            lambda c: llm_tags.get(int(c), {}).get("confidence")
        )
    else:
        stats["llm_tag"] = None
        stats["llm_conf"] = None

    # If no gold drops, apply LLM/heuristic noise fallback
    gold_drops = set(stats.loc[stats.action == "drop_gold", "cluster_id"].astype(int))
    if not gold_drops:
        _log("no gold-rule drops — applying noise-tag fallback")
        for i, row in stats.iterrows():
            cid = int(row.cluster_id)
            tag = row.llm_tag or row.heuristic_tag
            conf = float(row.llm_conf) if pd.notna(row.llm_conf) else float(row.heuristic_conf)
            pr = row["pass"]
            if tag not in NOISE_TAGS:
                continue
            if conf < 0.65:
                continue
            if pr is not None and float(pr) > 0.50:
                continue  # gold says mostly T
            # no gold or pass<=50%
            if row.n_tf == 0 or (pr is not None and float(pr) <= 0.50):
                stats.at[i, "action"] = "drop_tag"
                stats.at[i, "action_reason"] = f"tag={tag} conf={conf:.2f} pass={pr}"

    stats_path = cdir / "cluster_gold_stats.csv"
    stats.to_csv(stats_path, index=False)
    drop_ids = set(stats.loc[stats.action.str.startswith("drop"), "cluster_id"].astype(int))
    _log(f"drop clusters: {sorted(drop_ids)} ({len(drop_ids)})")

    # filter keep
    keep_labels = lab_df.set_index("video_id")["cluster_id"]
    # stream filter with duckdb
    import duckdb

    out_keep = keep_path.with_name(keep_path.stem.replace("_ml_keep", "_ml_keep_cluster_filt") + ".csv")
    if "_ml_keep" not in keep_path.stem:
        out_keep = keep_path.with_name(keep_path.stem + "_cluster_filt.csv")

    # write drop video list
    drop_vids = lab_df[lab_df.cluster_id.isin(drop_ids)]["video_id"].astype(str)
    drop_vid_path = cdir / "drop_cluster_video_ids.txt"
    drop_vid_path.write_text("\n".join(drop_vids.tolist()) + "\n")

    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE drop_v(video_id VARCHAR)")
    if len(drop_vids):
        con.executemany(
            "INSERT INTO drop_v VALUES (?)", [(v,) for v in drop_vids.tolist()]
        )

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
        # copy keep as-is for consistent artifact
        import shutil

        shutil.copy2(keep_path, out_keep)

    n1, h1 = con.execute(
        f"""
        SELECT COUNT(*), COALESCE(SUM(TRY_CAST(duration_seconds AS DOUBLE)),0)/3600.0
        FROM read_csv_auto('{out_keep}', header=true, all_varchar=true)
        """
    ).fetchone()

    report = {
        "gold_n": int(len(gold)),
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
        "llm_used": bool(llm_tags),
        "note": "vectors=title; clusters=semantic not T/F; gold map uses nearest centroid",
    }
    (cdir / "gold_filter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

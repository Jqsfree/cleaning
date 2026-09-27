#!/usr/bin/env python3
"""exo_service 理发缩略图分类器 — 冻结 CLIP + Logistic Regression（人标 T/F）。"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "02_脚本"))

from core.exemplar_sim import ClipEncoder, fetch_thumbnails_batch  # noqa: E402
from core.visual_filter import (  # noqa: E402
    assign_actions,
    build_feature_matrix,
    choose_action_thresholds,
    load_embedding_rows,
    split_labeled_frame,
    train_grouped_visual_model,
    write_embedding_store,
)
from PIL import Image  # noqa: E402

DEFAULT_LABELS = [
    _ROOT / "data/runs/exo_service/machine_0813/03_qc/human270_hair_clip_c90_e4585263.csv",
]
FALLBACK_TMP = Path("/home/jqs/tmp/商业服务理发_e4585263_qc_result.csv")

MODEL_OUT = _ROOT / "models/exo_service_hair_thumb_lr.pkl"
EMBED_DIR = _ROOT / "data/assets/embeddings/exo_service_hair_human"


def _resolve_label_paths(extra: list[str] | None) -> list[Path]:
    paths: list[Path] = []
    for p in DEFAULT_LABELS + [FALLBACK_TMP]:
        if p.exists() and p not in paths:
            paths.append(p)
    if extra:
        for raw in extra:
            path = Path(raw)
            if path.exists() and path not in paths:
                paths.append(path)
    if not paths:
        raise FileNotFoundError("未找到人标 CSV；请用 --labels 指定")
    return paths


def load_human_labels(
    paths: list[Path],
    *,
    pools: set[str] | None = None,
) -> pd.DataFrame:
    """合并人标；qc_result=T/F，qc_status=ok。"""
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path, dtype=str, low_memory=False)
        if "qc_status" in df.columns:
            df = df[df["qc_status"].fillna("ok").astype(str).str.lower().eq("ok")]
        col = "qc_result" if "qc_result" in df.columns else "label"
        df["qc_result"] = df[col].astype(str).str.strip().str.upper()
        df = df[df["qc_result"].isin(["T", "F"])].copy()
        pool = path.stem
        if pools and pool not in pools and path.name not in pools:
            # allow matching by filename fragment
            if not any(x in path.name for x in pools):
                continue
        df["label_pool"] = pool
        frames.append(df)
    if not frames:
        raise ValueError("过滤后无标签")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates("video_id", keep="first")
    out["video_id"] = out["video_id"].astype(str).str.strip()
    return out


def encode_video_ids(
    video_ids: list[str],
    *,
    cache_dir: Path,
    batch_size: int = 64,
    thumb_workers: int = 16,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (embeddings float32, ok_mask bool)。"""
    encoder = ClipEncoder("ViT-B-32", "openai", device=device)
    n = len(video_ids)
    feats = np.zeros((n, 512), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    paths = fetch_thumbnails_batch(video_ids, cache_dir, workers=thumb_workers)
    batch_idx: list[int] = []
    batch_imgs: list[Image.Image] = []
    for i, path in enumerate(paths):
        if path is None:
            continue
        batch_idx.append(i)
        batch_imgs.append(Image.open(path).convert("RGB"))
        if len(batch_imgs) >= batch_size:
            vecs = encoder.encode_images(batch_imgs)
            for j, row in enumerate(batch_idx):
                feats[row] = vecs[j]
                ok[row] = True
            batch_idx, batch_imgs = [], []
    if batch_imgs:
        vecs = encoder.encode_images(batch_imgs)
        for j, row in enumerate(batch_idx):
            feats[row] = vecs[j]
            ok[row] = True
    return feats, ok


def prepare_labels(labels: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    work = labels.copy()
    if "channel" not in work.columns:
        work["channel"] = work.get("source_ref", work["video_id"])
    work = split_labeled_frame(work, group_col="channel", seed=seed)
    work["qc_bool"] = work["qc_result"].eq("T")
    return work


def train_model(
    labels: pd.DataFrame,
    *,
    embedding_store: Path,
    target_pass_rate: float,
    max_overturn: float,
    seed: int,
) -> dict:
    work = prepare_labels(labels, seed=seed)
    usable = work[work["split"].isin(["train", "calibration", "holdout"])].copy()
    ids = usable["video_id"].tolist()

    if embedding_store.exists() and (embedding_store / "embeddings.npy").exists():
        embs, found = load_embedding_rows(embedding_store, ids)
    else:
        embedding_store.mkdir(parents=True, exist_ok=True)
        feats, ok_mask = encode_video_ids(ids, cache_dir=_ROOT / "qc_thumb_cache/exemplar_sim")
        found_idx = [i for i, vid in enumerate(ids) if ok_mask[i]]
        found = [ids[i] for i in found_idx]
        embs = feats[found_idx]
        write_embedding_store(embedding_store, found, embs)

    order = {vid: i for i, vid in enumerate(found)}
    usable = usable[usable["video_id"].isin(found)].copy()
    usable["_ord"] = usable["video_id"].map(order)
    usable = usable.sort_values("_ord").drop(columns="_ord")
    if len(usable) != len(embs):
        raise RuntimeError("embedding 对齐失败")

    duration = pd.to_numeric(usable.get("duration_seconds"), errors="coerce")
    features = build_feature_matrix(
        embs,
        pos_sim=np.zeros(len(embs)),
        neg_sim=np.zeros(len(embs)),
        duration_seconds=duration.fillna(0),
    )
    y = usable["qc_bool"].astype(int).to_numpy()
    groups = usable.get("label_group", usable["channel"]).fillna(usable["video_id"]).astype(str)

    fit_mask = usable["split"].isin(["train", "calibration"]).to_numpy()
    holdout_mask = usable["split"].eq("holdout").to_numpy()
    cal_mask = usable["split"].eq("calibration").to_numpy()

    model, oof = train_grouped_visual_model(
        features[fit_mask], y[fit_mask], groups[fit_mask].to_numpy(), seed=seed,
    )
    cal_oof = np.full(len(y), np.nan)
    cal_oof[fit_mask] = oof
    cal_probs = cal_oof[cal_mask]
    cal_y = y[cal_mask]
    if len(cal_y) >= 10 and len(np.unique(cal_y)) == 2:
        thresholds = choose_action_thresholds(
            cal_y, cal_probs,
            target_pass_rate=target_pass_rate,
            max_overturn=max_overturn,
            min_keep_labels=min(15, cal_mask.sum()),
            min_drop_labels=min(15, cal_mask.sum()),
        )
    else:
        thresholds = choose_action_thresholds(
            y[fit_mask], oof,
            target_pass_rate=target_pass_rate,
            max_overturn=max_overturn,
            min_keep_labels=min(15, fit_mask.sum()),
            min_drop_labels=min(15, fit_mask.sum()),
        )

    metrics: dict = {
        "n_labels": int(len(usable)),
        "n_T": int((y == 1).sum()),
        "n_F": int((y == 0).sum()),
        "n_thumb_ok": int(len(embs)),
        "oof_auc": float(roc_auc_score(y[fit_mask], oof)),
        "thresholds": thresholds,
    }
    if holdout_mask.any() and len(np.unique(y[holdout_mask])) == 2:
        holdout_p = model.predict_proba(features[holdout_mask])[:, 1]
        metrics["holdout_auc"] = float(roc_auc_score(y[holdout_mask], holdout_p))
        pred = assign_actions(
            holdout_p,
            keep_threshold=float(thresholds["keep_threshold"]),
            drop_threshold=float(thresholds["drop_threshold"]),
        )
        keep_pred = pred == "keep_candidate"
        if keep_pred.any():
            metrics["holdout_keep_pass_rate"] = float(y[holdout_mask][keep_pred].mean())
            metrics["holdout_keep_n"] = int(keep_pred.sum())

    artifact = {
        "category": "exo_service_hair",
        "modality": "thumb_clip_lr",
        "model": model,
        "keep_threshold": float(thresholds["keep_threshold"]),
        "drop_threshold": float(thresholds["drop_threshold"]),
        "thresholds": thresholds,
        "feature_layout": "clip_embedding,log1p_duration",
        "embedding_dim": 512,
        "metrics": metrics,
        "label_pools": usable["label_pool"].value_counts().to_dict(),
    }
    return artifact


def apply_model(
    input_csv: Path,
    output_dir: Path,
    artifact: dict,
    *,
    cache_dir: Path,
    batch_rows: int = 5000,
    encode_batch: int = 64,
    thumb_workers: int = 16,
    chunksize: int = 0,
    device: str | None = None,
) -> dict:
    """对池子打分；输出 keep / drop / scored CSV。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = artifact["model"]
    keep_thr = float(artifact["keep_threshold"])
    drop_thr = float(artifact["drop_threshold"])

    if chunksize > 0:
        reader = pd.read_csv(input_csv, dtype=str, low_memory=False, chunksize=chunksize)
        frames = list(reader)
        frame = pd.concat(frames, ignore_index=True)
    else:
        frame = pd.read_csv(input_csv, dtype=str, low_memory=False)
    frame = frame.drop_duplicates("video_id").copy()
    frame["video_id"] = frame["video_id"].astype(str).str.strip()
    ids = frame["video_id"].tolist()

    enc_device = None if device == "auto" else device
    encoder = ClipEncoder("ViT-B-32", "openai", device=enc_device)
    all_scores: list[float] = []
    all_actions: list[str] = []
    all_ok: list[bool] = []

    for start in range(0, len(ids), batch_rows):
        chunk_ids = ids[start : start + batch_rows]
        sub = frame[frame["video_id"].isin(chunk_ids)].copy()
        feats, ok_mask = encode_video_ids(
            chunk_ids,
            cache_dir=cache_dir,
            batch_size=encode_batch,
            thumb_workers=thumb_workers,
            device=str(encoder.device),
        )
        duration = pd.to_numeric(
            sub.set_index("video_id").reindex(chunk_ids)["duration_seconds"],
            errors="coerce",
        ).fillna(0)
        x = build_feature_matrix(
            feats,
            pos_sim=np.zeros(len(chunk_ids)),
            neg_sim=np.zeros(len(chunk_ids)),
            duration_seconds=duration.to_numpy(),
        )
        probs = np.full(len(chunk_ids), np.nan)
        if ok_mask.any():
            probs[ok_mask] = model.predict_proba(x[ok_mask])[:, 1]
        actions = assign_actions(probs, keep_threshold=keep_thr, drop_threshold=drop_thr)
        all_scores.extend(probs.tolist())
        all_actions.extend(actions.tolist())
        all_ok.extend(ok_mask.tolist())
        print(f"  scored {min(start + batch_rows, len(ids))}/{len(ids)}", flush=True)

    result = frame.copy()
    result["ml_score"] = all_scores
    result["ml_action"] = all_actions
    result["thumb_ok"] = all_ok

    date_tag = time.strftime("%m%d")
    scored = out / f"hair_thumb_lr_scored_{date_tag}.csv"
    keep = out / f"hair_thumb_lr_keep_{date_tag}.csv"
    drop = out / f"hair_thumb_lr_drop_{date_tag}.csv"
    result.to_csv(scored, index=False)
    result[result["ml_action"] == "keep_candidate"].to_csv(keep, index=False)
    result[result["ml_action"] == "highconf_drop"].to_csv(drop, index=False)

    n = len(result)
    summary = {
        "input": str(input_csv),
        "n": n,
        "keep_candidate": int((result["ml_action"] == "keep_candidate").sum()),
        "highconf_drop": int((result["ml_action"] == "highconf_drop").sum()),
        "uncertain": int((result["ml_action"] == "uncertain").sum()),
        "no_thumb": int((~result["thumb_ok"].astype(bool)).sum()),
        "keep_threshold": keep_thr,
        "drop_threshold": drop_thr,
        "scored_csv": str(scored),
        "keep_csv": str(keep),
        "drop_csv": str(drop),
    }
    (out / f"hair_thumb_lr_summary_{date_tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summary


def cmd_train(args: argparse.Namespace) -> None:
    paths = _resolve_label_paths(args.labels)
    pools = set(args.pools.split(",")) if args.pools else None
    labels = load_human_labels(paths, pools=pools)
    print(f"labels: n={len(labels)} T={(labels.qc_result=='T').sum()} F={(labels.qc_result=='F').sum()}")
    artifact = train_model(
        labels,
        embedding_store=Path(args.embeddings),
        target_pass_rate=args.target_pass_rate,
        max_overturn=args.max_overturn,
        seed=args.seed,
    )
    MODEL_OUT.parent.mkdir(exist_ok=True)
    with MODEL_OUT.open("wb") as f:
        pickle.dump(artifact, f)
    report_path = Path(args.output_dir) / "train_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {k: v for k, v in artifact.items() if k != "model"}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"model → {MODEL_OUT}")


def cmd_apply(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}；先 --train")
    with model_path.open("rb") as f:
        artifact = pickle.load(f)
    summary = apply_model(
        Path(args.input),
        Path(args.output_dir),
        artifact,
        cache_dir=Path(args.cache_dir),
        batch_rows=args.batch_rows,
        chunksize=args.chunksize,
        device=None if args.device == "auto" else args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="exo_service 理发缩略图 LR（人标 CLIP embedding）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="用人标 T/F 训练")
    tr.add_argument("--labels", action="append", help="人标 CSV，可多次指定")
    tr.add_argument("--pools", help="仅保留文件名含这些片段的池，逗号分隔")
    tr.add_argument("--embeddings", default=str(EMBED_DIR))
    tr.add_argument("-o", "--output-dir", default=str(_ROOT / "data/assets/models/exo_service_hair_thumb"))
    tr.add_argument("--target-pass-rate", type=float, default=0.85)
    tr.add_argument("--max-overturn", type=float, default=0.08)
    tr.add_argument("--seed", type=int, default=42)
    tr.set_defaults(func=cmd_train)

    aply = sub.add_parser("apply", help="对 CSV 池过滤")
    aply.add_argument("input")
    aply.add_argument("-o", "--output-dir", required=True)
    aply.add_argument("--model", default=str(MODEL_OUT))
    aply.add_argument("--cache-dir", default=str(_ROOT / "qc_thumb_cache/exemplar_sim"))
    aply.add_argument("--batch-rows", type=int, default=5000)
    aply.add_argument("--chunksize", type=int, default=0)
    aply.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    aply.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

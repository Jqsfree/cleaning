#!/usr/bin/env python3
"""建筑施工 CLIP 辅助过滤：用人工 T/F 图像样本训练风险模型，输出通过表与复核/过滤候选。

谨慎策略：默认高风险记录进入 clip_review；--drop-high-risk 才把候选从 pass 表移出。
原始输入不会被覆盖。CLIP 分数不可用（无图/坏图）的记录保留并标记 no_thumbnail。
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys, time, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "02_脚本"
sys.path.insert(0, str(SCRIPTS))
from core.exemplar_sim import ClipEncoder, fetch_thumbnails_batch

EVAL = ROOT / "work/exo_construction_0928/clip_eval"
INPUT = ROOT / "data/runs/exo_construction/machine_0923/05_clean/run08_v023/建筑施工_merged_0923_clean_0928.csv"
CACHE = ROOT / "qc_thumb_cache/exemplar_sim"
THRESHOLD = 0.9926  # OOF: CLIP-only, weighted T-hurt <=2% (observed 1.28%)


def load_models() -> tuple[list[LogisticRegression], dict[str, float]]:
    lab = pd.read_csv(EVAL / "labeled_sample.csv", dtype={"video_id": str}, low_memory=False)
    X = np.load(EVAL / "clip_image_embeddings_vit_b32.npy").astype(np.float32)
    if len(lab) != len(X):
        raise ValueError(f"样本/embedding 行数不同: {len(lab)} vs {len(X)}")
    lab["qc_result"] = lab["qc_result"].astype(str).str.upper().str.strip()
    good = lab["qc_result"].isin(["T", "F"]).to_numpy()
    y = (lab.loc[good, "qc_result"].to_numpy() == "F").astype(int)
    weights = pd.to_numeric(lab.loc[good, "sample_weight"], errors="coerce").fillna(1).to_numpy(float)
    groups = lab.loc[good, "channel"].fillna("").astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=4242)
    models = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        for train_idx, _ in splitter.split(X[good], y, groups):
            model = LogisticRegression(C=0.1, max_iter=2000, solver="liblinear", class_weight=None, random_state=4242)
            model.fit(X[good][train_idx], y[train_idx], sample_weight=weights[train_idx])
            models.append(model)
    return models, {"n_train": int(good.sum()), "n_T": int((y == 0).sum()), "n_F": int((y == 1).sum()), "fold_models": len(models), "fold_method": "StratifiedGroupKFold(seed=4242); liblinear reproduces saved OOF scores"}


def model_for_id(models: list[LogisticRegression], video_id: str) -> LogisticRegression:
    digest = hashlib.sha1(video_id.encode("utf-8")).digest()
    return models[int.from_bytes(digest[:4], "big") % len(models)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", nargs="?", type=Path, default=INPUT)
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "work/exo_construction_0928/clip_filter_run01")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--thumb-workers", type=int, default=128)
    ap.add_argument("--thumb-timeout", type=float, default=4.0)
    ap.add_argument("--resume", action="store_true", help="验证已有 scored 文件是输入前缀后，从断点续跑")
    ap.add_argument("--limit", type=int, default=0, help="调试/试跑行数；0 表示全量")
    args = ap.parse_args()
    t0 = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    models, train_summary = load_models()
    import torch
    clip_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    encoder = ClipEncoder("ViT-B-32", "openai", device=clip_device)
    source = pd.read_csv(args.input, dtype={"video_id": str}, low_memory=False)
    if args.limit:
        source = source.sample(n=min(args.limit, len(source)), random_state=4242).sort_index().reset_index(drop=True)
    if "video_id" not in source:
        raise ValueError("输入缺少 video_id")
    pass_path = args.output / "建筑施工_clip_pass.csv"
    review_path = args.output / "建筑施工_clip_review.csv"
    drop_path = args.output / "建筑施工_clip_drop.csv"
    scored_path = args.output / "建筑施工_clip_scored.csv"
    start_idx = 0
    n_seen = n_flagged = n_ok = 0
    if args.resume:
        if args.limit:
            raise ValueError("--resume 不能和 --limit 同时用")
        if not scored_path.is_file():
            raise FileNotFoundError(f"续跑文件不存在: {scored_path}")
        prior = pd.read_csv(scored_path, usecols=["video_id", "clip_action"], dtype={"video_id": str}, low_memory=False, lineterminator="\n")
        start_idx = len(prior)
        if start_idx > len(source) or not prior["video_id"].reset_index(drop=True).equals(
            source.loc[:start_idx-1, "video_id"].astype(str).reset_index(drop=True)
        ):
            raise ValueError("已有 scored 文件不是当前输入的完整前缀，拒绝续跑")
        n_seen = start_idx
        n_flagged = int(prior["clip_action"].eq("clip_review").sum())
        n_ok = int(prior["clip_action"].ne("no_thumbnail").sum())
    else:
        for p in (pass_path, review_path, drop_path, scored_path):
            p.unlink(missing_ok=True)
    first = not scored_path.is_file()
    for start in range(start_idx, len(source), args.chunk_size):
        part = source.iloc[start:start + args.chunk_size].copy()
        ids = part["video_id"].fillna("").astype(str).tolist()
        paths = fetch_thumbnails_batch(ids, CACHE, workers=args.thumb_workers, timeout=args.thumb_timeout)
        feats = np.zeros((len(part), 512), dtype=np.float32)
        ok = np.zeros(len(part), dtype=bool)
        img_idx, imgs = [], []
        for i, path in enumerate(paths):
            if path is None:
                continue
            try:
                with Image.open(path) as im:
                    imgs.append(im.convert("RGB"))
                img_idx.append(i)
            except Exception:
                continue
            if len(imgs) >= 64:
                feats[img_idx] = encoder.encode_images(imgs)
                ok[img_idx] = True
                img_idx, imgs = [], []
        if imgs:
            feats[img_idx] = encoder.encode_images(imgs)
            ok[img_idx] = True
        scores = np.full(len(part), np.nan, dtype=float)
        if ok.any():
            fold_ids = np.array([int.from_bytes(hashlib.sha1(ids[ix].encode("utf-8")).digest()[:4], "big") % len(models) for ix in range(len(part))])
            for fold_id in range(len(models)):
                idx = np.flatnonzero(ok & (fold_ids == fold_id))
                if len(idx):
                    scores[idx] = models[fold_id].predict_proba(feats[idx])[:, 1]
        part["clip_f_risk"] = scores
        part["clip_filter_threshold"] = args.threshold
        part["clip_action"] = np.where(~ok, "no_thumbnail", np.where(scores >= args.threshold, "clip_review", "clip_pass"))
        flagged = part["clip_action"].eq("clip_review")
        # CSV serializer does not reliably quote lone CR; normalize embedded line separators before writing.
        for col in part.select_dtypes(include=["object", "string"]).columns:
            part[col] = part[col].map(lambda v: re.sub(r"[\r\n\u2028\u2029]+", " ", v) if isinstance(v, str) else v)
        keep = part["clip_action"].eq("clip_pass")
        drop = ~keep  # user's policy: only CLIP pass stays in keep; all others go to drop
        part.loc[keep].to_csv(pass_path, mode="a", index=False, header=first)
        part.loc[drop].to_csv(drop_path, mode="a", index=False, header=first)
        part.loc[flagged].to_csv(review_path, mode="a", index=False, header=first)
        part.to_csv(scored_path, mode="a", index=False, header=first)
        first = False
        n_seen += len(part); n_flagged += int(flagged.sum()); n_ok += int(ok.sum())
        print(f"progress={n_seen:,}/{len(source):,} thumbnail_ok={n_ok:,} high_risk={n_flagged:,}", flush=True)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "input": str(args.input.resolve()),
        "n_input": len(source), "resumed_from_row": start_idx, "n_thumbnail_ok": n_ok, "n_no_thumbnail": len(source)-n_ok,
        "n_high_risk": n_flagged, "device": clip_device, "high_risk_rate": round(n_flagged/max(len(source),1), 6),
        "threshold": args.threshold, "threshold_basis": "CLIP-only 5-fold channel-group OOF; weighted T-hurt=1.28%, F recall=7.84%",
        "action": "keep_clip_pass_only; drop_clip_review_and_no_thumbnail",
        "training": train_summary, "model": "five fold-held-out weighted LogisticRegression(C=0.1, solver=liblinear) on OpenCLIP ViT-B-32/openai image embeddings; model assigned by SHA1(video_id) mod 5",
        "thumb_timeout_seconds": args.thumb_timeout, "elapsed_seconds": round(time.perf_counter()-t0, 1), "outputs": {"pass": str(pass_path), "drop": str(drop_path), "review": str(review_path), "scored": str(scored_path)},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

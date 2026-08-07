#!/usr/bin/env python3
"""
film_tv 多模态融合 — 文本 TF-IDF + 图像 CNN 特征 → LightGBM

用法:
  python3 experiments/film_tv_multimodal.py --extract    # 提取特征（先跑）
  python3 experiments/film_tv_multimodal.py --train       # 训练融合模型
"""

import sys, os, pickle, argparse, time
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    precision_recall_curve, f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "experiments"))
from film_tv_text_classifier import DualTfidfVectorizer, build_text

MODEL_DIR = _project_root / "models"
THUMB_CACHE = _project_root / "qc_thumb_cache"
FEATURES_DIR = _project_root / "data/runs/film_tv/ml"
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
IMG_SIZE = 224

THUMB_QC_FILES = [
    "data/runs/film_tv/005_clean/影视剧-播放列表_771f94e0_records/run05/影视剧-播放列表_771f94e0_records_run01_keep_high_clean_0722_thumb_qc.csv",
    "data/runs/film_tv/005_clean/影视剧-频道_0316a650_records/run05/影视剧-频道_0316a650_records_clean_0722_thumb_qc.csv",
    "data/runs/film_tv/005_clean/影视剧核心片段批量_4292e310_records/run01/影视剧核心片段批量_4292e310_records_clean_0722_thumb_qc.csv",
    "data/runs/film_tv/005_clean/影视剧核心片段批量-无负面_096a9099_records/run01/影视剧核心片段批量-无负面_096a9099_records_clean_0722_thumb_qc.csv",
    "data/runs/film_tv/005_clean/影视剧_31768118_records/run01/影视剧_31768118_records_run01_keep_thumb_qc.csv",
]


# ══════════════════════════════════════════════════════════════
# 特征提取
# ══════════════════════════════════════════════════════════════

class FeatureDataset(Dataset):
    def __init__(self, video_ids: list[str], transform):
        self.video_ids = video_ids
        self.transform = transform

    def _find_thumbnail(self, video_id: str) -> str | None:
        for suffix in ["maxresdefault", "hqdefault", "mqdefault", "sddefault", "0"]:
            path = THUMB_CACHE / f"{video_id}_{suffix}.jpg"
            if path.exists() and path.stat().st_size >= 1500:
                return str(path)
        return None

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        img_path = self._find_thumbnail(vid)
        if img_path is None:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        else:
            img = Image.open(img_path).convert("RGB")
        return self.transform(img)


def build_image_encoder() -> torch.nn.Module:
    """与训练脚本完全一致的模型结构，加载权重后去掉最后两层取特征。"""
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    # 与 film_tv_thumb_train.py 完全相同的修改
    model.classifier[-1] = nn.Identity()
    model.classifier = nn.Sequential(
        model.classifier,
        nn.Linear(in_features, 1),
        nn.Sigmoid(),
    )
    state = torch.load(MODEL_DIR / "film_tv_thumb_clf.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    # 取 classifier[0] 的输出作为图像特征（576 维，Identity 前的池化输出）
    model.classifier = model.classifier[0]
    return model.to(DEVICE).eval()


@torch.no_grad()
def extract_image_features(video_ids: list[str], batch_size: int = 64) -> np.ndarray:
    """提取图像特征矩阵 (N, 576)"""
    encoder = build_image_encoder()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = FeatureDataset(video_ids, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    features = []
    for imgs in loader:
        imgs = imgs.to(DEVICE)
        feats = encoder(imgs).squeeze()
        features.append(feats.cpu().numpy())

    return np.vstack(features)


# ══════════════════════════════════════════════════════════════
# 数据构建
# ══════════════════════════════════════════════════════════════

# 文本特征需要从 textqc 文件加载，因为只有这些有 text label
# 图像特征从 thumb_qc 加载，因为只有这些有 thumb label
# 我们需要交集：同时有 text 和 thumb label 的数据
# 但交集很小（~7K）。改用 thumb label 作为训练 label，并采样。

def build_multimodal_data(max_samples: int = 40000):
    """构建包含文本+图像特征+label 的训练数据。"""
    import pickle as _pickle

    # 加载文本模型
    text_pipe = _pickle.load(open(MODEL_DIR / "film_tv_text_clf_tfidf.pkl", "rb"))

    # 加载 thumb QC 数据
    print("加载 vision QC 标注...")
    frames = []
    for f in THUMB_QC_FILES:
        path = _project_root / f
        if not path.exists():
            continue
        df = pd.read_csv(str(path))
        labeled = df[df["qc_thumb_result"].isin(["T", "F"])].copy()
        frames.append(labeled)

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.drop_duplicates("video_id", keep="first")

    # 平衡采样
    t_df = df_all[df_all["qc_thumb_result"] == "T"]
    f_df = df_all[df_all["qc_thumb_result"] == "F"]
    n = min(max_samples // 2, len(t_df), len(f_df))
    sampled = pd.concat([
        t_df.sample(n=n, random_state=RANDOM_SEED),
        f_df.sample(n=n, random_state=RANDOM_SEED),
    ], ignore_index=True)
    print(f"采样: T={n:,}  F={n:,}  total={len(sampled):,}")

    # 文本特征
    print("提取文本特征...")
    texts = sampled.apply(build_text, axis=1).tolist()
    X_text = text_pipe.named_steps["tfidf"].transform(texts)

    # 图像特征
    print(f"提取图像特征 ({len(sampled):,} 张缩略图)...")
    t0 = time.time()
    X_image = extract_image_features(sampled["video_id"].tolist(), batch_size=64)
    print(f"  耗时: {time.time()-t0:.0f}s  shape={X_image.shape}")

    y = (sampled["qc_thumb_result"] == "T").astype(int).values
    return X_text, X_image, y, sampled


# ══════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════

def train_multimodal():
    print("=" * 60)
    print("film_tv 多模态融合 — TF-IDF + CNN → LogisticRegression")
    print("=" * 60)

    t0 = time.time()
    X_text, X_image, y, df = build_multimodal_data(max_samples=40000)
    n = len(y)
    n_pos = y.sum()
    print(f"\n数据: {n:,} 条  T={n_pos:,}  F={n-n_pos:,}  [{time.time()-t0:.0f}s]")

    # 图像特征降维
    print("\n图像特征 PCA (576→200)...")
    svd = TruncatedSVD(n_components=200, random_state=RANDOM_SEED)
    X_image_reduced = svd.fit_transform(X_image)
    print(f"  解释方差: {svd.explained_variance_ratio_.sum():.3f}")

    # 文本特征也降维 (原始 ~16K 维太稀疏)
    print("文本特征 SVD (→500)...")
    text_svd = TruncatedSVD(n_components=500, random_state=RANDOM_SEED)
    X_text_dense = text_svd.fit_transform(X_text)
    print(f"  解释方差: {text_svd.explained_variance_ratio_.sum():.3f}")

    # 拼接特征
    X_combined = np.hstack([X_text_dense, X_image_reduced])

    # ── 单模态 baseline ──
    print("\n--- 单模态 baseline ---")

    for name, X in [("text-only", X_text_dense), ("image-only", X_image_reduced), ("multimodal", X_combined)]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", lr)])

        scoring = {"auc": "roc_auc", "ap": "average_precision", "f1": "f1"}
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)

        print(f"  {name:16s}: AUC={scores['test_auc'].mean():.4f} (±{scores['test_auc'].std():.4f})  "
              f"AP={scores['test_ap'].mean():.4f} (±{scores['test_ap'].std():.4f})  "
              f"F1={scores['test_f1'].mean():.4f}")

    # ── Hold-out 详细评估 ──
    print("\n--- Hold-out 评估 (multimodal) ---")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_combined, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)),
    ])
    pipe.fit(X_tr, y_tr)
    y_proba = pipe.predict_proba(X_te)[:, 1]
    y_pred = pipe.predict(X_te)

    print(classification_report(y_te, y_pred, target_names=["F(非影视)", "T(影视剧)"]))
    auc = roc_auc_score(y_te, y_proba)
    ap = average_precision_score(y_te, y_proba)
    print(f"Multimodal: AUC={auc:.4f}  AP={ap:.4f}")

    # ── 阈值 ──
    print("\n--- 阈值分析 (multimodal) ---")
    precisions, recalls, thresholds = precision_recall_curve(y_te, y_proba)
    for tr in [0.95, 0.90, 0.85, 0.80, 0.75]:
        idx = np.argmin(np.abs(recalls[:-1] - tr))
        print(f"  recall={tr:.2f}: thr={thresholds[idx]:.3f}  prec={precisions[idx]:.3f}  f1={f1_score(y_te, (y_proba>=thresholds[idx]).astype(int)):.4f}")

    # ── 保存 ──
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "film_tv_multimodal_clf.pkl", "wb") as f:
        pickle.dump({
            "pipe": pipe, "text_svd": text_svd, "image_svd": svd,
            "text_model_path": str(MODEL_DIR / "film_tv_text_clf_tfidf.pkl"),
            "image_model_path": str(MODEL_DIR / "film_tv_thumb_clf.pth"),
        }, f)
    print(f"\n模型已保存: models/film_tv_multimodal_clf.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    train_multimodal()


import torch.nn as nn

if __name__ == "__main__":
    main()

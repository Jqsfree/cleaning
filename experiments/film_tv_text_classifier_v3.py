#!/usr/bin/env python3
"""
film_tv 文本分类器 V3 — TF-IDF + 数值 + 图像 → LightGBM

三路特征融合:
  文本: TF-IDF word(1-2) + char(2-4) → 17K 维
  数值: log_dur + ch_score + ch_log_freq → 3 维
  图像: MobileNetV3 pool 层 → PCA 128 维

用法:
  python3 experiments/film_tv_text_classifier_v3.py --extract-images   # 预提取图像特征(先跑)
  python3 experiments/film_tv_text_classifier_v3.py --train             # 训练
  python3 experiments/film_tv_text_classifier_v3.py --score PATH        # 打分
"""

import sys, os, re, glob, argparse, time, pickle
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import sparse

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    average_precision_score, f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "experiments"))
sys.path.insert(0, str(_project_root / "02_脚本"))

from film_tv_text_classifier import load_labeled_data, build_text, DualTfidfVectorizer

MODEL_DIR = _project_root / "models"
THUMB_CACHE = _project_root / "qc_thumb_cache"
FEAT_CACHE = _project_root / "data/runs/film_tv/ml"
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 64


# ══════════════════════════════════════════════════════════════
# 图像特征提取
# ══════════════════════════════════════════════════════════════

class ImageDataset(Dataset):
    def __init__(self, video_ids, transform):
        self.video_ids = video_ids
        self.transform = transform
    def _find(self, vid):
        for s in ["maxresdefault","hqdefault","mqdefault","sddefault","0"]:
            p = THUMB_CACHE / f"{vid}_{s}.jpg"
            if p.exists() and p.stat().st_size >= 1500:
                return str(p)
        return None
    def __len__(self): return len(self.video_ids)
    def __getitem__(self, i):
        p = self._find(self.video_ids[i])
        img = Image.open(p).convert("RGB") if p else Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0,0,0))
        return self.transform(img)


def build_image_encoder():
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Identity()
    model.classifier = torch.nn.Sequential(model.classifier, torch.nn.Linear(in_features, 1), torch.nn.Sigmoid())
    state = torch.load(MODEL_DIR / "film_tv_thumb_clf.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.classifier = model.classifier[0]  # 取 pool 层 → 576 维
    return model.to(DEVICE).eval()


@torch.no_grad()
def extract_image_features(video_ids, batch_size=64):
    encoder = build_image_encoder()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = ImageDataset(video_ids, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    feats = []
    for imgs in loader:
        feats.append(encoder(imgs.to(DEVICE)).squeeze().cpu().numpy())
    return np.vstack(feats)


def get_image_features(video_ids, cache_name):
    """从缓存加载或提取图像特征"""
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = FEAT_CACHE / f"{cache_name}_img_feats.npy"
    vid_path = FEAT_CACHE / f"{cache_name}_img_vids.npy"
    if cache_path.exists():
        cached_vids = np.load(vid_path)
        cached_feats = np.load(cache_path)
        vid_to_feat = {v: cached_feats[i] for i, v in enumerate(cached_vids)}
        new_vids = [v for v in video_ids if v not in vid_to_feat]
        if not new_vids:
            return np.array([vid_to_feat[v] for v in video_ids])
        print(f"  提取 {len(new_vids):,} 张新缩略图...")
        new_feats = extract_image_features(new_vids, batch_size=64)
        all_vids = np.concatenate([cached_vids, np.array(new_vids)])
        all_feats = np.vstack([cached_feats, new_feats])
        np.save(vid_path, all_vids)
        np.save(cache_path, all_feats)
        vid_to_feat = {v: all_feats[i] for i, v in enumerate(all_vids)}
        return np.array([vid_to_feat[v] for v in video_ids])
    else:
        feats = extract_image_features(video_ids, batch_size=64)
        np.save(vid_path, np.array(video_ids))
        np.save(cache_path, feats)
        return feats


# ══════════════════════════════════════════════════════════════
# 数值特征 (同 V2)
# ══════════════════════════════════════════════════════════════

class NumericFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, min_channel_samples=5):
        self.min_channel_samples = min_channel_samples
    def fit(self, X_df, y=None):
        df = X_df.copy(); df['_y'] = y
        ch_counts = df.groupby('channel').size()
        valid = ch_counts[ch_counts >= self.min_channel_samples].index
        self.ch_score_ = df[df['channel'].isin(valid)].groupby('channel')['_y'].mean().to_dict()
        self.ch_freq_ = ch_counts.to_dict()
        return self
    def transform(self, X_df):
        df = X_df.copy()
        dur = pd.to_numeric(df['duration_seconds'], errors='coerce').fillna(600).clip(1, 86400)
        return np.column_stack([
            np.log1p(dur),
            df['channel'].map(self.ch_score_).fillna(0.5),
            np.log1p(df['channel'].map(self.ch_freq_).fillna(1)),
        ])


# ══════════════════════════════════════════════════════════════
# LightGBM 包装器 (用于 sklearn Pipeline)
# ══════════════════════════════════════════════════════════════

class LGBMClassifier(BaseEstimator):
    def __init__(self):
        self.model_ = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
            subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=RANDOM_SEED,
            verbose=-1, force_col_wise=True,
        )
    def fit(self, X, y):
        self.model_.fit(X, y)
        return self
    def predict(self, X):
        return self.model_.predict(X)
    def predict_proba(self, X):
        return self.model_.predict_proba(X)


# ══════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════

class StackedFeaturesV3(BaseEstimator, TransformerMixin):
    def __init__(self, use_image=True, pca_dim=128):
        self.use_image = use_image
        self.pca_dim = pca_dim

    def fit(self, X_df, y=None):
        texts = X_df['_texts'].tolist()
        self.tfidf_ = DualTfidfVectorizer(word_min_df=3, char_min_df=3, max_features=12000)
        self.tfidf_.fit(texts, y)
        self.numeric_ = NumericFeatures(min_channel_samples=5)
        self.numeric_.fit(X_df, y)
        num = self.numeric_.transform(X_df)
        self.num_scaler_ = StandardScaler().fit(num)
        if self.use_image and '_img_feats' in X_df.columns:
            img = np.vstack(X_df['_img_feats'].values)
            self.img_pca_ = PCA(n_components=self.pca_dim, random_state=RANDOM_SEED).fit(img)
            self.img_scaler_ = StandardScaler().fit(self.img_pca_.transform(img))
        return self

    def transform(self, X_df):
        parts = [self.tfidf_.transform(X_df['_texts'].tolist())]
        num = self.num_scaler_.transform(self.numeric_.transform(X_df))
        parts.append(num)
        if self.use_image and '_img_feats' in X_df.columns:
            img = np.vstack(X_df['_img_feats'].values)
            img_reduced = self.img_scaler_.transform(self.img_pca_.transform(img))
            parts.append(img_reduced)
        # 全部转 dense (LightGBM 需要)
        stacked = np.hstack([p.toarray() if sparse.issparse(p) else p for p in parts])
        return stacked


def train():
    print("=" * 60)
    print("V3 — TF-IDF + 数值 + 图像 → LightGBM")
    print("=" * 60)

    df = load_labeled_data()
    df['_texts'] = df.apply(build_text, axis=1).tolist()

    # 提取图像特征
    print(f"\n加载/提取图像特征 ({len(df):,} 行)...")
    t0 = time.time()
    img_feats = get_image_features(df['video_id'].tolist(), "train_v3")
    df['_img_feats'] = list(img_feats)
    print(f"  耗时: {time.time()-t0:.0f}s  shape={img_feats.shape}")

    y = (df["qc_text_result"] == "T").astype(int).to_numpy(dtype=np.int64)
    print(f"标签: T={y.sum():,} ({y.sum()/len(y)*100:.1f}%)  F={len(y)-y.sum():,}")

    pipe = Pipeline([
        ("features", StackedFeaturesV3(use_image=True, pca_dim=128)),
        ("clf", LGBMClassifier()),
    ])

    # ── 5-Fold CV ──
    print("\n--- 5-Fold CV (V3) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = {"auc":"roc_auc","ap":"average_precision","f1":"f1","precision":"precision","recall":"recall"}
    scores = cross_validate(pipe, df, y, cv=cv, scoring=scoring, n_jobs=1)

    for m in scoring:
        vals = scores[f"test_{m}"]
        print(f"  {m:12s}: {vals.mean():.4f} (±{vals.std():.4f})  [{', '.join(f'{v:.4f}' for v in vals)}]")

    # ── Hold-out ──
    print("\n--- Hold-out (20%) ---")
    X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    print(f"fit: {time.time()-t0:.1f}s")

    y_proba = pipe.predict_proba(X_te)[:, 1]
    y_pred = pipe.predict(X_te)
    print(classification_report(y_te, y_pred, target_names=["F","T"]))
    print(f"AUC: {roc_auc_score(y_te, y_proba):.4f}  AP: {average_precision_score(y_te, y_proba):.4f}")

    precs, recs, thrs = precision_recall_curve(y_te, y_proba)
    for r in [0.95, 0.90, 0.85, 0.80]:
        idx = np.argmin(np.abs(recs[:-1] - r))
        print(f"  recall={r:.2f}: thr={thrs[idx]:.3f}  prec={precs[idx]:.3f}")

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "film_tv_text_clf_v3.pkl", "wb") as f:
        pickle.dump(pipe, f)
    print(f"\n模型已保存: models/film_tv_text_clf_v3.pkl")


# ══════════════════════════════════════════════════════════════
# 打分
# ══════════════════════════════════════════════════════════════

def score_file(input_path, output_dir=None):
    model_path = MODEL_DIR / "film_tv_text_clf_v3.pkl"
    if not model_path.exists():
        print(f"[ERROR] 模型不存在，请先 --train"); sys.exit(1)
    with open(model_path, "rb") as f:
        pipe = pickle.load(f)
    ext = os.path.splitext(input_path)[1].lower()
    df_raw = pd.read_parquet(input_path) if ext==".parquet" else pd.read_csv(input_path)
    print(f"输入: {input_path} ({len(df_raw):,} 行)")
    df = df_raw.copy()
    df['_texts'] = df.apply(build_text, axis=1).tolist()
    print("提取图像特征...")
    img = get_image_features(df['video_id'].tolist(), f"score_{Path(input_path).stem}")
    df['_img_feats'] = list(img)
    t0 = time.time()
    df_raw["ml_score"] = pipe.predict_proba(df)[:, 1]
    print(f"打分: {time.time()-t0:.1f}s")
    out_dir = output_dir or os.path.dirname(input_path)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_scored_v3.parquet")
    df_raw.to_parquet(out_path, index=False)
    for th in [0.5, 0.3, 0.7]:
        n = (df_raw["ml_score"]>=th).sum()
        print(f"  score>={th}: {n:,} ({n/len(df_raw)*100:.1f}%)")
    print(f"输出: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--score", type=str, default=None)
    parser.add_argument("-o", "--output-dir", default=None)
    args = parser.parse_args()
    if args.train: train()
    elif args.score: score_file(args.score, args.output_dir)


if __name__=="__main__":
    main()

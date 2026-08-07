#!/usr/bin/env python3
"""
film_tv 文本分类器 V2 — TF-IDF + duration + channel + LogisticRegression

对比 V1（仅 TF-IDF），V2 增加:
  - log_duration: 取对数
  - ch_score: 频道 T 率（训练集聚合，min_df=5）
  - ch_log_freq: 频道出现次数的对数
"""

import sys, os, re, glob, argparse, time, pickle
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    average_precision_score, f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "experiments"))
sys.path.insert(0, str(_project_root / "02_脚本"))

from film_tv_text_classifier import load_labeled_data, build_text, DualTfidfVectorizer

MODEL_DIR = _project_root / "models"
RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════
# 数值特征
# ══════════════════════════════════════════════════════════════

class NumericFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, min_channel_samples=5):
        self.min_channel_samples = min_channel_samples

    def fit(self, X_df, y=None):
        df = X_df.copy()
        df['_y'] = y
        ch_counts = df.groupby('channel').size()
        valid = ch_counts[ch_counts >= self.min_channel_samples].index
        self.ch_score_ = df[df['channel'].isin(valid)].groupby('channel')['_y'].mean().to_dict()
        self.ch_freq_ = ch_counts.to_dict()
        return self

    def transform(self, X_df):
        df = X_df.copy()
        dur = pd.to_numeric(df['duration_seconds'], errors='coerce').fillna(600).clip(1, 86400)
        log_dur = np.log1p(dur).values.reshape(-1, 1)
        ch_score = df['channel'].map(self.ch_score_).fillna(0.5).values.reshape(-1, 1)
        ch_freq = df['channel'].map(self.ch_freq_).fillna(1).values
        ch_log_freq = np.log1p(ch_freq).reshape(-1, 1)
        return np.hstack([log_dur, ch_score, ch_log_freq])


# ══════════════════════════════════════════════════════════════

class StackedFeatures(BaseEstimator, TransformerMixin):
    """先对 _texts 做 TF-IDF，对全量做数值特征，再 hstack。"""

    def __init__(self):
        self.tfidf_ = DualTfidfVectorizer(word_min_df=3, char_min_df=3, max_features=12000)
        self.numeric_ = NumericFeatures(min_channel_samples=5)
        self.scaler_ = StandardScaler()

    def fit(self, X_df, y=None):
        texts = X_df['_texts'].tolist()
        self.tfidf_.fit(texts, y)
        self.numeric_.fit(X_df, y)
        num = self.numeric_.transform(X_df)
        self.scaler_.fit(num)
        return self

    def transform(self, X_df):
        texts = X_df['_texts'].tolist()
        X_tfidf = self.tfidf_.transform(texts)
        X_num = self.scaler_.transform(self.numeric_.transform(X_df))
        return sparse.hstack([X_tfidf, X_num], format="csr")


# ══════════════════════════════════════════════════════════════

def train():
    print("=" * 60)
    print("V2 — TF-IDF + duration + channel + LogisticRegression")
    print("=" * 60)

    df = load_labeled_data()
    df['_texts'] = df.apply(build_text, axis=1).tolist()
    y = (df["qc_text_result"] == "T").astype(int).to_numpy(dtype=np.int64)

    n_pos = y.sum()
    print(f"标签: T={n_pos:,} ({n_pos/len(y)*100:.1f}%)  F={len(y)-n_pos:,}")

    pipe = Pipeline([
        ("features", StackedFeatures()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)),
    ])

    # ── CV ──
    print("\n--- 5-Fold CV (V2) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = {"auc": "roc_auc", "ap": "average_precision", "f1": "f1", "precision": "precision", "recall": "recall"}
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
    print(classification_report(y_te, y_pred, target_names=["F", "T"]))
    print(f"AUC: {roc_auc_score(y_te, y_proba):.4f}  AP: {average_precision_score(y_te, y_proba):.4f}")

    # ── 数值系数 ──
    coef = pipe[1].coef_[0]
    for i, name in enumerate(['log_dur', 'ch_score', 'ch_log_freq']):
        print(f"  {name}: {coef[-3+i]:+.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "film_tv_text_clf_v2.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"\n模型已保存: {model_path}")


def score_file(input_path: str, output_dir: str | None = None):
    model_path = MODEL_DIR / "film_tv_text_clf_v2.pkl"
    if not model_path.exists():
        print(f"[ERROR] 模型不存在: {model_path}，请先 --train")
        sys.exit(1)
    with open(model_path, "rb") as f:
        pipe = pickle.load(f)
    print(f"加载模型: {model_path}")
    ext = os.path.splitext(input_path)[1].lower()
    df_raw = pd.read_parquet(input_path) if ext == ".parquet" else pd.read_csv(input_path)
    print(f"输入: {input_path}  ({len(df_raw):,} 行)")
    df = df_raw.copy()
    df['_texts'] = df.apply(build_text, axis=1).tolist()
    t0 = time.time()
    df_raw["ml_score"] = pipe.predict_proba(df)[:, 1]
    print(f"打分耗时: {time.time()-t0:.1f}s")
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{stem}_scored_v2.parquet")
    df_raw.to_parquet(out_path, index=False)
    for th in [0.5, 0.3, 0.7]:
        n = (df_raw["ml_score"] >= th).sum()
        print(f"  score >= {th}: {n:,} ({n/len(df_raw)*100:.1f}%)")
    print(f"输出: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--score", type=str, default=None, metavar="PATH")
    parser.add_argument("-o", "--output-dir", default=None)
    args = parser.parse_args()
    if args.train:
        train()
    elif args.score:
        score_file(args.score, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

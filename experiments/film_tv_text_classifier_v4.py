#!/usr/bin/env python3
"""
film_tv 文本分类器 V4 — TF-IDF + LinearSVM (from research benchmark)

论文证据: TF-IDF + LinearSVM 在 5K-10K 短文本上比 LR 稳定高 2-3pp F1
(SVM→F1 0.906 vs LR→0.88, LightGBM→0.685 on imbalanced small data)

用法:
  python3 experiments/film_tv_text_classifier_v4.py --train
  python3 experiments/film_tv_text_classifier_v4.py --score PATH
"""

import sys, os, re, glob, argparse, time, pickle
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    average_precision_score, f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "experiments"))
sys.path.insert(0, str(_project_root / "02_脚本"))

from film_tv_text_classifier import load_labeled_data, build_text, DualTfidfVectorizer

MODEL_DIR = _project_root / "models"
RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════

def train():
    print("=" * 60)
    print("V4 — TF-IDF + LinearSVM (research-backed)")
    print("=" * 60)

    df = load_labeled_data()
    df['_texts'] = df.apply(build_text, axis=1).tolist()
    y = (df["qc_text_result"] == "T").astype(int).to_numpy(dtype=np.int64)

    n_pos = y.sum()
    print(f"标签: T={n_pos:,} ({n_pos/len(y)*100:.1f}%)  F={len(y)-n_pos:,}")

    # LinearSVC 不支持 probability，用 SVC(kernel='linear') + CalibratedClassifierCV
    # 或者直接用 SVC(kernel='linear', probability=True) 内建 Platt scaling
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import MaxAbsScaler

    svm = SVC(kernel='linear', class_weight='balanced',
              C=1.0, max_iter=5000, random_state=RANDOM_SEED)
    clf = CalibratedClassifierCV(svm, cv=3, method='sigmoid')

    pipe = Pipeline([
        ("tfidf", DualTfidfVectorizer(word_min_df=3, char_min_df=3, max_features=12000)),
        ("scaler", MaxAbsScaler()),
        ("clf", clf),
    ])

    # ── 5-Fold CV ──
    print("\n--- 5-Fold CV (V4: SVM) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = {"auc":"roc_auc","ap":"average_precision","f1":"f1","precision":"precision","recall":"recall"}
    scores = cross_validate(pipe, df['_texts'].tolist(), y, cv=cv, scoring=scoring, n_jobs=1)

    for m in scoring:
        vals = scores[f"test_{m}"]
        print(f"  {m:12s}: {vals.mean():.4f} (±{vals.std():.4f})  [{', '.join(f'{v:.4f}' for v in vals)}]")

    # ── Hold-out ──
    print("\n--- Hold-out (20%) ---")
    texts = df['_texts'].tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(texts, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
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

    # ── 特征重要性 (SVM coefficients) ──
    coef = pipe[1].coef_[0]
    feat_names = pipe[0].get_feature_names_out()
    print("\n--- Top 20 正向特征 ---")
    for i in np.argsort(coef)[-20:][::-1]:
        print(f"  {feat_names[i]:22s}  {coef[i]:+.4f}")
    print("\n--- Top 20 负向特征 ---")
    for i in np.argsort(coef)[:20][::-1]:
        print(f"  {feat_names[i]:22s}  {coef[i]:+.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "film_tv_text_clf_svm.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"\n模型已保存: {model_path}")
    print(f"\n对比 V1(LR): F1=0.856, V4(SVM): F1={scores['test_f1'].mean():.4f}")


def score_file(input_path, output_dir=None):
    model_path = MODEL_DIR / "film_tv_text_clf_svm.pkl"
    if not model_path.exists():
        print(f"[ERROR] 模型不存在"); sys.exit(1)
    with open(model_path, "rb") as f:
        pipe = pickle.load(f)
    ext = os.path.splitext(input_path)[1].lower()
    df = pd.read_parquet(input_path) if ext==".parquet" else pd.read_csv(input_path)
    print(f"输入: {input_path} ({len(df):,} 行)")
    texts = df.apply(build_text, axis=1).tolist()
    t0 = time.time()
    df["ml_score"] = pipe.predict_proba(texts)[:, 1]
    print(f"打分: {time.time()-t0:.1f}s")
    out_dir = output_dir or os.path.dirname(input_path)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_scored_v4.parquet")
    df.to_parquet(out_path, index=False)
    for th in [0.5, 0.3, 0.7]:
        n = (df["ml_score"]>=th).sum()
        print(f"  score>={th}: {n:,} ({n/len(df)*100:.1f}%)")
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

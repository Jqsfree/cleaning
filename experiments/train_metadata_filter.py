# -*- coding: utf-8 -*-
"""
Stage -1: 元数据文本过滤器（训练脚本）

特征构造在 02_脚本/core/metadata_filter.py，推理与之完全一致。

用法:
    PYTHONPATH=02_脚本 python experiments/train_metadata_filter.py \\
      --csv_dir /path/to/qc_csvs --out_dir models/exo_agriculture_metadata_filter
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "02_脚本"))
from core.metadata_filter import build_numeric_features, build_text_field  # noqa: E402


def load_data(csv_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"没有在 {csv_dir} 下找到csv")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset="video_id", keep="first")
    df = df[df["qc_result"].isin(["T", "F"])].copy()
    df["y"] = (df["qc_result"] == "T").astype(int)
    for col in ["title", "keyword", "channel", "description"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce").fillna(0)
    df["view_count"] = pd.to_numeric(df["view_count"], errors="coerce").fillna(0)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=".")
    ap.add_argument("--out_dir", default="./model_out")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_data(args.csv_dir)
    print(f"[数据] 总样本 {len(df)}，正样本(T) {df['y'].sum()}，负样本(F) {(1-df['y']).sum()}")

    text = build_text_field(df)
    num = build_numeric_features(df)
    y = df["y"].values

    X_train_idx, X_test_idx = train_test_split(
        np.arange(len(df)), test_size=args.test_size,
        stratify=y, random_state=args.random_state,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.9,
        sublinear_tf=True, stop_words="english",
    )
    X_text_train = vectorizer.fit_transform(text.iloc[X_train_idx])
    X_text_test = vectorizer.transform(text.iloc[X_test_idx])

    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(num[X_train_idx])
    X_num_test = scaler.transform(num[X_test_idx])

    X_train = hstack([X_text_train, csr_matrix(X_num_train)])
    X_test = hstack([X_text_test, csr_matrix(X_num_test)])
    y_train, y_test = y[X_train_idx], y[X_test_idx]

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(X_train, y_train)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    X_full_text = vectorizer.transform(text)
    X_full = hstack([X_full_text, csr_matrix(scaler.transform(num))])
    cv_proba = cross_val_predict(
        LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        X_full, y, cv=skf, method="predict_proba",
    )[:, 1]

    print("\n=== 5折交叉验证 整体表现 ===")
    print(f"ROC-AUC: {roc_auc_score(y, cv_proba):.4f}")
    print(f"PR-AUC : {average_precision_score(y, cv_proba):.4f}")

    n_neg = int((y == 0).sum())
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred = (cv_proba >= thr).astype(int)
        cm = confusion_matrix(y, pred)
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        neg_filtered = tn / max(n_neg, 1)
        print(
            f"阈值={thr:.1f}  precision={precision:.3f}  recall={recall:.3f}  "
            f"正样本误杀率={fn / max(tp + fn, 1):.3f}  负样本滤除占比={neg_filtered:.3f}",
        )

    print("\n=== held-out test set 分类报告 (阈值0.5) ===")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["F(负)", "T(正)"]))

    joblib.dump(clf, os.path.join(args.out_dir, "clf.joblib"))
    joblib.dump(vectorizer, os.path.join(args.out_dir, "vectorizer.joblib"))
    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.joblib"))
    print(f"\n模型已保存到 {args.out_dir}")

    feat_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0][:len(feat_names)]
    top_pos = feat_names[np.argsort(coefs)[-25:][::-1]]
    top_neg = feat_names[np.argsort(coefs)[:25]]
    print("\n=== 模型学到的最强正向词/短语(仅供 review，勿写硬规则) ===")
    print(", ".join(top_pos))
    print("\n=== 模型学到的最强负向词/短语(仅供 review，勿写硬规则) ===")
    print(", ".join(top_neg))

    kw_stats = df.groupby("keyword")["y"].agg(total="count", pass_cnt="sum")
    kw_stats["pass_rate"] = kw_stats["pass_cnt"] / kw_stats["total"]
    blocklist = kw_stats[(kw_stats["total"] >= 5) & (kw_stats["pass_rate"] <= 0.2)]
    blocklist = blocklist.sort_values("total", ascending=False)
    blocklist.to_csv(os.path.join(args.out_dir, "blocklist_keywords.csv"))
    print(f"\n低产keyword黑名单已导出，共 {len(blocklist)} 个")


if __name__ == "__main__":
    main()

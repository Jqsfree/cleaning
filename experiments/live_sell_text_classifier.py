#!/usr/bin/env python3
"""
live_sell 文本分类器 — Dual TF-IDF + LogisticRegression

训练锚点：human_yb01 人工正负样本（可选追加 0804 boundary_sell_like）。
阈值：分层 OOF 扫分，输出严/中/松三档校准 JSON（宁少勿错）。

用法:
  python experiments/live_sell_text_classifier.py --train
  python experiments/live_sell_text_classifier.py --train --with-boundary
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

_PROJECT = Path(__file__).resolve().parent.parent
MODEL_DIR = _PROJECT / "models"
YB01_TRAIN = _PROJECT / "data/runs/live_sell/human_yb01/03_qc/train_export.csv"
BOUNDARY_CSV = _PROJECT / "data/runs/live_sell/human_0804/03_qc/fail_type_taxonomy.csv"
CALIB_PATH = MODEL_DIR / "live_sell_text_clf_yb01_calibration.json"
MODEL_PATH = MODEL_DIR / "live_sell_text_clf_yb01.pkl"
RANDOM_SEED = 42


class DualTfidfVectorizer(BaseEstimator, TransformerMixin):
    """word ngram(1-2) + char_wb ngram(2-4)。与 film_tv 同构，便于 unpickle。"""

    def __init__(self, word_min_df=2, char_min_df=2, max_features=12000):
        self.word_min_df = word_min_df
        self.char_min_df = char_min_df
        self.max_features = max_features

    def fit(self, X, y=None):
        self.word_vec_ = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=self.word_min_df, max_df=0.85,
            sublinear_tf=True,
        )
        self.char_vec_ = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=self.max_features,
            min_df=self.char_min_df, max_df=0.85,
            sublinear_tf=True,
        )
        self.word_vec_.fit(X)
        self.char_vec_.fit(X)
        return self

    def transform(self, X):
        return sparse.hstack(
            [self.word_vec_.transform(X), self.char_vec_.transform(X)],
            format="csr",
        )

    def get_feature_names_out(self, input_features=None):
        w = list(self.word_vec_.get_feature_names_out())
        c = list(self.char_vec_.get_feature_names_out())
        return np.array(w + c)


def build_text(row: pd.Series) -> str:
    t = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
    k = str(row.get("keyword", "")) if pd.notna(row.get("keyword")) else ""
    if not k.strip():
        ch = row.get("channel", "")
        k = str(ch) if pd.notna(ch) else ""
    k = re.sub(r"(^|\s)-[a-zA-Z0-9*?]+", "", k).strip()
    combined = re.sub(r"\s+", " ", f"{t} {k}".strip())
    return combined


def load_train_frame(*, with_boundary: bool) -> pd.DataFrame:
    if not YB01_TRAIN.is_file():
        raise FileNotFoundError(
            f"缺少 {YB01_TRAIN}；请先 ingest_human_qc 写入 train_export"
        )
    df = pd.read_csv(YB01_TRAIN)
    df = df[df["human_label"].isin(["pass", "fail"])].copy()
    df["y"] = (df["human_label"] == "pass").astype(int)
    df["_src"] = "yb01"

    if with_boundary and BOUNDARY_CSV.is_file():
        tax = pd.read_csv(BOUNDARY_CSV)
        b = tax[tax["fail_type"] == "boundary_sell_like"].copy()
        if len(b):
            b["human_label"] = "fail"
            b["y"] = 0
            b["_src"] = "0804_boundary"
            # 去重：已在 yb01 的 video_id 不重复加
            seen = set(df["video_id"].astype(str))
            b = b[~b["video_id"].astype(str).isin(seen)]
            keep_cols = [c for c in df.columns if c in b.columns or c in ("y", "_src", "human_label", "video_id", "title", "channel", "keyword")]
            # align columns
            for c in df.columns:
                if c not in b.columns:
                    b[c] = np.nan
            df = pd.concat([df, b[df.columns]], ignore_index=True)
            print(f"  + boundary_sell_like: {len(b)} 条（0804）")

    df = df.drop_duplicates("video_id", keep="first")
    print(
        f"训练行: {len(df)}  "
        f"pass={int((df.y==1).sum())} fail={int((df.y==0).sum())}"
    )
    return df


def _neg_precision_at_drop(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    """
    drop = score < thr（高置信负例）。
    drop_precision = 实际负例 / drop 数；pos_hurt = 被 drop 的正例数。
    """
    pred_drop = scores < thr
    n_drop = int(pred_drop.sum())
    if n_drop == 0:
        return {
            "drop_threshold": thr,
            "n_drop": 0,
            "drop_precision": 1.0,
            "pos_hurt": 0,
            "neg_caught": 0,
            "neg_recall": 0.0,
        }
    y_drop = y_true[pred_drop]
    neg_caught = int((y_drop == 0).sum())
    pos_hurt = int((y_drop == 1).sum())
    n_neg = int((y_true == 0).sum())
    return {
        "drop_threshold": float(thr),
        "n_drop": n_drop,
        "drop_precision": neg_caught / n_drop,
        "pos_hurt": pos_hurt,
        "neg_caught": neg_caught,
        "neg_recall": neg_caught / max(n_neg, 1),
    }


def calibrate_oof(pipe_template: Pipeline, texts: list[str], y: np.ndarray) -> dict:
    """分层 OOF 概率 → 扫 drop 阈值 → 严/中/松三档。"""
    n_splits = 5
    # 27 neg → 每折约 5；若更少则降折
    n_neg = int((y == 0).sum())
    if n_neg < n_splits:
        n_splits = max(2, n_neg)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(y), dtype=float)
    for tr, te in cv.split(texts, y):
        pipe = Pipeline([
            ("tfidf", DualTfidfVectorizer(
                word_min_df=pipe_template.named_steps["tfidf"].word_min_df,
                char_min_df=pipe_template.named_steps["tfidf"].char_min_df,
                max_features=pipe_template.named_steps["tfidf"].max_features,
            )),
            ("clf", LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000,
                random_state=RANDOM_SEED,
            )),
        ])
        X_tr = [texts[i] for i in tr]
        X_te = [texts[i] for i in te]
        pipe.fit(X_tr, y[tr])
        oof[te] = pipe.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)
    print(f"\nOOF ROC-AUC={auc:.4f}  AP={ap:.4f}  folds={n_splits}")

    # 扫阈值：score < t → drop
    candidates = []
    for thr in np.round(np.arange(0.05, 0.55, 0.01), 2):
        m = _neg_precision_at_drop(y, oof, float(thr))
        candidates.append(m)

    def pick(min_prec: float, max_pos_hurt: int, prefer_recall: bool) -> dict | None:
        ok = [
            c for c in candidates
            if c["n_drop"] > 0
            and c["drop_precision"] + 1e-9 >= min_prec
            and c["pos_hurt"] <= max_pos_hurt
        ]
        if not ok:
            return None
        if prefer_recall:
            return max(ok, key=lambda c: (c["neg_recall"], c["drop_precision"], -c["drop_threshold"]))
        # 更严：优先 pos_hurt=0，再 precision，再略高阈值（更少 drop）
        return max(
            ok,
            key=lambda c: (-c["pos_hurt"], c["drop_precision"], c["drop_threshold"]),
        )

    strict = pick(0.90, 0, prefer_recall=False) or pick(0.80, 0, prefer_recall=False)
    mid = pick(0.80, 1, prefer_recall=True) or pick(0.70, 1, prefer_recall=True)
    loose = pick(0.70, 2, prefer_recall=True) or pick(0.60, 3, prefer_recall=True)

    # keep_threshold：高置信正例（与 apply_small_model 对称）
    # 用 OOF：score >= k 时正类 precision 高
    keep_cands = []
    for k in np.round(np.arange(0.55, 0.96, 0.01), 2):
        mask = oof >= k
        n = int(mask.sum())
        if n == 0:
            continue
        prec = float((y[mask] == 1).sum() / n)
        keep_cands.append({"keep_threshold": float(k), "n": n, "precision": prec})
    keep_thr = 0.85
    for row in sorted(keep_cands, key=lambda r: r["keep_threshold"]):
        if row["precision"] >= 0.95 and row["n"] >= 10:
            keep_thr = row["keep_threshold"]
            break

    profiles = {
        "strict": {
            **(strict or {"drop_threshold": 0.10, "n_drop": 0, "drop_precision": 0, "pos_hurt": 0, "neg_caught": 0, "neg_recall": 0}),
            "keep_threshold": keep_thr,
            "note": "宁少勿错；全量默认用此档",
        },
        "mid": {
            **(mid or strict or {"drop_threshold": 0.15}),
            "keep_threshold": keep_thr,
            "note": "小流量灰度",
        },
        "loose": {
            **(loose or mid or strict or {"drop_threshold": 0.20}),
            "keep_threshold": max(0.75, keep_thr - 0.05),
            "note": "仅实验；勿直接全量",
        },
    }
    # 保证 mid/loose 有 drop_threshold
    for name in ("mid", "loose"):
        if "drop_threshold" not in profiles[name]:
            profiles[name]["drop_threshold"] = profiles["strict"]["drop_threshold"]

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "oof_auc": float(auc),
        "oof_ap": float(ap),
        "n_splits": n_splits,
        "profiles": profiles,
        "curve_head": candidates[::5],
        "notes": [
            "抽样置信度 90% ≠ 模型阈值",
            "ml_action 不是交付 KPI；交付只认人工 pass_rate",
            "overturn 抽检 drop 集 n>=100，贴临界补 150-200",
        ],
    }
    return report


def train_and_calibrate(*, with_boundary: bool) -> None:
    print("=" * 60)
    print("live_sell 文本分类器 — Dual TF-IDF + LR (yb01 锚点)")
    print("=" * 60)
    df = load_train_frame(with_boundary=with_boundary)
    texts = df.apply(build_text, axis=1).tolist()
    y = df["y"].to_numpy(dtype=np.int64)

    pipe = Pipeline([
        ("tfidf", DualTfidfVectorizer(word_min_df=2, char_min_df=2, max_features=12000)),
        ("clf", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000,
            random_state=RANDOM_SEED,
        )),
    ])

    print("\n--- Stratified CV ---")
    n_neg = int((y == 0).sum())
    n_splits = 5 if n_neg >= 5 else max(2, n_neg)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "auc": "roc_auc",
        "avg_precision": "average_precision",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }
    scores = cross_validate(pipe, texts, y, cv=cv, scoring=scoring, n_jobs=1)
    for metric in scoring:
        vals = scores[f"test_{metric}"]
        print(f"  {metric:18s}: {vals.mean():.4f} (±{vals.std():.4f})")

    print("\n--- OOF 阈值校准 ---")
    calib = calibrate_oof(pipe, texts, y)
    for name, prof in calib["profiles"].items():
        print(
            f"  [{name}] drop<{prof.get('drop_threshold')}  "
            f"keep>={prof.get('keep_threshold')}  "
            f"drop_prec={prof.get('drop_precision', 0):.2f}  "
            f"pos_hurt={prof.get('pos_hurt')}  "
            f"neg_recall={prof.get('neg_recall', 0):.2f}"
        )

    print("\n--- Fit full ---")
    t0 = time.time()
    pipe.fit(texts, y)
    print(f"fit {time.time()-t0:.1f}s")
    y_pred = pipe.predict(texts)
    print(classification_report(y, y_pred, target_names=["F", "T"]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # 保证 pickle 以 live_sell_text_classifier.DualTfidfVectorizer 加载（非 __main__）
    import sys
    DualTfidfVectorizer.__module__ = "live_sell_text_classifier"
    sys.modules["live_sell_text_classifier"] = sys.modules[__name__]
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    calib["model_path"] = str(MODEL_PATH)
    calib["with_boundary"] = with_boundary
    with open(CALIB_PATH, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    print(f"模型: {MODEL_PATH}")
    print(f"校准: {CALIB_PATH}")


def main() -> None:
    p = argparse.ArgumentParser(description="live_sell yb01 文本小模型训练+校准")
    p.add_argument("--train", action="store_true")
    p.add_argument(
        "--with-boundary", action="store_true",
        help="追加 0804 fail_type=boundary_sell_like 作负例",
    )
    args = p.parse_args()
    if args.train:
        train_and_calibrate(with_boundary=args.with_boundary)
    else:
        p.print_help()


if __name__ == "__main__":
    main()

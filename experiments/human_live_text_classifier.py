#!/usr/bin/env python3
"""
experiments/human_live_text_classifier.py — 真人直播「非真人直播检测器」训练+校准

训练数据：machine_0805 文本 QC 快照（human_live 口径，367 样本）。
标签语义（与 apply_small_model 对齐）：
    y = 1  # 非F（T+U：真人直播 + 边界），保留
    y = 0  # F（确定非真人直播），高置信时可 drop
score = P(非F)；apply_small_model 的 score < drop_threshold → drop 命中高置信 F。

用法:
  python experiments/human_live_text_classifier.py --train
"""

import json
import pickle
import re
import sys
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
QC_SNAP = (
    _PROJECT
    / "data/runs/live_sell/machine_0805/03_qc"
    / "纯直播机采_0805_records_sample_0805_textqc_20260805_053511.csv"
)
MODEL_PATH = MODEL_DIR / "human_live_text_clf_f.pkl"
CALIB_PATH = MODEL_DIR / "human_live_text_clf_f_calibration.json"
RANDOM_SEED = 42


class DualTfidfVectorizer(BaseEstimator, TransformerMixin):
    """word ngram(1-2) + char_wb ngram(2-4)。与 live_sell 同构，便于 unpickle。"""

    def __init__(self, word_min_df=2, char_min_df=2, max_features=12000):
        self.word_min_df = word_min_df
        self.char_min_df = char_min_df
        self.max_features = max_features
        self._word = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=word_min_df,
            max_features=max_features,
            lowercase=True,
            strip_accents="unicode",
        )
        self._char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=char_min_df,
            max_features=max_features,
            lowercase=True,
        )

    def fit(self, X, y=None):
        self._word.fit(X)
        self._char.fit(X)
        return self

    def transform(self, X):
        return sparse.hstack(
            [self._word.transform(X), self._char.transform(X)]
        ).tocsr()

    def get_feature_names_out(self, input_features=None):
        return np.concatenate(
            [self._word.get_feature_names_out(), self._char.get_feature_names_out()]
        )


def build_text(row: pd.Series) -> str:
    t = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
    k = str(row.get("keyword", "")) if pd.notna(row.get("keyword")) else ""
    if not k.strip():
        ch = row.get("channel", "")
        k = str(ch) if pd.notna(ch) else ""
    k = re.sub(r"(^|\s)-[a-zA-Z0-9*?]+", "", k).strip()
    return re.sub(r"\s+", " ", f"{t} {k}".strip())


def load_train_frame() -> pd.DataFrame:
    """读 machine_0805 QC 快照；T/U → 非F(1)，F → 0，ERROR 排除。"""
    if not QC_SNAP.is_file():
        raise FileNotFoundError(f"缺少 QC 快照: {QC_SNAP}")
    df = pd.read_csv(QC_SNAP, encoding="utf-8-sig")
    df = df[df["qc_text_result"].isin(["T", "F", "U"])].copy()
    df["y"] = (df["qc_text_result"] != "F").astype(int)
    df["_src"] = "machine_0805"
    df = df.drop_duplicates("video_id", keep="first")
    print(
        f"训练行: {len(df)}  "
        f"非F(保留)={int((df.y == 1).sum())}  "
        f"F(可drop)={int((df.y == 0).sum())}"
    )
    return df


def _neg_precision_at_drop(y_true, scores, thr):
    """drop = score < thr（高置信负例=F）。drop_precision = 实际 F / drop 数。"""
    pred_drop = scores < thr
    n_drop = int(pred_drop.sum())
    if n_drop == 0:
        return {
            "drop_threshold": float(thr), "n_drop": 0, "drop_precision": 1.0,
            "pos_hurt": 0, "neg_caught": 0, "neg_recall": 0.0,
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


def calibrate_oof(pipe_template, texts, y):
    """分层 OOF 概率 → 扫 drop 阈值 → 严/中/松三档。"""
    n_splits = 5
    n_pos = int((y == 1).sum())
    if n_pos < n_splits:
        n_splits = max(2, n_pos)
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
        pipe.fit([texts[i] for i in tr], y[tr])
        oof[te] = pipe.predict_proba([texts[i] for i in te])[:, 1]

    auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)
    print(f"\nOOF ROC-AUC={auc:.4f}  AP={ap:.4f}  folds={n_splits}")

    candidates = []
    for thr in np.round(np.arange(0.05, 0.55, 0.01), 2):
        candidates.append(_neg_precision_at_drop(y, oof, float(thr)))

    def pick(min_prec, max_pos_hurt, prefer_recall):
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
        return max(
            ok,
            key=lambda c: (-c["pos_hurt"], c["drop_precision"], c["drop_threshold"]),
        )

    strict = pick(0.90, 0, prefer_recall=False) or pick(0.80, 0, prefer_recall=False)
    mid = pick(0.80, 1, prefer_recall=True) or pick(0.70, 1, prefer_recall=True)
    loose = pick(0.70, 2, prefer_recall=True) or pick(0.60, 3, prefer_recall=True)

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
    for name in ("mid", "loose"):
        if "drop_threshold" not in profiles[name]:
            profiles[name]["drop_threshold"] = profiles["strict"]["drop_threshold"]

    return {
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
            "ml_action 不是交付 KPI；交付只认人工 pass_rate",
            "overturn 抽检 drop 集 n>=100，贴临界补 150-200",
        ],
    }


def train_and_calibrate() -> None:
    print("=" * 60)
    print("human_live 文本分类器 — Dual TF-IDF + LR（F 检测器）")
    print("=" * 60)
    df = load_train_frame()
    texts = df.apply(build_text, axis=1).tolist()
    y = df["y"].to_numpy(dtype=np.int64)

    pipe = Pipeline([
        ("tfidf", DualTfidfVectorizer(word_min_df=2, char_min_df=2, max_features=12000)),
        ("clf", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000,
            random_state=RANDOM_SEED,
        )),
    ])

    n_pos = int((y == 1).sum())
    n_splits = 5 if n_pos >= 5 else max(2, n_pos)
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
    print(f"fit {time.time() - t0:.1f}s")
    print(classification_report(y, pipe.predict(texts), target_names=["F(可drop)", "非F(保留)"]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # 保证 pickle 以 human_live_text_classifier.DualTfidfVectorizer 加载（非 __main__）
    DualTfidfVectorizer.__module__ = "human_live_text_classifier"
    sys.modules["human_live_text_classifier"] = sys.modules[__name__]
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    calib["model_path"] = str(MODEL_PATH)
    with open(CALIB_PATH, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    print(f"模型: {MODEL_PATH}")
    print(f"校准: {CALIB_PATH}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="human_live F 检测器训练+校准")
    p.add_argument("--train", action="store_true")
    args = p.parse_args()
    if args.train:
        train_and_calibrate()
    else:
        p.print_help()


if __name__ == "__main__":
    main()

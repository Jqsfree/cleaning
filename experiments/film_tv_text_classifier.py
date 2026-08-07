#!/usr/bin/env python3
"""
film_tv 文本分类器 — Dual TF-IDF (word+char) + LogisticRegression

用法:
  python3 experiments/film_tv_text_classifier.py --train       # 训练 + CV 评估
  python3 experiments/film_tv_text_classifier.py --score PATH   # 对 CSV/Parquet 打分
"""

import sys, os, re, glob, argparse, time, pickle
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    average_precision_score, f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "02_脚本"))

# ── 配置 ──
DATA_GLOB = str(_project_root / "data/runs/film_tv/**/*textqc*.csv")
MANUAL_LABEL_FILES = [
    str(Path.home() / "Documents/人工collect_0723_6167cd3a_qc_result.csv"),
    str(Path.home() / "tmp/完整现代影视剧-05_sampled_c90_e005.csv_18853dd3_qc_result.csv"),
    str(Path.home() / "tmp/完整现代影视剧-03_records_test_4f626b1d_qc_result.csv"),
    str(Path.home() / "tmp/完整现代影视剧-01-04_去重合并_sampled_c90_e005.csv_cec07002_qc_result.csv"),
    str(Path.home() / "tmp/MQALL_qc_da59ca68_qc_result.csv"),
    str(Path.home() / "tmp/MQALL_qc_da59ca68_qc_result (1).csv"),
]
MODEL_DIR = _project_root / "models"
RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════
# 自定义 Transformer：双 TfidfVectorizer 堆叠
# ══════════════════════════════════════════════════════════════

class DualTfidfVectorizer(BaseEstimator, TransformerMixin):
    """word ngram(1-2) + char_wb ngram(2-4)，各自 fit_transform 后 sparse.hstack。"""

    def __init__(self, word_min_df=3, char_min_df=3, max_features=12000):
        self.word_min_df = word_min_df
        self.char_min_df = char_min_df
        self.max_features = max_features

    def fit(self, X, y=None):
        self.word_vec_ = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=self.word_min_df, max_df=0.7,
            sublinear_tf=True,
        )
        self.char_vec_ = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=self.max_features,
            min_df=self.char_min_df, max_df=0.7,
            sublinear_tf=True,
        )
        self.word_vec_.fit(X)
        self.char_vec_.fit(X)
        return self

    def transform(self, X):
        X_word = self.word_vec_.transform(X)
        X_char = self.char_vec_.transform(X)
        return sparse.hstack([X_word, X_char], format="csr")

    def get_feature_names_out(self, input_features=None):
        w = list(self.word_vec_.get_feature_names_out())
        c = list(self.char_vec_.get_feature_names_out())
        return np.array(w + c)


# ══════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════

def load_labeled_data() -> pd.DataFrame:
    files = sorted(glob.glob(DATA_GLOB, recursive=True))

    for mp in MANUAL_LABEL_FILES:
        manual_path = Path(mp)
        if manual_path.exists():
            files.append(str(manual_path))
            print(f"  → 加入人工标注: {manual_path.name}")
        else:
            print(f"  [WARN] 人工标注文件不存在: {mp}")

    if not files:
        raise FileNotFoundError("未找到任何标注文件")

    print(f"加载 {len(files)} 个标注文件...")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        label_col = "qc_text_result" if "qc_text_result" in df.columns else "qc_result"
        if label_col not in df.columns:
            continue
        labeled = df[df[label_col].isin(["T", "F"])].copy()
        if label_col == "qc_result":
            labeled["qc_text_result"] = labeled["qc_result"]
        labeled["_source"] = Path(f).name[:50]
        frames.append(labeled)

    df_all = pd.concat(frames, ignore_index=True)
    print(f"  标注行: {len(df_all):,}  (T={len(df_all[df_all.qc_text_result=='T']):,}, "
          f"F={len(df_all[df_all.qc_text_result=='F']):,})")

    n_before = len(df_all)
    df_all = df_all.sort_values("_source").drop_duplicates("video_id", keep="last")
    print(f"  去重后: {len(df_all):,}  (移除 {n_before - len(df_all):,} 条)")
    return df_all


def build_text(row: pd.Series) -> str:
    t = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
    k = str(row.get("keyword", "")) if pd.notna(row.get("keyword")) else ""
    k = re.sub(r"(^|\s)-[a-zA-Z0-9*?]+", "", k).strip()
    combined = f"{t} {k}".strip()
    combined = re.sub(r"\s+", " ", combined)
    # 注入关键词特征：年份标记、完整剧集标记
    extra = []
    if re.search(r'\(\s*(19|20)\d{2}\s*\)', t):
        extra.append("FILM_YEAR_TOKEN")
    if re.search(r'\b\d{4}\b', t):
        extra.append("HAS_YEAR_TOKEN")
    if extra:
        combined = combined + " " + " ".join(extra)
    return combined


# ══════════════════════════════════════════════════════════════
# 训练 & 评估
# ══════════════════════════════════════════════════════════════

def train_and_evaluate():
    print("=" * 60)
    print("film_tv 文本分类器 — Dual TF-IDF + LogisticRegression")
    print("=" * 60)

    df = load_labeled_data()
    texts = df.apply(build_text, axis=1).tolist()
    y = (df["qc_text_result"] == "T").astype(int).to_numpy(dtype=np.int64)

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"\n标签分布: T={n_pos:,} ({n_pos/len(y)*100:.1f}%)  "
          f"F={n_neg:,} ({n_neg/len(y)*100:.1f}%)")

    # ── 构建 pipeline ──
    tfidf = DualTfidfVectorizer(word_min_df=3, char_min_df=3, max_features=12000)
    lr = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000,
        random_state=RANDOM_SEED,
    )
    pipe = Pipeline([("tfidf", tfidf), ("clf", lr)])

    # ── 5-fold stratified CV ──
    print("\n--- 5-Fold Stratified CV ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

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
        print(f"  {metric:18s}: {vals.mean():.4f} (±{vals.std():.4f})  "
              f"[{', '.join(f'{v:.4f}' for v in vals)}]")

    # ── Hold-out 评估 ──
    print("\n--- Hold-out 评估 (20%) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    t0 = time.time()
    pipe.fit(X_train, y_train)
    print(f"fit 耗时: {time.time()-t0:.1f}s")

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print(f"\n词表大小: word={pipe[0].word_vec_.vocabulary_.__len__():,}  "
          f"char={pipe[0].char_vec_.vocabulary_.__len__():,}  "
          f"total={pipe[0].get_feature_names_out().__len__():,}")

    print(f"\nClassification Report (test set, {len(y_test):,} samples):")
    print(classification_report(y_test, y_pred, target_names=["F(非影视)", "T(影视剧)"]))

    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.4f}  Average Precision: {ap:.4f}")

    # ── PR 曲线 阈值分析 ──
    print("\n--- PR 曲线 阈值分析 ---")
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    print(f"  {'Recall':>8s}  {'Threshold':>10s}  {'Precision':>10s}  {'F1':>8s}")
    print(f"  {'-'*42}")
    for target_recall in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
        idx = np.argmin(np.abs(recalls[:-1] - target_recall))
        t = thresholds[idx]
        p = precisions[idx]
        y_pred_t = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t)
        print(f"  {target_recall:>8.2f}  {t:>10.3f}  {p:>10.3f}  {f1:>8.4f}")

    # ── 特征重要性 ──
    print("\n--- Top 20 正向特征 (→T/影视剧) ---")
    feature_names = pipe[0].get_feature_names_out()
    coef = pipe[1].coef_[0]
    top_idx = np.argsort(coef)[-20:][::-1]
    for i in top_idx:
        print(f"  {feature_names[i]:22s}  {coef[i]:+.4f}")

    print(f"\n--- Top 20 负向特征 (→F/非影视) ---")
    bottom_idx = np.argsort(coef)[:20]
    for i in bottom_idx[::-1]:
        print(f"  {feature_names[i]:22s}  {coef[i]:+.4f}")

    # ── 保存 ──
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "film_tv_text_clf_tfidf.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"\n模型已保存: {model_path}")

    return pipe


# ══════════════════════════════════════════════════════════════
# 打分
# ══════════════════════════════════════════════════════════════

def score_file(input_path: str, output_dir: str | None = None):
    model_path = MODEL_DIR / "film_tv_text_clf_tfidf.pkl"
    if not model_path.exists():
        print(f"[ERROR] 模型不存在: {model_path}，请先 --train")
        sys.exit(1)

    with open(model_path, "rb") as f:
        pipe = pickle.load(f)
    print(f"加载模型: {model_path}")

    ext = os.path.splitext(input_path)[1].lower()
    df = pd.read_parquet(input_path) if ext == ".parquet" else pd.read_csv(input_path)
    print(f"输入: {input_path}  ({len(df):,} 行)")

    texts = df.apply(build_text, axis=1).tolist()
    t0 = time.time()
    df["ml_score"] = pipe.predict_proba(texts)[:, 1]
    print(f"打分耗时: {time.time()-t0:.1f}s")

    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{stem}_scored.parquet")
    df.to_parquet(out_path, index=False)

    for th in [0.5, 0.3, 0.7]:
        n = (df["ml_score"] >= th).sum()
        print(f"  score >= {th}: {n:,} ({n/len(df)*100:.1f}%)")
    print(f"输出: {out_path}")


# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="film_tv 文本分类器")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--score", type=str, default=None, metavar="PATH")
    parser.add_argument("-o", "--output-dir", default=None)
    args = parser.parse_args()

    if args.train:
        train_and_evaluate()
    elif args.score:
        score_file(args.score, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

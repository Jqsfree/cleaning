#!/usr/bin/env python3
"""exo_fitness 标题句向量否决器：MiniLM + LR。

监督：人工 QC T=KEEP、F=DROP（黑板讲解/非真人为 F）。
特征：title + channel，不含采集 keyword。宁漏勿杀。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

PROJECT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT / "models/exo_fitness_text_clf_f.pkl"
CALIB_PATH = PROJECT / "models/exo_fitness_text_clf_f_calibration.json"
DEFAULT_QC = (
    Path("/home/jqs/tmp/健身训练2_b99edc4e_qc_result.csv"),
    Path("/home/jqs/tmp/健身训练人工1_ea9f482e_qc_result.csv"),
)
MINILM_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MINILM_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
    / "snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
)
RANDOM_SEED = 42
FEATURE_FIELDS = ("title", "channel")


def build_text(row: pd.Series) -> str:
    title = str(row.get("title", "") or "") if pd.notna(row.get("title")) else ""
    channel = str(row.get("channel", "") or "") if pd.notna(row.get("channel")) else ""
    return re.sub(r"\s+", " ", f"{title} {channel}".strip())


def _label_series(frame: pd.DataFrame) -> pd.Series:
    if "qc_result" in frame.columns:
        return frame["qc_result"]
    if "qc_text_result" in frame.columns:
        return frame["qc_text_result"]
    raise ValueError("需要 qc_result 或 qc_text_result")


def load_training_frame(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path_arg in paths:
        path = Path(path_arg)
        if not path.is_file():
            raise FileNotFoundError(f"缺少 QC: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if "video_id" not in frame.columns or "title" not in frame.columns:
            raise ValueError(f"{path} 缺少 video_id/title")
        lab = _label_series(frame).astype(str).str.upper()
        frame = frame[lab.isin(["T", "F"])].copy()
        lab = _label_series(frame).astype(str).str.upper()
        frame["label_kind"] = lab
        frame["y"] = (lab != "F").astype(int)
        frame["_qc_round"] = path.stem
        frames.append(frame)
    if not frames:
        raise ValueError("至少需要一个 QC 文件")
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("video_id", keep="first").reset_index(drop=True)


class MiniLMEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name: str = MINILM_NAME,
        snapshot: str | None = str(MINILM_SNAPSHOT),
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.snapshot = snapshot
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        source = self.snapshot if self.snapshot and Path(self.snapshot).is_dir() else self.model_name
        self._model = SentenceTransformer(source, local_files_only=True)

    def fit(self, X, y=None):
        self._load()
        return self

    def transform(self, X):
        self._load()
        texts = [str(x) if x is not None else "" for x in X]
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 200,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state


def make_pipeline() -> Pipeline:
    MiniLMEncoder.__module__ = "exo_fitness_text_classifier"
    return Pipeline([
        ("emb", MiniLMEncoder()),
        ("clf", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED,
        )),
    ])


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, *, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=str)
    scores = np.asarray(scores, dtype=float)
    drop = scores < threshold
    dropped = labels[drop]
    n_drop = int(drop.sum())
    f_caught = int((dropped == "F").sum())
    t_hurt = int((dropped == "T").sum())
    n_f = int((labels == "F").sum())
    return {
        "drop_threshold": float(threshold),
        "n_drop": n_drop,
        "drop_coverage": n_drop / max(len(labels), 1),
        "drop_precision": f_caught / max(n_drop, 1),
        "f_caught": f_caught,
        "f_recall": f_caught / max(n_f, 1),
        "t_hurt": t_hurt,
        "u_hurt": 0,
        "u_hurt_rate": 0.0,
    }


def pick_strict_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    min_precision: float = 0.90,
    min_drop: int = 5,
) -> dict | None:
    candidates = []
    for threshold in np.round(np.arange(0.05, 0.61, 0.01), 2):
        row = threshold_metrics(labels, scores, threshold=float(threshold))
        if row["n_drop"] >= min_drop and row["drop_precision"] >= min_precision and row["t_hurt"] == 0:
            candidates.append(row)
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["f_caught"], row["drop_precision"]))


def _oof_scores(texts: list[str], y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    encoder = MiniLMEncoder()
    encoder.fit(texts)
    X = encoder.transform(texts)
    minority = min(int((y == 0).sum()), int((y == 1).sum()))
    n_splits = min(5, minority)
    if n_splits < 2:
        raise ValueError("T 与 F 各至少需要 2 条")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in cv.split(X, y):
        lr = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED,
        )
        lr.fit(X[train_idx], y[train_idx])
        oof[test_idx] = lr.predict_proba(X[test_idx])[:, 1]
    return oof, X


def train_and_calibrate(
    paths: Iterable[str | Path],
    *,
    model_path: Path = MODEL_PATH,
    calibration_path: Path = CALIB_PATH,
) -> dict:
    frame = load_training_frame(paths)
    texts = frame.apply(build_text, axis=1).tolist()
    y = frame["y"].to_numpy(dtype=np.int64)
    labels = frame["label_kind"].to_numpy(dtype=str)
    oof, X = _oof_scores(texts, y)
    strict = pick_strict_threshold(labels, oof)
    if strict is None:
        strict = {
            **threshold_metrics(labels, oof, threshold=0.25),
            "note": "OOF 无满足宁漏勿杀门槛；回退 drop<0.25",
        }

    pipe = make_pipeline()
    pipe.named_steps["emb"].fit(texts)
    pipe.named_steps["clf"].fit(X, y)
    MiniLMEncoder.__module__ = "exo_fitness_text_classifier"
    sys.modules["exo_fitness_text_classifier"] = sys.modules[__name__]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(pipe, fh)

    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "encoder": MINILM_NAME,
        "feature_fields": list(FEATURE_FIELDS),
        "n_train": int(len(frame)),
        "n_f": int((labels == "F").sum()),
        "n_t": int((labels == "T").sum()),
        "n_u": 0,
        "oof_auc_non_f": float(roc_auc_score(y, oof)),
        "oof_ap_non_f": float(average_precision_score(y, oof)),
        "strict": strict,
        "model_path": str(model_path),
        "qc_snapshots": [str(Path(p)) for p in paths],
        "notes": [
            "监督=人工 T/F；黑板讲解与非真人为 F",
            "title+channel，不含 keyword",
            "宁漏勿杀；ml_score 不当交付 KPI",
        ],
    }
    calibration_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="exo_fitness MiniLM 文本否决器")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--qc-snapshot", action="append", type=Path, dest="qc_snapshots")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--calibration", type=Path, default=CALIB_PATH)
    args = parser.parse_args()
    if not args.train:
        parser.print_help()
        return
    paths = tuple(args.qc_snapshots) if args.qc_snapshots else DEFAULT_QC
    print(json.dumps(
        train_and_calibrate(paths, model_path=args.model, calibration_path=args.calibration),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""exo_agriculture 标题语义否决器：MiniLM 采摘原型对比。

不要用人工 qc_result 直接训 LR：人工 F 里有真采摘标题，人工 T 里有畜牧，
OOF 无法同时做到 t_hurt=0 且 precision≥0.95。

改成「农业种植/采摘」vs「确定非采摘」原型 max-sim 对比：
  ml_score = sigmoid((sim_keep - sim_drop) / tau)
只自动丢高把握非采摘（畜牧/教程/风景/烹饪/游戏/新闻等）。
特征：title + channel，不含采集 keyword。
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
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT / "models/exo_agriculture_text_clf_f.pkl"
CALIB_PATH = PROJECT / "models/exo_agriculture_text_clf_f_calibration.json"
DEFAULT_HUMAN = (
    PROJECT
    / "data/runs/exo_agriculture/machine_0814/03_qc/human268_text_plus"
    / "exo农业_human_qc.csv"
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
DEFAULT_TAU = 0.08

# 种植/采摘劳作（对齐 qc.toml 的 t，不是「凡农业都留」）
KEEP_PROTOTYPES = (
    "people harvesting vegetables and fruit in a farm field",
    "picking strawberries mangoes grapes in an orchard third person",
    "farmers planting rice and picking crops packing into crates",
    "crop harvest in the field loading baskets onto a truck",
    "田间采摘水果蔬菜 收割庄稼装筐",
    "果园采摘芒果草莓葡萄 第三人称劳作",
    "种植水稻玉米小麦 田间劳作过程",
    "thu hoạch rau quả ngoài đồng đóng thùng",
    "cosecha de frutas y verduras en el campo",
    "recolección de café y plátano en la finca",
    "panen sayuran dan buah di ladang",
    "récolte de légumes dans le champ",
    "Gemüse und Obst auf dem Feld ernten",
    "आलू टमाटर आम की फसल खेत में तोड़ना",
)

# 确定非农业采摘（对齐 qc.toml 的 f；畜牧/风景/教程单独成原型）
DROP_PROTOTYPES = (
    "how to grow tomatoes garden tips tutorial beginners guide",
    "onion germination secret strategies planting lecture",
    "cooking recipe taste test mukbang homemade vinegar cake",
    "beautiful village travel documentary peaceful scenery paradise",
    "chicken pig duck fish livestock harvest sell at market",
    "sheep shearing cattle feed dairy farm animal raising",
    "silkworm silk farming catching eels turtles in a swamp",
    "farming simulator gameplay video game let's play",
    "bbc agriculture news interview podcast success story talking",
    "official music video kids cartoon nursery rhyme dvd",
    "building a wooden house homestead construction kitchen",
    "rainwater harvesting system off grid invention review",
    "tractor product advertisement machinery for sale",
    "种植教程 how to 讲解 养护秘诀 口播网课",
    "乡村风景旅拍 最美村庄 纪录片",
    "养鸡养猪捕鱼 不是种田采摘",
    "做菜试吃 食谱 mukbang",
)

CROP_KEEP_RE = re.compile(
    r"harvest|picking|picked|planting|planted|cosecha|recolec|r[eé]colte|"
    r"ernte|thu ho[aạ]ch|panen|采摘|收割|收获|种植|播种|hái|"
    r"\bpick(?:ing)?\b|\bharvesting\b",
    re.I,
)
ANIMAL_RE = re.compile(
    r"chicken|chickens|pig(?:let)?s?|duck|hen|sheep|cattle|cow|fish|"
    r"eel|turtle|snail|silkworm|geoduck|clam|dairy|livestock|"
    r"chicken eggs|whitetail|hunti",
    re.I,
)
CERTAIN_DROP_RE = re.compile(
    r"how to grow|garden tips|tutorial|germination|recipe|cook(?:ing)?|"
    r"taste test|mukbang|vinegar|cake|beautiful village|paradise|"
    r"documentary|travel|bbc |podcast|interview|success story|"
    r"silkworm|kids? dvd|cartoon|gameplay|simulator|"
    r"building.{0,20}(house|kitchen|cabin)|off-grid|"
    r"chicken|piglet|fish harvest|eels|turtles|sheep shearing|"
    r"cattle feed|micro dairy|geoduck|"
    r"种植教程|最美村庄|养鸡|捕鱼",
    re.I,
)
# 标题已写明作物采摘时，不因混入动物/生活词被阈值误杀
RESCUE_CROP_RE = re.compile(
    r"lychee|mango|strawberr|grape|apple|orange|banana|tomato|potato|"
    r"rice|corn|maize|wheat|chili|pepper|cabbage|lettuce|onion|garlic|"
    r"coffee|cocoa|cotton|durian|papaya|pineapple|watermelon|melon|"
    r"coconut|olive|avocado|peach|plum|cherr(?:y|ies)|blueberr|"
    r"raspberr|zucchini|cucumber|eggplant|peanut|cassava|sugarcane|"
    r"soy|barley|oat|guava|longan|rambutan|jackfruit|watermelon|"
    r"白菜|水稻|小麦|玉米|番茄|土豆|草莓|芒果|荔枝|葡萄|苹果|香蕉|"
    r"辣椒|黄瓜|茄子|花生|甘蔗|咖啡",
    re.I,
)
RESCUE_BLOCK_RE = re.compile(
    r"how to|tutorial|tips|gameplay|\bgames?\b|simulator|\brecipe\b|"
    r"cook(?:ing)?|mukbang|taste test|\bmilk(?:ing)?\b|silkworm|"
    r"chicken|piglet|fish harvest",
    re.I,
)


def should_rescue_crop_harvest(title: str) -> bool:
    text = str(title or "")
    return bool(
        CROP_KEEP_RE.search(text)
        and RESCUE_CROP_RE.search(text)
        and not RESCUE_BLOCK_RE.search(text)
    )


def build_text(row: pd.Series) -> str:
    title = str(row.get("title", "") or "") if pd.notna(row.get("title")) else ""
    channel = str(row.get("channel", "") or "") if pd.notna(row.get("channel")) else ""
    return re.sub(r"\s+", " ", f"{title} {channel}".strip())


def contrast_score(sim_keep: np.ndarray, sim_drop: np.ndarray, *, tau: float = DEFAULT_TAU) -> np.ndarray:
    """P(keep) ∈ (0,1)；sim_keep/sim_drop 为与原型的最大余弦。"""
    keep = np.asarray(sim_keep, dtype=float)
    drop = np.asarray(sim_drop, dtype=float)
    tau = max(float(tau), 1e-6)
    return 1.0 / (1.0 + np.exp(-(keep - drop) / tau))


def is_crop_keep_title(title: str) -> bool:
    text = str(title or "")
    return bool(CROP_KEEP_RE.search(text)) and not bool(ANIMAL_RE.search(text))


def is_certain_drop_title(title: str) -> bool:
    return bool(CERTAIN_DROP_RE.search(str(title or "")))


def load_training_frame(paths: Iterable[str | Path]) -> pd.DataFrame:
    """优先人工 qc_result；否则 qc_text_result。ERROR 剔除。"""
    frames: list[pd.DataFrame] = []
    for path_arg in paths:
        path = Path(path_arg)
        if not path.is_file():
            raise FileNotFoundError(f"缺少 QC 表: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if "video_id" not in frame.columns or "title" not in frame.columns:
            raise ValueError(f"{path} 缺少 video_id/title")

        human = frame.get("qc_result")
        if human is not None and human.astype(str).str.upper().isin(["T", "F"]).any():
            lab = human.astype(str).str.strip().str.upper()
            frame = frame.loc[lab.isin(["T", "F"])].copy()
            frame["label_kind"] = (
                frame["qc_result"].astype(str).str.strip().str.upper()
            )
            frame["y"] = (frame["label_kind"] != "F").astype(int)
        elif "qc_text_result" in frame.columns:
            frame = frame[frame["qc_text_result"].isin(["T", "F", "U"])].copy()
            frame["label_kind"] = frame["qc_text_result"]
            frame["y"] = (frame["qc_text_result"] != "F").astype(int)
        else:
            raise ValueError(f"{path} 需要 qc_result 或 qc_text_result")

        frame["_qc_round"] = path.stem
        frames.append(frame)
    if not frames:
        raise ValueError("至少需要一个 QC 表")
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("video_id", keep="first").reset_index(drop=True)


def fewshot_prototypes(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """人工表里抽少量标题当原型：作物采摘 T → keep；确定非采摘 F → drop。"""
    keep: list[str] = []
    drop: list[str] = []
    for _, row in frame.iterrows():
        title = str(row.get("title", "") or "")
        kind = str(row.get("label_kind", "") or "")
        if kind == "T" and is_crop_keep_title(title):
            keep.append(build_text(row))
        elif kind == "F" and is_certain_drop_title(title) and not is_crop_keep_title(title):
            drop.append(build_text(row))
    return keep, drop


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


class HarvestPrototypeClf(BaseEstimator, ClassifierMixin):
    """max-sim(keep) vs max-sim(drop)；pickle 只存原型向量。"""

    classes_ = np.asarray([0, 1])

    def __init__(
        self,
        keep_texts: list[str] | None = None,
        drop_texts: list[str] | None = None,
        tau: float = DEFAULT_TAU,
        encoder: MiniLMEncoder | None = None,
    ):
        self.keep_texts = list(keep_texts or KEEP_PROTOTYPES)
        self.drop_texts = list(drop_texts or DROP_PROTOTYPES)
        self.tau = float(tau)
        self.encoder = encoder or MiniLMEncoder()
        self.keep_emb_: np.ndarray | None = None
        self.drop_emb_: np.ndarray | None = None

    def fit(self, X=None, y=None):
        self.encoder.fit(self.keep_texts + self.drop_texts)
        self.keep_emb_ = self.encoder.transform(self.keep_texts)
        self.drop_emb_ = self.encoder.transform(self.drop_texts)
        return self

    def _sims(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if self.keep_emb_ is None or self.drop_emb_ is None:
            raise RuntimeError("HarvestPrototypeClf 尚未 fit")
        X = self.encoder.transform(texts)
        sim_keep = X @ self.keep_emb_.T
        sim_drop = X @ self.drop_emb_.T
        return sim_keep.max(axis=1), sim_drop.max(axis=1)

    def predict_proba(self, texts):
        sim_keep, sim_drop = self._sims(list(texts))
        pos = contrast_score(sim_keep, sim_drop, tau=self.tau)
        pos = np.clip(pos, 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - pos, pos])

    def predict(self, texts):
        return (self.predict_proba(texts)[:, 1] >= 0.5).astype(int)


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, *, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=str)
    scores = np.asarray(scores, dtype=float)
    drop = scores < threshold
    dropped = labels[drop]
    n_drop = int(drop.sum())
    f_caught = int((dropped == "F").sum())
    t_hurt = int((dropped == "T").sum())
    u_hurt = int((dropped == "U").sum())
    n_u = int((labels == "U").sum())
    n_f = int((labels == "F").sum())
    return {
        "drop_threshold": float(threshold),
        "n_drop": n_drop,
        "drop_coverage": n_drop / max(len(labels), 1),
        "drop_precision": f_caught / max(n_drop, 1),
        "f_caught": f_caught,
        "f_recall": f_caught / max(n_f, 1),
        "t_hurt": t_hurt,
        "u_hurt": u_hurt,
        "u_hurt_rate": u_hurt / max(n_u, 1) if n_u else 0.0,
    }


def pick_strict_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    min_precision: float = 0.95,
    max_u_hurt_rate: float = 0.05,
    min_drop: int = 5,
) -> dict | None:
    candidates = []
    for threshold in np.round(np.arange(0.05, 0.61, 0.01), 2):
        row = threshold_metrics(labels, scores, threshold=float(threshold))
        if (
            row["n_drop"] >= min_drop
            and row["drop_precision"] >= min_precision
            and row["t_hurt"] == 0
            and row["u_hurt_rate"] <= max_u_hurt_rate
        ):
            candidates.append(row)
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["f_caught"], row["drop_precision"]))


def _loo_scores(
    texts: list[str],
    keep_mask: np.ndarray,
    drop_mask: np.ndarray,
    encoder: MiniLMEncoder,
    *,
    tau: float,
) -> np.ndarray:
    """每个 few-shot 标题打分时剔除自身，避免原型泄漏。"""
    X = encoder.transform(texts)
    tmpl_keep = encoder.transform(list(KEEP_PROTOTYPES))
    tmpl_drop = encoder.transform(list(DROP_PROTOTYPES))
    scores = np.zeros(len(texts), dtype=float)
    keep_idx = np.flatnonzero(keep_mask)
    drop_idx = np.flatnonzero(drop_mask)
    for i in range(len(texts)):
        k_idx = keep_idx[keep_idx != i]
        d_idx = drop_idx[drop_idx != i]
        keep_emb = np.vstack([tmpl_keep, X[k_idx]]) if len(k_idx) else tmpl_keep
        drop_emb = np.vstack([tmpl_drop, X[d_idx]]) if len(d_idx) else tmpl_drop
        sim_k = float((X[i] @ keep_emb.T).max())
        sim_d = float((X[i] @ drop_emb.T).max())
        scores[i] = float(contrast_score([sim_k], [sim_d], tau=tau)[0])
    return scores


def train_and_calibrate(
    paths: Iterable[str | Path],
    *,
    model_path: Path = MODEL_PATH,
    calibration_path: Path = CALIB_PATH,
    tau: float = DEFAULT_TAU,
) -> dict:
    frame = load_training_frame(paths)
    texts = frame.apply(build_text, axis=1).tolist()
    titles = frame["title"].astype(str).tolist()
    human_labels = frame["label_kind"].to_numpy(dtype=str)

    eval_labels = np.full(len(frame), "U", dtype=object)
    for i, title in enumerate(titles):
        if is_crop_keep_title(title):
            eval_labels[i] = "T"
        elif is_certain_drop_title(title):
            eval_labels[i] = "F"

    keep_mask = np.array(
        [lab == "T" and is_crop_keep_title(t) for lab, t in zip(human_labels, titles)],
        dtype=bool,
    )
    drop_mask = np.array(
        [
            lab == "F" and is_certain_drop_title(t) and not is_crop_keep_title(t)
            for lab, t in zip(human_labels, titles)
        ],
        dtype=bool,
    )

    encoder = MiniLMEncoder()
    encoder.fit(texts)
    oof = _loo_scores(texts, keep_mask, drop_mask, encoder, tau=tau)

    gate_mask = eval_labels != "U"
    strict = pick_strict_threshold(eval_labels[gate_mask], oof[gate_mask])
    if strict is None:
        strict = {
            **threshold_metrics(eval_labels[gate_mask], oof[gate_mask], threshold=0.35),
            "note": "OOF 无满足宁漏勿杀门槛；回退 drop<0.35，应用前须看样本",
        }

    few_keep, few_drop = fewshot_prototypes(frame)
    clf = HarvestPrototypeClf(
        keep_texts=list(KEEP_PROTOTYPES) + few_keep,
        drop_texts=list(DROP_PROTOTYPES) + few_drop,
        tau=tau,
        encoder=encoder,
    )
    clf.fit()
    HarvestPrototypeClf.__module__ = "exo_agriculture_text_classifier"
    MiniLMEncoder.__module__ = "exo_agriculture_text_classifier"
    sys.modules["exo_agriculture_text_classifier"] = sys.modules[__name__]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(clf, fh)

    y_eval = (eval_labels[gate_mask] != "F").astype(int)
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "encoder": MINILM_NAME,
        "method": "harvest_prototype_contrast",
        "feature_fields": list(FEATURE_FIELDS),
        "tau": tau,
        "n_train": int(len(frame)),
        "n_keep_prototypes": len(clf.keep_texts),
        "n_drop_prototypes": len(clf.drop_texts),
        "n_eval_t": int((eval_labels == "T").sum()),
        "n_eval_f": int((eval_labels == "F").sum()),
        "n_eval_u": int((eval_labels == "U").sum()),
        "oof_auc_non_f": float(roc_auc_score(y_eval, oof[gate_mask])) if y_eval.min() != y_eval.max() else None,
        "oof_ap_non_f": float(average_precision_score(y_eval, oof[gate_mask])) if y_eval.min() != y_eval.max() else None,
        "strict": strict,
        "model_path": str(model_path),
        "qc_snapshots": [str(Path(p)) for p in paths],
        "notes": [
            "原型对比而非人工 T/F 的 LR；任务=丢非农业采摘",
            "校准闸门只用作物采摘标题 vs 确定非采摘标题",
            "title+channel，不含 keyword",
            "ml_score=P(采摘)；score<drop_threshold 才自动丢",
            "ml_score 不当交付 KPI",
        ],
    }
    calibration_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="exo_agriculture MiniLM 采摘原型否决器")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--qc-snapshot", action="append", type=Path, dest="qc_snapshots")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--calibration", type=Path, default=CALIB_PATH)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    args = parser.parse_args()
    if not args.train:
        parser.print_help()
        return
    paths = tuple(args.qc_snapshots) if args.qc_snapshots else (DEFAULT_HUMAN,)
    print(json.dumps(
        train_and_calibrate(
            paths,
            model_path=args.model,
            calibration_path=args.calibration,
            tau=args.tau,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""exo_service 店内商业服务劳动 — MiniLM + LR 文本否决器。

ml_score = P(店内服务劳动相关)；score < drop_threshold → 自动丢。
特征：仅 title（禁止 channel / keyword）。
v0 可用弱监督 CSV（qc_result=T/F）；人标入库后 --train 重训。

用法:
  .venv/bin/python3 experiments/exo_service_minilm_text_classifier.py --train \\
    --qc-snapshot work/exo_service_minilm_v0_0915/weak_train_v0.csv
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
MODEL_PATH = PROJECT / "models/exo_service_minilm_text_clf_v0.pkl"
CALIB_PATH = PROJECT / "models/exo_service_minilm_text_clf_v0_calibration.json"
MINILM_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MINILM_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
    / "snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
)
RANDOM_SEED = 42
FEATURE_FIELDS = ("title",)

KEEP_PROTOTYPES = (
    "barber cutting customer hair in salon third person",
    "hairdresser styling client hair wash cut blow dry",
    "nail technician manicure pedicure for customer in salon",
    "beautician facial massage for client in beauty parlor",
    "chef cooking in restaurant kitchen back of house",
    "barista making coffee for customers at cafe counter",
    "waiter serving food to table in restaurant",
    "cashier scanning groceries at supermarket checkout",
    "hotel housekeeping cleaning guest room",
    "auto mechanic repairing car in garage body shop",
    "janitor cleaning office floor commercial cleaning",
    "pet groomer bathing dog in grooming salon",
    "理发店给顾客剪发 洗剪吹 美发师工作",
    "美容院为顾客做护理 美甲 纹绣 操作过程",
    "餐厅后厨传菜 咖啡师做咖啡 服务员上菜",
    "超市收银 理货 保洁打扫 酒店客房服务",
    "汽修钣金喷漆 洗车 宠物美容",
)

DROP_PROTOTYPES = (
    "podcast interview talking head panel discussion lecture",
    "ceo business tips marketing startup company profile webinar",
    "product review unboxing haul advertisement commercial",
    "apartment tour documentary news travel vlog scenery",
    "makeup tutorial how to skincare routine diy at home",
    "hair transplant clinic advertisement fue fut promo",
    "short drama romance story entertainment clip",
    "TED talk presentation keynote conference",
    "播客 访谈 CEO 营销课 创业分享",
    "产品评测 开箱 种草 广告片",
    "公寓tour 纪录片 新闻 风景旅拍",
    "化妆教程 护肤routine 家用diy 植发广告",
    "短剧 爽文 娱乐剪辑",
)

# 低分但标题已强服务劳动 → 不自动丢（仍交人标/后续层）
SERVICE_RESCUE_RE = re.compile(
    r"(理发|美发|剪发|洗剪吹|美甲|纹绣|传菜|后厨|收银|保洁|"
    r"\bbarber\b|haircut|manicure|\bbarista\b|\bcashier\b)",
    re.I,
)
RESCUE_BLOCK_RE = re.compile(
    r"(教程|妆教|how\s*to|植发|transplant|podcast|interview|短剧)",
    re.I,
)


def build_text(row: pd.Series) -> str:
    title = str(row.get("title", "") or "") if pd.notna(row.get("title")) else ""
    return re.sub(r"\s+", " ", title.strip())


def should_rescue_service(title: str) -> bool:
    text = str(title or "")
    return bool(SERVICE_RESCUE_RE.search(text) and not RESCUE_BLOCK_RE.search(text))


def _normalize_tf_label(raw: str) -> str | None:
    s = str(raw or "").strip().upper()
    if s.startswith("T"):
        return "T"
    if s.startswith("F"):
        return "F"
    return None


def load_training_frame(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path_arg in paths:
        path = Path(path_arg)
        if not path.is_file():
            raise FileNotFoundError(f"缺少训练表: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if "video_id" not in frame.columns or "title" not in frame.columns:
            raise ValueError(f"{path} 缺少 video_id/title")

        used = False
        if "qc_result" in frame.columns:
            kinds = frame["qc_result"].map(_normalize_tf_label)
            if kinds.notna().any():
                frame = frame.loc[kinds.notna()].copy()
                frame["label_kind"] = kinds.loc[kinds.notna()].astype(str).values
                frame["y"] = (frame["label_kind"] != "F").astype(int)
                used = True
        if not used and "qc_text_result" in frame.columns:
            frame = frame[frame["qc_text_result"].isin(["T", "F", "U"])].copy()
            frame["label_kind"] = frame["qc_text_result"]
            frame["y"] = (frame["qc_text_result"] != "F").astype(int)
            used = True
        if not used:
            raise ValueError(f"{path} 需要 qc_result 或 qc_text_result")
        frame["_qc_round"] = path.stem
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("video_id", keep="last").reset_index(drop=True)


def resolve_torch_device(device: str | None = None) -> str:
    if device:
        return str(device)
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class MiniLMEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name: str = MINILM_NAME,
        snapshot: str | None = str(MINILM_SNAPSHOT),
        batch_size: int = 256,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.snapshot = snapshot
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._resolved_device: str | None = None

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        source = self.snapshot if self.snapshot and Path(self.snapshot).is_dir() else self.model_name
        resolved = resolve_torch_device(getattr(self, "device", None))
        self._resolved_device = resolved
        self._model = SentenceTransformer(source, local_files_only=True, device=resolved)
        print(
            f"[MiniLMEncoder] device={resolved} batch_size={self.batch_size}",
            flush=True,
        )

    def fit(self, X, y=None):
        self._load()
        return self

    def transform(self, X):
        self._load()
        texts = [str(x) if x is not None else "" for x in X]
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        state["_resolved_device"] = None
        return state


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
    }


def pick_recall_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    max_t_hurt_rate: float = 0.10,
    min_precision: float = 0.85,
    min_drop: int = 5,
) -> dict | None:
    n_t = max(int((labels == "T").sum()), 1)
    candidates = []
    for threshold in np.round(np.arange(0.05, 0.61, 0.01), 2):
        row = threshold_metrics(labels, scores, threshold=float(threshold))
        t_hurt_rate = row["t_hurt"] / n_t
        if (
            row["n_drop"] >= min_drop
            and row["drop_precision"] >= min_precision
            and t_hurt_rate <= max_t_hurt_rate
        ):
            candidates.append({**row, "t_hurt_rate": t_hurt_rate})
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["f_caught"], row["drop_precision"]))


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


def make_pipeline() -> Pipeline:
    MiniLMEncoder.__module__ = "exo_service_minilm_text_classifier"
    return Pipeline([
        ("emb", MiniLMEncoder()),
        ("clf", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED,
        )),
    ])


def _oof_lr_scores(texts: list[str], y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def _register_module() -> None:
    MiniLMEncoder.__module__ = "exo_service_minilm_text_classifier"
    sys.modules["exo_service_minilm_text_classifier"] = sys.modules[__name__]


def train_lr_and_calibrate(
    paths: Iterable[str | Path],
    *,
    model_path: Path = MODEL_PATH,
    calibration_path: Path = CALIB_PATH,
    weak: bool = False,
) -> dict:
    frame = load_training_frame(paths)
    texts = frame.apply(build_text, axis=1).tolist()
    y = frame["y"].to_numpy(dtype=np.int64)
    labels = frame["label_kind"].to_numpy(dtype=str)

    print(f"[train] encoding {len(texts):,} texts …", flush=True)
    oof, X = _oof_lr_scores(texts, y)

    strict = pick_strict_threshold(labels, oof, min_precision=0.90)
    if strict is None:
        strict = pick_strict_threshold(labels, oof, min_precision=0.85)
    if strict is None:
        strict = {
            **threshold_metrics(labels, oof, threshold=0.20),
            "note": "OOF 无满足 strict；回退 drop<0.20",
        }

    recall = pick_recall_threshold(
        labels, oof, max_t_hurt_rate=0.10, min_precision=0.85,
    )
    if recall is None:
        recall = {
            **threshold_metrics(labels, oof, threshold=0.25),
            "t_hurt_rate": 0.0,
            "note": "OOF 无满足 recall；回退 drop<0.25",
        }
    n_t = max(int((labels == "T").sum()), 1)
    recall["t_hurt_rate"] = float(recall.get("t_hurt_rate", recall["t_hurt"] / n_t))
    recall["t_coverage"] = float(1.0 - recall["t_hurt_rate"])
    recall["note"] = (
        "弱监督 OOF 参考阈；人标后须重标定" if weak
        else "默认 apply：t_hurt≤10% 内最大化 f_recall"
    )

    pipe = make_pipeline()
    pipe.named_steps["emb"].fit(texts)
    pipe.named_steps["clf"].fit(X, y)
    _register_module()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(pipe, fh)

    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "encoder": MINILM_NAME,
        "method": "mini_lm_lr",
        "task": "exo_service_in_store_labor",
        "supervision": "weak_v0" if weak else "human",
        "feature_fields": list(FEATURE_FIELDS),
        "n_train": int(len(frame)),
        "n_t": int((labels == "T").sum()),
        "n_f": int((labels == "F").sum()),
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_ap": float(average_precision_score(y, oof)),
        "policy": "宁杀不要漏（弱标阶段仅参考）",
        "recall": recall,
        "strict": strict,
        "keep_prototypes_n": len(KEEP_PROTOTYPES),
        "drop_prototypes_n": len(DROP_PROTOTYPES),
        "model_path": str(model_path.resolve()),
        "qc_snapshots": [str(Path(p).resolve()) for p in paths],
        "notes": [
            "口径=店内商业服务劳动（第三人称）",
            "仅 title，不含 channel/keyword",
            "ml_score / keep% 不当交付 KPI；人标 pass_rate 为准",
            "人标 SRS270 入库后应用金标重训并重锁 τ",
        ],
    }
    print(
        f"[train] n={result['n_train']} T={result['n_t']} F={result['n_f']} "
        f"AUC={result['oof_auc']:.3f} AP={result['oof_ap']:.3f} | "
        f"recall τ={recall['drop_threshold']} f_recall={recall['f_recall']:.1%} "
        f"t_hurt={recall['t_hurt']} ({recall['t_hurt_rate']:.1%})"
    )
    calibration_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="exo_service MiniLM 店内服务劳动文本否决器")
    ap.add_argument("--train", action="store_true")
    ap.add_argument(
        "--qc-snapshot", action="append", default=[],
        help="训练 CSV（可多次）；含 qc_result=T/F",
    )
    ap.add_argument("--model-path", type=Path, default=MODEL_PATH)
    ap.add_argument("--calibration-path", type=Path, default=CALIB_PATH)
    ap.add_argument(
        "--weak", action="store_true",
        help="标记本次为弱监督 v0（写入 calibration.supervision）",
    )
    args = ap.parse_args()
    if not args.train:
        ap.error("请指定 --train")
    paths = [Path(p) for p in args.qc_snapshot]
    if not paths:
        ap.error("请用 --qc-snapshot 指定训练集")
    train_lr_and_calibrate(
        paths,
        model_path=args.model_path,
        calibration_path=args.calibration_path,
        weak=args.weak,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

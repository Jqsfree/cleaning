#!/usr/bin/env python3
"""exo_pingpong 标题语义排序器：MiniLM + LR（与网球模型独立）。

口径：**乒乓球/桌球**双打/混双/团体合作竞争（中近景比赛实拍）。
场地网球、羽球、其他球类 = F（勿与 exo_tennis 混训）。
台球/单打 MS·WS/教程 = F。
特征：title + channel；ml_score = P(像乒乓双打/团体 T)；τ 越高 → 留下越像 T。

权重：``models/exo_pingpong_text_clf_f.pkl``（≠ ``exo_tennis_text_clf_f.pkl``）。
监督优先人工 qc_result（``machine_0922_pingpong/03_qc``）。
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
MODEL_PATH = PROJECT / "models/exo_pingpong_text_clf_f.pkl"
CALIB_PATH = PROJECT / "models/exo_pingpong_text_clf_f_calibration.json"

MINILM_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MINILM_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
    / "snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
)
RANDOM_SEED = 42
FEATURE_FIELDS = ("title", "channel")


def _default_qc_paths() -> tuple[Path, ...]:
    roots = (
        Path("/Users/muse/Downloads/humen-双人乒乓"),
        Path("/Users/muse/Downloads/humen-乒乓"),
        PROJECT / "data/runs/exo_团队协作/machine_0922_pingpong/03_qc",
        PROJECT / "data/runs/exo_pingpong/machine_0922/03_qc",  # 兼容旧路径
        PROJECT / "data/runs/exo_pingpong/machine_0813/03_qc",
        Path("/Users/muse/Downloads"),
    )
    for root in roots:
        if not root.is_dir():
            continue
        hits = sorted(
            p for p in root.glob("*qc_result.csv")
            if any(k in p.name.lower() for k in ("乒乓", "桌球", "pingpong", "pp_"))
            and "网球" not in p.name
            and "tennis" not in p.name.lower()
        )
        if hits:
            return tuple(hits)
        weak = root / "weak_gold_team_coop_0922.csv"
        if weak.is_file():
            return (weak,)
        labeled = root / "labeled.csv"
        if labeled.is_file():
            return (labeled,)
        train_export = root / "train_export.csv"
        if train_export.is_file():
            return (train_export,)
    weak = PROJECT / "data/runs/exo_团队协作/machine_0922_pingpong/03_qc/weak_gold_team_coop_0922.csv"
    if not weak.exists():
        weak = PROJECT / "data/runs/exo_pingpong/machine_0922/03_qc/weak_gold_team_coop_0922.csv"
    if weak.is_file():
        return (weak,)
    raise FileNotFoundError(
        "未找到乒乓 QC：请用 --qc-snapshot 指定，或放入 ~/Downloads/humen-双人乒乓/"
    )


def build_text(row: pd.Series) -> str:
    """title + channel（人标过滤口径）。"""
    title = str(row.get("title", "") or "") if pd.notna(row.get("title")) else ""
    channel = str(row.get("channel", "") or "") if pd.notna(row.get("channel")) else ""
    return re.sub(r"\s+", " ", f"{title} {channel}".strip())


def should_rescue_labor(title: str) -> bool:
    """窄救援：明确乒乓球双打/混双/团体比赛句式。"""
    s = str(title or "")
    if re.search(r"(?i)(billiard|snooker|台球)", s):
        return False
    return bool(
        re.search(
            r"(?i)((table\s*tennis|ping[\s-]*pong|乒乓球).{0,40}(doubles|team|混双|双打|团体)|"
            r"(mixed\s*doubles|men.?s\s*doubles|women.?s\s*doubles).{0,40}(table\s*tennis|wtt|ittf)|"
            r"wtt.*(xd|md|wd)|ittf.*(mixed\s*team|doubles))",
            s,
        )
    )


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
            raise FileNotFoundError(f"缺少 QC 表: {path}")
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
        if not used and "human_label" in frame.columns:
            hl = frame["human_label"].astype(str).str.strip().str.lower()
            mapped = hl.map({"pass": "T", "fail": "F", "t": "T", "f": "F"})
            if mapped.notna().any():
                frame = frame.loc[mapped.notna()].copy()
                frame["label_kind"] = mapped.loc[mapped.notna()].astype(str).values
                frame["y"] = (frame["label_kind"] != "F").astype(int)
                used = True
        if not used and "qc_text_result" in frame.columns:
            # 旧 LLM QC：F=DROP，T/U=KEEP（过渡）
            frame = frame[frame["qc_text_result"].isin(["T", "F", "U"])].copy()
            frame["label_kind"] = frame["qc_text_result"]
            frame["y"] = (frame["qc_text_result"] != "F").astype(int)
            used = True
        if not used:
            raise ValueError(f"{path} 需要 qc_result / human_label / qc_text_result")

        frame["_qc_round"] = path.stem
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("video_id", keep="last").reset_index(drop=True)


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


def pick_recall_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    max_t_hurt_rate: float = 0.10,
    min_precision: float = 0.90,
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


def pick_t_like_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    target_t_coverage: float = 0.50,
) -> dict:
    """交付旋钮：τ 越高越像 T；默认对齐 T 的 OOF 中位附近（约保住一半 T）。"""
    t_scores = scores[labels == "T"]
    if len(t_scores) == 0:
        return {**threshold_metrics(labels, scores, threshold=0.50), "note": "无 T；回退 0.50"}
    # target_t_coverage=0.50 → 阈值 ≈ T 的 p50（score≥τ 留下约一半 T）
    q = max(0.0, min(100.0, (1.0 - target_t_coverage) * 100.0))
    tau = float(np.percentile(t_scores, q))
    row = threshold_metrics(labels, scores, threshold=tau)
    n_t = max(int((labels == "T").sum()), 1)
    row["t_hurt_rate"] = row["t_hurt"] / n_t
    row["t_coverage"] = 1.0 - row["t_hurt_rate"]
    row["grid"] = f"OOF T 分位 p{q:.0f}（目标 t_coverage≈{target_t_coverage:.0%}）"
    row["note"] = "t_like：τ越高越像T；交付须独立人标验收"
    return row


def make_pipeline() -> Pipeline:
    MiniLMEncoder.__module__ = "exo_pingpong_text_classifier"
    return Pipeline([
        ("emb", MiniLMEncoder()),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_SEED,
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
    MiniLMEncoder.__module__ = "exo_pingpong_text_classifier"
    sys.modules["exo_pingpong_text_classifier"] = sys.modules[__name__]


def train_lr_and_calibrate(
    paths: Iterable[str | Path],
    *,
    model_path: Path = MODEL_PATH,
    calibration_path: Path = CALIB_PATH,
) -> dict:
    frame = load_training_frame(paths)
    texts = frame.apply(build_text, axis=1).tolist()
    y = frame["y"].to_numpy(dtype=np.int64)
    labels = frame["label_kind"].to_numpy(dtype=str)

    oof, X = _oof_lr_scores(texts, y)
    strict = pick_strict_threshold(labels, oof, min_precision=0.95)
    if strict is None:
        strict = pick_strict_threshold(labels, oof, min_precision=0.90)
    if strict is None:
        strict = {
            **threshold_metrics(labels, oof, threshold=0.20),
            "note": "OOF 无满足 strict；回退 drop<0.20",
        }

    recall_aggressive = pick_recall_threshold(
        labels, oof, max_t_hurt_rate=0.10, min_precision=0.90,
    )
    if recall_aggressive is None:
        recall_aggressive = {
            **threshold_metrics(labels, oof, threshold=0.25),
            "note": "OOF 无满足 recall_aggressive；回退 drop<0.25",
        }
    n_t = max(int((labels == "T").sum()), 1)
    recall_aggressive["t_hurt_rate"] = float(
        recall_aggressive.get("t_hurt_rate", recall_aggressive["t_hurt"] / n_t)
    )
    recall_aggressive["t_coverage"] = float(1.0 - recall_aggressive["t_hurt_rate"])
    recall_aggressive["grid"] = "OOF；t_hurt_rate≤10% 内最大化 f_recall"
    recall_aggressive["note"] = "可选更激进阈值；非默认 apply"

    OLD_T_HURT_CAP = 0.032
    recall = pick_recall_threshold(
        labels, oof, max_t_hurt_rate=OLD_T_HURT_CAP, min_precision=0.90,
    )
    if recall is None:
        recall = dict(recall_aggressive)
        recall["note"] = f"无法在 t_hurt≤{OLD_T_HURT_CAP:.1%} 内选阈；回退 recall_aggressive"
    else:
        recall["t_hurt_rate"] = float(recall.get("t_hurt_rate", recall["t_hurt"] / n_t))
        recall["t_coverage"] = float(1.0 - recall["t_hurt_rate"])
        recall["grid"] = (
            f"OOF；默认保住 T（t_hurt_rate≤{OLD_T_HURT_CAP:.1%}）内最大化 f_recall"
        )
        recall["note"] = "宽召回对照；交付/排序用 t_like（更高τ）"

    t_like = pick_t_like_threshold(labels, oof, target_t_coverage=0.50)

    t_scores = oof[labels == "T"]
    t_score_quantiles = {}
    if len(t_scores):
        for q in (5, 10, 25, 50, 75, 90, 95):
            t_score_quantiles[f"p{q}"] = float(np.percentile(t_scores, q))

    pipe = make_pipeline()
    pipe.named_steps["emb"].fit(texts)
    pipe.named_steps["clf"].fit(X, y)
    _register_module()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(pipe, fh)

    weak = any("weak_gold" in str(p) for p in paths)
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "encoder": MINILM_NAME,
        "method": "mini_lm_lr",
        "task": "pingpong_doubles_only",
        "sport": "pingpong",
        "family": "exo_团队协作",
        "feature_fields": list(FEATURE_FIELDS),
        "n_train": int(len(frame)),
        "n_t": int((labels == "T").sum()),
        "n_f": int((labels == "F").sum()),
        "n_u": int((labels == "U").sum()),
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_ap": float(average_precision_score(y, oof)),
        "t_score_quantiles": t_score_quantiles,
        "policy": "title+channel：ml_score=像乒乓球双打/混双/团体T；场地网球及其他球类/单打/台球=F；默认 apply=t_like",
        "t_like": t_like,
        "recall": recall,
        "recall_aggressive": recall_aggressive,
        "strict": strict,
        "model_path": str(model_path.resolve()),
        "qc_snapshots": [str(Path(p).resolve()) for p in paths],
        "notes": [
            "独立于 exo_tennis_text_clf_f（勿混用权重/金标）",
            "监督=乒乓批次人工 T/F（场地网球已翻 F）",
            "title+channel，不含 keyword",
            "口径=乒乓球双打/混双/团体；场地网球=F；台球/MS·WS单打=F",
            "默认 t_like；recall/strict 为对照",
            "ml_score 不当交付 KPI",
            ("weak_gold 过渡" if weak else "监督=人工 QC"),
        ],
    }
    print(
        f"[train] n={result['n_train']} T={result['n_t']} F={result['n_f']} "
        f"AUC={result['oof_auc']:.3f} AP={result['oof_ap']:.3f} | "
        f"t_like τ={t_like['drop_threshold']:.3f} "
        f"t_coverage={t_like.get('t_coverage', float('nan')):.1%} | "
        f"recall τ={recall['drop_threshold']} "
        f"f_recall={recall['f_recall']:.1%} t_hurt={recall['t_hurt']}"
    )
    calibration_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="exo_pingpong MiniLM 文本排序器")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--qc-snapshot", action="append", type=Path, dest="qc_snapshots")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--calibration", type=Path, default=CALIB_PATH)
    args = parser.parse_args()
    if not args.train:
        parser.print_help()
        return
    paths = tuple(args.qc_snapshots) if args.qc_snapshots else _default_qc_paths()
    print(json.dumps(
        train_lr_and_calibrate(paths, model_path=args.model, calibration_path=args.calibration),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

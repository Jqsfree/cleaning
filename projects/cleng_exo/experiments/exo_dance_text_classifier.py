#!/usr/bin/env python3
"""exo_dance「单人舞蹈」标题排序器：MiniLM + LR。

口径（依人标金标 kpop_0df26300，265 行 T=117 / F=148 反推）：
    本品类 = **单人**舞蹈影像（教学 / 练习 / 表演 / 即兴，一个人跳）。
    T：单人舞蹈教学（how to / lesson / tutorial / breakdown / step by step / drills）、
       单人舞蹈表演或即兴（belly dance solo、lyrical solo、freestyle、街舞单人 battle）、
       单人跟练（mirrored 单人练习、单人跟练健身操 / 拉伸）。
    F：团体·多人舞蹈（团体 kpop 练习室 / KPOP IN PUBLIC crew / line dance / 民俗团体）、
       双人舞（ballroom·latin·swing·salsa 伴侣套路教学与演示）、
       观看向直拍（직캠 / fancam·facecam / BE ORIGINAL / Performance ver. / MV）、
       舞蹈健身课（zumba·cardio dance workout）、
       非舞蹈主题串台（英语语法 / 吉他 / 圣经 / 羽毛球 / 美发 / 游戏 / 纯音乐发行）。

⚠ 金标噪声提示（务必与下一轮人标一起看）：
    「教学向」在 T / F 两侧都出现（T: belly dance「How to…」、salsa footwork tutorial；
    F: popping/locking/krump tutorial、"Learn the Charleston"、"Salsa Beginners 01"）。
    同频道亦同时出现 T 与 F（Marius & Elena Official、M2、ADTC Dance Camps…）。
    故本模型只作**排序**，不做口径仲裁；τ 一律以 OOF t_hurt 约束选取，
    交付前必须走 tools/batch_ops/sample_qc.py 独立人标验收（见 CLAUDE.md 治理原则）。

特征：仅 title（channel 在本品类金标里 T/F 混住，纳入即学频道偏见）。
权重：``models/exo_dance_text_clf_f.pkl``；标定 ``models/exo_dance_text_clf_f_calibration.json``。
    默认 τ = calibration.t_like（τ 越高 → 留下越像人标 T）。
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
if str(PROJECT / "02_脚本") not in sys.path:
    sys.path.insert(0, str(PROJECT / "02_脚本"))

import core.text_embeddings as _emb  # noqa: E402  文本向量缓存（跨 run 复用编码）

MODEL_PATH = PROJECT / "models/exo_dance_text_clf_f.pkl"
CALIB_PATH = PROJECT / "models/exo_dance_text_clf_f_calibration.json"

MINILM_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MINILM_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
    / "snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
)
RANDOM_SEED = 42
FEATURE_FIELDS = ("title",)


def _default_qc_paths() -> tuple[Path, ...]:
    """优先本批次人标入库产物 03_qc/labeled.csv；其次 Downloads 的 kpop/舞蹈 人标。"""
    labeled = PROJECT / "data/runs/exo_dance/machine_0923/03_qc/labeled.csv"
    if labeled.is_file():
        return (labeled,)
    downloads = Path("/Users/muse/Downloads")
    roots = (downloads / "建工", downloads)
    for root in roots:
        if not root.is_dir():
            continue
        hits = sorted(
            p for p in root.glob("*qc_result.csv")
            if any(k in p.name.lower() for k in ("kpop", "0df26300", "dance", "舞蹈"))
        )
        if hits:
            return tuple(hits)
    raise FileNotFoundError(
        "未找到 exo_dance 人工 QC：请用 --qc-snapshot 指定，"
        "或放入 ~/Downloads/建工/*kpop*qc_result*.csv"
    )


# ── 窄救援（仅对 ml_auto_drop 生效）────────────────────────────────────────
# 金标显示「tutorial」在 T/F 两侧都出现，救不得；只救**明确的单人信号**。
SOLO_SIGNAL_RE = re.compile(
    r"(?i)(\bsolos?\b|\bimprovis\w*|即兴|单人舞蹈|solo\s+(?:dance|performance|improv))"
)
# 命中即不救：团体 / 双人 / 舞cover / 直拍 / 练习室 / 健身 / 非舞蹈主题。
RESCUE_BLOCK_RE = re.compile(
    r"(?i)(K[\s-]*POP\s+IN\s+PUBLIC|dance\s+cover|댄스커버|line\s+dance|two[\s-]?step|"
    r"oversway|underarm\s+turn|\bcouples?\b|\bduet\b|\btrio\b|\bcrew\b|\bgroup\b|"
    r"fancam|facecam|직캠|be\s*original|dance\s*practic|official\s+(?:hd\s*)?video|"
    r"zumba|cardio|workout|\bglutes\b|posture|mobility|physio|"
    r"guitar|grammar|bible|badminton|hair\s+how)"
)


def should_rescue_dance(title: str) -> bool:
    """窄救援：标题明确写了单人/即兴，且不含团体·双人·直拍·非舞蹈阻断词。"""
    s = str(title or "")
    if RESCUE_BLOCK_RE.search(s):
        return False
    return bool(SOLO_SIGNAL_RE.search(s))


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
    """对齐全仓口径：startswith T/F（兼容 ``T|无声音``）；空/U/其他 → None。"""
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
                used = True
        if not used and "human_label" in frame.columns:
            mapped = (
                frame["human_label"].astype(str).str.strip().str.lower()
                .map({"pass": "T", "fail": "F", "t": "T", "f": "F"})
            )
            if mapped.notna().any():
                frame = frame.loc[mapped.notna()].copy()
                frame["label_kind"] = mapped.loc[mapped.notna()].astype(str).values
                used = True
        if not used and "qc_text_result" in frame.columns:
            sel = frame[frame["qc_text_result"].isin(["T", "F", "U"])].copy()
            if not sel.empty:
                frame = sel
                frame["label_kind"] = frame["qc_text_result"]
                used = True
        if not used:
            raise ValueError(f"{path} 需要 qc_result / human_label / qc_text_result")

        frame["y"] = (frame["label_kind"] != "F").astype(int)
        frame["_qc_round"] = path.stem
        frames.append(frame)

    if not frames:
        raise ValueError("至少需要一个 QC 表")
    out = pd.concat(frames, ignore_index=True)
    # 同 video_id 保留最后一批（对齐全仓口径）
    return out.drop_duplicates("video_id", keep="last").reset_index(drop=True)


def build_text(row: pd.Series) -> str:
    """仅 title（人标口径）。"""
    title = str(row.get("title", "") or "") if pd.notna(row.get("title")) else ""
    return re.sub(r"\s+", " ", title.strip())


class MiniLMEncoder(BaseEstimator, TransformerMixin):
    """MiniLM 句向量编码器；``transform`` 走内容寻址缓存（见 ``core.text_embeddings``）。

    关闭缓存：构造时 ``cache_dir=None``，或环境变量 ``TEXT_EMBED_CACHE=0``。
    """

    def __init__(
        self,
        model_name: str = MINILM_NAME,
        snapshot: str | None = str(MINILM_SNAPSHOT),
        batch_size: int = 256,
        device: str | None = None,
        cache_dir: str | None = "auto",
    ):
        self.model_name = model_name
        self.snapshot = snapshot
        self.batch_size = batch_size
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._resolved_device: str | None = None
        self._cache_obj = None
        self._cache_disabled = False

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
        print(f"[MiniLMEncoder] device={resolved} batch_size={self.batch_size}", flush=True)

    def fit(self, X, y=None):
        self._load()
        return self

    def _source(self) -> str:
        if self.snapshot and Path(self.snapshot).is_dir():
            return str(self.snapshot)
        return str(self.model_name)

    def _get_cache(self):
        """惰性建缓存连接；关闭、被独占或不可用时返回 None（回退直接编码）。"""
        if getattr(self, "_cache_obj", None) is not None:
            return self._cache_obj
        if getattr(self, "_cache_disabled", False):
            return None
        if not _emb.cache_enabled():
            return None
        configured = getattr(self, "cache_dir", "auto")
        if configured is None:
            return None
        root = _emb.default_cache_dir() if configured in (None, "auto") else Path(configured)
        try:
            self._cache_obj = _emb.TextEmbeddingCache(
                root,
                encoder_key=_emb.encoder_key(
                    self._source(),
                    self._model.get_sentence_embedding_dimension(),
                    "normalize=1",
                ),
                dim=self._model.get_sentence_embedding_dimension(),
                log=print,
            )
        except _emb.CacheUnavailable as exc:
            self._cache_disabled = True
            print(f"[embcache] WARN 缓存不可用，本次直接编码：{exc}", flush=True)
            return None
        return self._cache_obj

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def transform(self, X):
        self._load()
        texts = [str(x) if x is not None else "" for x in X]
        cache = self._get_cache()
        if cache is None:
            return self._encode(texts)
        return cache.encode(texts, self._encode)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        state["_resolved_device"] = None
        state["_cache_obj"] = None
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
    """宁漏勿杀：t_hurt=0 前提下尽量多抓 F。"""
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
    """宁杀勿漏：t_hurt 率上限内最大化 F 召回。"""
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
    """交付旋钮：τ 越高越像 T；默认落在 T 的 OOF 中位数附近（约保住一半 T）。"""
    t_scores = scores[labels == "T"]
    if len(t_scores) == 0:
        return {**threshold_metrics(labels, scores, threshold=0.50), "note": "无 T；回退 0.50"}
    q = max(0.0, min(100.0, (1.0 - target_t_coverage) * 100.0))
    tau = float(np.percentile(t_scores, q))
    row = threshold_metrics(labels, scores, threshold=tau)
    n_t = max(int((labels == "T").sum()), 1)
    row["t_hurt_rate"] = row["t_hurt"] / n_t
    row["t_coverage"] = 1.0 - row["t_hurt_rate"]
    row["grid"] = f"OOF T 分位 p{q:.0f}（目标 t_coverage≈{target_t_coverage:.0%}）"
    row["note"] = "t_like：τ越高越像T；交付须独立人标验收"
    return row


def pick_apply_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    max_t_hurt_rate: float = 0.032,
    min_drop: int = 5,
) -> dict | None:
    """本品类**默认 apply**：全仓「先保住 T」口径（t_hurt_rate 上限内最大化 f_caught）。

    与 ``pick_recall_threshold`` 的唯一差别是不设 min_precision——exo_dance 金标噪声大
    （教学向在 T/F 两侧都出现），drop 精度天花板只有 ~0.7，卡 0.90 会把阈值逼到无操作
    区间（τ=0.25 仅砍 4 行）。这里如实暴露"保 T 前提下能砍多少"，精度另列。
    """
    n_t = max(int((labels == "T").sum()), 1)
    candidates = []
    for threshold in np.round(np.arange(0.05, 0.61, 0.01), 2):
        row = threshold_metrics(labels, scores, threshold=float(threshold))
        row["t_hurt_rate"] = row["t_hurt"] / n_t
        if row["n_drop"] >= min_drop and row["t_hurt_rate"] <= max_t_hurt_rate:
            candidates.append(row)
    if not candidates:
        return None
    row = max(candidates, key=lambda r: (r["f_caught"], r["drop_precision"]))
    row["t_coverage"] = 1.0 - row["t_hurt_rate"]
    row["grid"] = f"OOF；保住 T（t_hurt_rate≤{max_t_hurt_rate:.1%}）内最大化 f_caught"
    row["note"] = "默认 apply；交付前须独立人标验收"
    return row


def make_pipeline() -> Pipeline:
    MiniLMEncoder.__module__ = "exo_dance_text_classifier"
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
    MiniLMEncoder.__module__ = "exo_dance_text_classifier"
    sys.modules["exo_dance_text_classifier"] = sys.modules[__name__]


def rescue_audit(labels: np.ndarray, titles: Iterable[str]) -> dict:
    """救援规则的回归检查：金标里被救到的 T / F 各几条（不应救到 F）。"""
    hit = np.array([should_rescue_dance(t) for t in titles], dtype=bool)
    return {
        "n_hit": int(hit.sum()),
        "t_rescued": int(((labels == "T") & hit).sum()),
        "f_mis_rescued": int(((labels == "F") & hit).sum()),
        "u_rescued": int(((labels == "U") & hit).sum()),
    }


def load_pool_ids(pool: str | Path | None) -> set[str] | None:
    """读 keep 池的 video_id 集合（用于把评估限定到生产分布上）。"""
    if not pool:
        return None
    frame = pd.read_csv(pool, encoding="utf-8-sig", usecols=["video_id"], dtype=str)
    return set(frame["video_id"].dropna().str.strip())


def train_lr_and_calibrate(
    paths: Iterable[str | Path],
    *,
    model_path: Path = MODEL_PATH,
    calibration_path: Path = CALIB_PATH,
    pool: str | Path | None = None,
) -> dict:
    frame = load_training_frame(paths)
    texts = frame.apply(build_text, axis=1).tolist()
    titles = frame["title"].astype(str).tolist()
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

    recall_aggressive = pick_recall_threshold(labels, oof, max_t_hurt_rate=0.10, min_precision=0.90)
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
    recall = pick_recall_threshold(labels, oof, max_t_hurt_rate=OLD_T_HURT_CAP, min_precision=0.90)
    if recall is None:
        recall = dict(recall_aggressive)
        recall["note"] = f"无法在 t_hurt≤{OLD_T_HURT_CAP:.1%} 内选阈；回退 recall_aggressive"
    else:
        recall["t_hurt_rate"] = float(recall.get("t_hurt_rate", recall["t_hurt"] / n_t))
        recall["t_coverage"] = float(1.0 - recall["t_hurt_rate"])
        recall["grid"] = f"OOF；默认保住 T（t_hurt_rate≤{OLD_T_HURT_CAP:.1%}）内最大化 f_recall"
        recall["note"] = "宽召回对照；交付/排序用 t_like（更高τ）"

    t_like = pick_t_like_threshold(labels, oof, target_t_coverage=0.50)

    # keep 池口径：人标本就是从机采池抽的，但黑名单已砍掉一批 F；
    # 生产上 MiniLM 只看 keep 池，故把「保 T」默认阈值定在 keep 子集上。
    pool_ids = load_pool_ids(pool)
    on_keep: dict | None = None
    if pool_ids is not None:
        in_pool = frame["video_id"].astype(str).str.strip().isin(pool_ids).to_numpy()
        n_pool_t = int(((labels == "T") & in_pool).sum())
        n_pool_f = int(((labels == "F") & in_pool).sum())
        if n_pool_t >= 5 and n_pool_f >= 5:
            on_keep = {
                "pool": str(Path(pool).resolve()),
                "n_t": n_pool_t,
                "n_f": n_pool_f,
                "auc": float(roc_auc_score((labels[in_pool] == "T").astype(int), oof[in_pool])),
                "apply": pick_apply_threshold(labels[in_pool], oof[in_pool]),
                "note": (
                    "在 keep 池金标子集上的 OOF 口径；mini 模型对本品类区分度弱，"
                    "apply 是「先保住 T」的唯一可辩护操作点"
                ),
            }

    apply_row = (on_keep or {}).get("apply") or pick_apply_threshold(labels, oof)
    if apply_row is None:
        apply_row = {**threshold_metrics(labels, oof, threshold=0.25),
                     "note": "OOF 无满足保 T 上限的阈值；回退 drop<0.25"}

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

    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "encoder": MINILM_NAME,
        "method": "mini_lm_lr",
        "task": "exo_dance_solo_dance_title",
        "category": "exo_dance",
        "feature_fields": list(FEATURE_FIELDS),
        "n_train": int(len(frame)),
        "n_t": int((labels == "T").sum()),
        "n_f": int((labels == "F").sum()),
        "n_u": int((labels == "U").sum()),
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_ap": float(average_precision_score(y, oof)),
        "t_score_quantiles": t_score_quantiles,
        "policy": (
            "ml_score = P(像人标 T)；T=单人舞蹈（教学/练习/表演/即兴）；"
            "F=团体·双人舞 / KPOP IN PUBLIC 舞cover / 舞台直拍·facecam·MV / "
            "舞蹈健身课 / 非舞蹈主题串台。默认 apply=t_like"
        ),
        "t_like": t_like,
        "apply": apply_row,
        "on_keep": on_keep,
        "recall": recall,
        "recall_aggressive": recall_aggressive,
        "strict": strict,
        "rescue_audit": rescue_audit(labels, titles),
        "model_path": str(model_path.resolve()),
        "qc_snapshots": [str(Path(p).resolve()) for p in paths],
        "notes": [
            "监督=人工 T/F（data/runs/exo_dance/machine_0923/03_qc/labeled.csv，265 行）",
            "仅 title；channel 在本品类 T/F 混住，纳入即学频道偏见",
            "金标噪声：教学向在 T/F 两侧都出现（tutorial 不可作救援信号），同频道亦有 T 有 F",
            "OOF 为随机 5 折上界；同频道泄漏使真实泛化更低，交付前须独立人标验收",
            "救援 should_rescue_dance 只认「单人/即兴」，且被团体·双人·直拍·非舞蹈阻断",
            "ml_score 不当交付 KPI；pass_rate 只认 tools/ingest_human_qc.py 入库的人标",
        ],
    }
    print(
        f"[train] n={result['n_train']} T={result['n_t']} F={result['n_f']} "
        f"AUC={result['oof_auc']:.3f} AP={result['oof_ap']:.3f} | "
        f"apply τ={apply_row['drop_threshold']:.3f} f_caught={apply_row.get('f_caught')} "
        f"t_hurt={apply_row.get('t_hurt')} prec={apply_row.get('drop_precision', 0):.2f} | "
        f"keep池 AUC={(on_keep or {}).get('auc', float('nan')):.3f} | "
        f"t_like τ={t_like['drop_threshold']:.3f} t_cov={t_like.get('t_coverage', float('nan')):.1%} | "
        f"rescue {result['rescue_audit']}"
    )
    calibration_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="exo_dance MiniLM 标题排序器")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--qc-snapshot", action="append", type=Path, dest="qc_snapshots")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--calibration", type=Path, default=CALIB_PATH)
    parser.add_argument("--pool", type=Path, default=None,
                        help="keep 池 CSV（含 video_id）：用于把「保 T」默认阈值定在生产分布上")
    args = parser.parse_args()
    if not args.train:
        parser.print_help()
        return
    paths = tuple(args.qc_snapshots) if args.qc_snapshots else _default_qc_paths()
    print(json.dumps(
        train_lr_and_calibrate(paths, model_path=args.model, calibration_path=args.calibration,
                               pool=args.pool),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()

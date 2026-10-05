#!/usr/bin/env python3
"""exo_factory 标题语义否决器：MiniLM + LR（默认）或原型对比（legacy）。

交付口径：第三人称工厂/车间/工坊制造劳动实拍（产线装配、机加工、工人在岗操作）。
DIY 家作/广告口播/游戏影视/访谈课件多为 F。

默认方法：QC T/F（优先人工 qc_result，其次 LLM qc_text_result）→ MiniLM + LR
  ml_score = P(像目标 T)；score < drop_threshold 才自动丢

legacy --method prototype：keep/drop 原型 max-sim 对比（仅对照）
特征：仅 title（不含 channel / keyword）。
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT / "02_脚本") not in sys.path:
    sys.path.insert(0, str(PROJECT / "02_脚本"))

import core.text_embeddings as _emb  # noqa: E402  文本向量缓存（跨 run 复用编码）

MODEL_PATH = PROJECT / "models/exo_factory_text_clf_f.pkl"
CALIB_PATH = PROJECT / "models/exo_factory_text_clf_f_calibration.json"
def _default_qc_paths() -> tuple[Path, ...]:
    """优先人标 Downloads；否则用本批 LLM textQC 快照。"""
    roots = (
        Path("/Users/muse/Downloads/humen-工厂生产"),
        Path("/Users/muse/Downloads/humen-工坊制造"),
        PROJECT / "data/runs/exo_factory/machine_0818/03_qc/minilm_train_n2000_0921",
        PROJECT / "data/runs/exo_factory/machine_0818/03_qc",
        Path("/Users/muse/Downloads"),
    )
    for root in roots:
        if not root.is_dir():
            continue
        hits = sorted(
            p for p in root.glob("*qc_result.csv")
            if any(k in p.name for k in ("工厂", "工坊", "factory", "制造"))
        )
        if hits:
            return tuple(hits)
        # LLM textQC 写回表 / 快照
        snaps = sorted(root.glob("*textqc_*.csv"))
        if snaps:
            return (snaps[-1],)
        labeled = root / "labeled.csv"
        if labeled.is_file():
            return (labeled,)
        sample = root / "工厂生产_sample_minilm_train_0921.csv"
        if sample.is_file():
            return (sample,)
    raise FileNotFoundError(
        "未找到工厂 QC：请用 --qc-snapshot 指定，或放入 ~/Downloads/humen-工厂生产/"
    )

MINILM_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MINILM_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
    / "snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
)
RANDOM_SEED = 42
FEATURE_FIELDS = ("title",)
DEFAULT_TAU = 0.08

# 工坊制造 / 工厂车间劳作（对齐 qc.toml 交付口径）
KEEP_PROTOTYPES = (
    "assembly line workers assembling products on factory floor third person",
    "CNC machine operator milling metal parts in workshop",
    "welding fabrication steel workers in manufacturing plant",
    "sewing garment factory production line stitching clothing",
    "injection molding plastic factory workers operating machines",
    "woodworking mill carpentry workshop cutting assembling furniture",
    "electronics PCB soldering assembly plant labor footage",
    "packaging line warehouse workers packing boxes on conveyor",
    "车间产线装配 机加工 工人在岗操作实拍",
    "工厂流水线生产 焊接 钣金 注塑",
    "工坊制造 木工 缝纫 手工装配过程",
    "机加工中心 CNC 车床 铣床操作",
)

DROP_PROTOTYPES = (
    "official music video lyrics song dance performance concert",
    "tv serial drama episode preview big brother reality show",
    "sky news politics interview podcast talk show recruitment ad",
    "gameplay minecraft roblox factory simulator let's play cartoon kids",
    "DIY home garage spray paint car yourself tutorial",
    "how does a factory work lecture animation explanation no labor",
    "product advertisement help your business corporate promo",
    "union protest rally outside factory gate not production work",
    "1930s historical archive documentary finished film",
    "yoga workout fitness gym makeup beauty grwm",
    "音乐 MV 歌词 舞蹈 演唱会 游戏实况",
    "家里 DIY 教程 招聘宣传片 原理课件动画",
)

FACTORY_ACTION_RE = re.compile(
    r"factory|workshop|manufactur|assembly\s*line|production\s*line|"
    r"cnc|lathe|milling|welding|fabrication|machining|"
    r"injection\s*mold|sewing\s*factory|garment\s*factory|"
    r"pcb|soldering|conveyor|packaging\s*line|"
    r"工厂|车间|产线|流水线|工坊|机加工|焊接|钣金|注塑|缝纫厂|装配",
    re.I,
)
CROP_KEEP_RE = FACTORY_ACTION_RE  # legacy alias

CERTAIN_DROP_RE = re.compile(
    r"sky\s*news|big\s*brother|bigg?\s*boss|survivor|"
    r"episode\s*preview|stock\s*market|ibps|lyrics?|"
    r"gameplay|minecraft|roblox|simulator|let.?s play|"
    r"diy\b|how\s*to\s*(paint|build|make)|garage\s*project|"
    r"how\s*does.{0,30}work|explained|animation\s*only|"
    r"recruit(?:ment|ing)|help\s*your\s*business|corporate\s*promo|"
    r"union\s*protest|rally\s*outside|"
    r"official music|music video|anime|cartoon|"
    r"yoga|workout|fitness|makeup|skincare|"
    r"音乐|MV|游戏实况|家里DIY|招聘宣传|原理讲解|课件",
    re.I,
)

RESCUE_BLOCK_RE = re.compile(
    r"how to|tutorial|tips|gameplay|\bgames?\b|simulator|"
    r"podcast|webinar|lecture|diy\b|recruit",
    re.I,
)


def should_rescue_labor(title: str) -> bool:
    """救援已停用（2026-09-22）。

    旧规则 `production line|cnc|factory floor` 过宽：会把
    「Production Line S1 E14」（剧集）、「CNC operator training」（教程）等
    明确 F 从低分捞回 keep。现有模型由人工 gold 训练，不再需要关键词救援。
    """
    return False



def build_text(row: pd.Series) -> str:
    title = str(row.get("title", "") or "") if pd.notna(row.get("title")) else ""
    return re.sub(r"\s+", " ", title.strip())


def contrast_score(sim_keep: np.ndarray, sim_drop: np.ndarray, *, tau: float = DEFAULT_TAU) -> np.ndarray:
    keep = np.asarray(sim_keep, dtype=float)
    drop = np.asarray(sim_drop, dtype=float)
    tau = max(float(tau), 1e-6)
    return 1.0 / (1.0 + np.exp(-(keep - drop) / tau))


def is_farm_action_title(title: str) -> bool:
    """工厂劳作标题粗筛（历史函数名保留）。"""
    return bool(FACTORY_ACTION_RE.search(str(title or "")))


def is_crop_keep_title(title: str) -> bool:
    """向后兼容。"""
    return is_farm_action_title(title)


def is_certain_drop_title(title: str) -> bool:
    return bool(CERTAIN_DROP_RE.search(str(title or "")))


def _normalize_tf_label(raw: str) -> str | None:
    """对齐唱跳：startswith T/F（兼容 ``T|无声音``）；空/U/其他 → None。"""
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

        used_human = False
        if "qc_result" in frame.columns:
            kinds = frame["qc_result"].map(_normalize_tf_label)
            if kinds.notna().any():
                frame = frame.loc[kinds.notna()].copy()
                frame["label_kind"] = kinds.loc[kinds.notna()].astype(str).values
                frame["y"] = (frame["label_kind"] != "F").astype(int)
                used_human = True
        if not used_human and "human_label" in frame.columns:
            hl = frame["human_label"].astype(str).str.strip().str.lower()
            mapped = hl.map({"pass": "T", "fail": "F", "t": "T", "f": "F"})
            if mapped.notna().any():
                frame = frame.loc[mapped.notna()].copy()
                frame["label_kind"] = mapped.loc[mapped.notna()].astype(str).values
                frame["y"] = (frame["label_kind"] != "F").astype(int)
                used_human = True
        if not used_human and "qc_text_result" in frame.columns:
            # LLM silver：只用确定 T/F 训练（丢弃 U/ERROR）
            frame = frame[frame["qc_text_result"].isin(["T", "F"])].copy()
            if frame.empty:
                raise ValueError(f"{path} qc_text_result 无可用 T/F")
            frame["label_kind"] = frame["qc_text_result"].astype(str)
            frame["y"] = (frame["label_kind"] != "F").astype(int)
            used_human = True
        if not used_human:
            raise ValueError(f"{path} 需要 qc_result（人工）或 qc_text_result（LLM）")

        frame["_qc_round"] = path.stem
        frames.append(frame)
    if not frames:
        raise ValueError("至少需要一个 QC 表")
    out = pd.concat(frames, ignore_index=True)
    # 同 video_id 保留最后一批（对齐唱跳 keep=last）
    return out.drop_duplicates("video_id", keep="last").reset_index(drop=True)


def fewshot_prototypes(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """prototype 模式：人工 T → keep；确定噪声 F → drop。"""
    keep: list[str] = []
    drop: list[str] = []
    for _, row in frame.iterrows():
        title = str(row.get("title", "") or "")
        kind = str(row.get("label_kind", "") or "")
        if kind == "T":
            keep.append(build_text(row))
        elif kind == "F" and is_certain_drop_title(title):
            drop.append(build_text(row))
    return keep, drop


def resolve_torch_device(device: str | None = None) -> str:
    """Prefer MPS (Apple) → CUDA → CPU. Explicit `device` wins."""
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
    """MiniLM 句向量编码器；``transform`` 走内容寻址缓存（见 ``core.text_embeddings``）。

    缓存让同一 title 只编码一次：换 LR 头 / 换 τ / 重打分都只读向量，不再重跑 MiniLM。
    关闭：构造时 ``cache_dir=None``，或环境变量 ``TEXT_EMBED_CACHE=0``。
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

    # ── 编码缓存 ────────────────────────────────────────────────────────────
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
            self._cache_disabled = True  # 本次运行不再重试
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
        return state


class FactoryPrototypeClf(BaseEstimator, ClassifierMixin):
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
            raise RuntimeError("FactoryPrototypeClf 尚未 fit")
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
    """旧策略（宁漏勿杀）：t_hurt=0 前提下尽量多抓 F。保留供对照。"""
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
    """宁杀不要漏：在 t_hurt 率≤上限内最大化 F 召回（与 CLIP/metadata 闸门同量级）。"""
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


def make_pipeline() -> Pipeline:
    MiniLMEncoder.__module__ = "exo_factory_text_classifier"
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


def _loo_prototype_scores(
    texts: list[str],
    keep_mask: np.ndarray,
    drop_mask: np.ndarray,
    encoder: MiniLMEncoder,
    *,
    tau: float,
) -> np.ndarray:
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


def _register_module() -> None:
    FactoryPrototypeClf.__module__ = "exo_factory_text_classifier"
    MiniLMEncoder.__module__ = "exo_factory_text_classifier"
    sys.modules["exo_factory_text_classifier"] = sys.modules[__name__]


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
            "note": "OOF 无满足旧 strict（宁漏勿杀）门槛；回退 drop<0.20",
        }

    recall_aggressive = pick_recall_threshold(
        labels, oof, max_t_hurt_rate=0.10, min_precision=0.90,
    )
    if recall_aggressive is None:
        recall_aggressive = {
            **threshold_metrics(labels, oof, threshold=0.25),
            "note": "OOF 无满足 recall（宁杀不要漏）门槛；回退 drop<0.25",
        }
    n_t = max(int((labels == "T").sum()), 1)
    for row in (recall_aggressive,):
        thr = float(row.get("t_hurt_rate", row["t_hurt"] / n_t))
        row["t_hurt_rate"] = thr
        row["t_coverage"] = float(1.0 - thr)

    # 默认 apply：对齐旧基线 T 覆盖（≈96.8% / t_hurt≤3.2%），再最大化 f_recall
    OLD_T_HURT_CAP = 0.032
    recall = pick_recall_threshold(
        labels, oof, max_t_hurt_rate=OLD_T_HURT_CAP, min_precision=0.90,
    )
    if recall is None:
        recall = dict(recall_aggressive)
        recall["note"] = (
            f"无法在 t_hurt≤{OLD_T_HURT_CAP:.1%} 内选阈；回退 recall_aggressive"
        )
    else:
        recall["t_hurt_rate"] = float(recall.get("t_hurt_rate", recall["t_hurt"] / n_t))
        recall["t_coverage"] = float(1.0 - recall["t_hurt_rate"])
        recall["grid"] = (
            f"OOF；默认保住 T（t_hurt_rate≤{OLD_T_HURT_CAP:.1%}）内最大化 f_recall"
        )
        recall["note"] = "默认 apply；更激进见 recall_aggressive（t_hurt≤10%）"
    recall_aggressive["grid"] = "OOF；t_hurt_rate≤10% 内最大化 f_recall"
    recall_aggressive["note"] = "可选更激进阈值；非默认 apply"

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
        "task": "factory_workshop_labor",
        "feature_fields": list(FEATURE_FIELDS),
        "n_train": int(len(frame)),
        "n_t": int((labels == "T").sum()),
        "n_f": int((labels == "F").sum()),
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_ap": float(average_precision_score(y, oof)),
        "t_score_quantiles": t_score_quantiles,
        "policy": "文本治理v2：默认保住 T（t_hurt≤3.2%）内最大化 f_recall；交付用 t_like（更高τ）",
        "recall": recall,
        "recall_aggressive": recall_aggressive,
        "strict": strict,
        "model_path": str(model_path.resolve()),
        "qc_snapshots": [str(Path(p).resolve()) for p in paths],
        "notes": [
            "监督=QC T/F（优先人工；否则 LLM silver）；口径=工厂/工坊制造劳作实拍",
            "仅 title，不含 channel/keyword",
            "标签解析：人工 startswith T/F；LLM qc_text_result∈{T,F}；同 video_id keep=last",
            "默认 recall：早期宽召回；过滤无关 title 用 calibration.t_like（τ越高越像T）",
            "recall_aggressive：t_hurt≤10% 上限内最大化 f_recall（历史对照）",
            "黑名单只做非工厂主题闸门；交付 KPI 须独立人标验收",
            "t_coverage=1-t_hurt_rate；strict 为宁漏勿杀对照；ml_score 不当交付 KPI",
        ],
    }
    print(
        f"[train] n={result['n_train']} T={result['n_t']} F={result['n_f']} "
        f"AUC={result['oof_auc']:.3f} AP={result['oof_ap']:.3f} | "
        f"recall τ={recall['drop_threshold']} f_recall={recall['f_recall']:.1%} "
        f"t_hurt={recall['t_hurt']} ({recall['t_hurt_rate']:.1%}) "
        f"t_coverage={recall['t_coverage']:.1%} | "
        f"aggressive τ={recall_aggressive['drop_threshold']} "
        f"f_recall={recall_aggressive['f_recall']:.1%} "
        f"t_coverage={recall_aggressive['t_coverage']:.1%}"
    )
    if calibration_path.is_file():
        try:
            old = json.loads(calibration_path.read_text(encoding="utf-8"))
            if old.get("t_like"):
                result["t_like"] = old["t_like"]
                result["t_like"] = {
                    **old["t_like"],
                    "note": (
                        str(old["t_like"].get("note") or "")
                        + "；重训后请在 keep 上重校准 t_like"
                    ).strip("；"),
                }
                if old.get("stage2_on_keep"):
                    result["stage2_on_keep"] = old["stage2_on_keep"]
        except (json.JSONDecodeError, OSError):
            pass
    result["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    calibration_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def train_prototype_and_calibrate(
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

    keep_mask = human_labels == "T"
    drop_mask = np.array(
        [lab == "F" and is_certain_drop_title(t) for lab, t in zip(human_labels, titles)],
        dtype=bool,
    )

    encoder = MiniLMEncoder()
    encoder.fit(texts)
    oof = _loo_prototype_scores(texts, keep_mask, drop_mask, encoder, tau=tau)

    strict = pick_strict_threshold(human_labels, oof)
    if strict is None:
        strict = {
            **threshold_metrics(human_labels, oof, threshold=0.25),
            "note": "OOF 无满足旧 strict（宁漏勿杀）门槛；回退 drop<0.25",
        }

    few_keep, few_drop = fewshot_prototypes(frame)
    clf = FactoryPrototypeClf(
        keep_texts=list(KEEP_PROTOTYPES) + few_keep,
        drop_texts=list(DROP_PROTOTYPES) + few_drop,
        tau=tau,
        encoder=encoder,
    )
    clf.fit()
    _register_module()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(clf, fh)

    y = (human_labels != "F").astype(int)
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "encoder": MINILM_NAME,
        "method": "farm_action_prototype_contrast",
        "feature_fields": list(FEATURE_FIELDS),
        "tau": tau,
        "n_train": int(len(frame)),
        "n_keep_prototypes": len(clf.keep_texts),
        "n_drop_prototypes": len(clf.drop_texts),
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_ap": float(average_precision_score(y, oof)),
        "strict": strict,
        "model_path": str(model_path),
        "qc_snapshots": [str(Path(p)) for p in paths],
        "notes": [
            "legacy 原型对比；默认请用 --method lr",
            "ml_score 不当交付 KPI",
        ],
    }
    calibration_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def train_and_calibrate(
    paths: Iterable[str | Path],
    *,
    method: str = "lr",
    model_path: Path = MODEL_PATH,
    calibration_path: Path = CALIB_PATH,
    tau: float = DEFAULT_TAU,
) -> dict:
    if method == "prototype":
        return train_prototype_and_calibrate(
            paths, model_path=model_path, calibration_path=calibration_path, tau=tau,
        )
    if method != "lr":
        raise ValueError(f"未知 method: {method!r}（可用 lr / prototype）")
    return train_lr_and_calibrate(paths, model_path=model_path, calibration_path=calibration_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="exo_factory MiniLM 工坊制造文本否决器")
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--method", choices=("lr", "prototype"), default="lr",
        help="lr=人工 T/F + LR（默认）；prototype=原型对比 legacy",
    )
    parser.add_argument("--qc-snapshot", action="append", type=Path, dest="qc_snapshots")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--calibration", type=Path, default=CALIB_PATH)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="仅 prototype 模式")
    args = parser.parse_args()
    if not args.train:
        parser.print_help()
        return
    paths = tuple(args.qc_snapshots) if args.qc_snapshots else _default_qc_paths()
    print(json.dumps(
        train_and_calibrate(
            paths,
            method=args.method,
            model_path=args.model,
            calibration_path=args.calibration,
            tau=args.tau,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

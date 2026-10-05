#!/usr/bin/env python3
"""exo_livestock 标题语义否决器：MiniLM + LR（默认）或原型对比（legacy）。

交付口径：渔业牧业劳作（第三人称语境下的撒网捕鱼/放牧挤奶/喂饲圈养劳作），
不限于「休闲钓/料理」关键词——徒手/传统网具捕鱼、放牧、挤奶、喂饲等算 T。

默认方法：人工 qc_result T/F → MiniLM 向量 + LogisticRegression
  ml_score = P(渔业牧业劳作)；score < drop_threshold 才自动丢（宁杀不要漏）

legacy --method prototype：keep/drop 原型 max-sim 对比（旧采摘口径，仅对照）
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
MODEL_PATH = PROJECT / "models/exo_livestock_text_clf_f.pkl"
CALIB_PATH = PROJECT / "models/exo_livestock_text_clf_f_calibration.json"
def _default_qc_paths() -> tuple[Path, ...]:
    """优先 Downloads/humen-渔业牧业，其次批次入库目录 human_downloads_0915。"""
    roots = (
        Path("/Users/muse/Downloads/humen-渔业牧业"),
        PROJECT / "data/runs/exo_livestock/machine_0818/03_qc/srs_v03_c90_0916",
        Path("/Users/muse/Downloads"),
    )
    for root in roots:
        if not root.is_dir():
            continue
        hits = sorted(
            p for p in root.glob("*qc_result.csv")
            if "渔业牧业" in p.name or "livestock" in p.name.lower()
        )
        if not hits:
            hits = sorted(root.glob("*渔业牧业*qc_result.csv"))
        if hits:
            return tuple(hits)
    raise FileNotFoundError(
        "未找到人工 QC：请用 --qc-snapshot 指定，或放入 ~/Downloads/humen-渔业牧业/"
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

# 渔业牧业劳作（广义工劳，对齐人工 T 口径）
KEEP_PROTOTYPES = (
    "people working in a farm field planting harvesting and packing vegetables",
    "farmers picking fruit in an orchard loading crates onto a truck",
    "greenhouse workers tending tomato plants and harvesting crops",
    "preparing garden beds planting seeds irrigation drip watering in the field",
    "raising silkworms on bamboo beds mulberry leaf farm work",
    "mushroom picking in the forest farm harvest baskets",
    "tilling soil sowing rice corn wheat field labor third person",
    "sorting and packing fresh produce at the farm market",
    "田间劳作 种植整地 灌溉 采摘 装筐 搬运",
    "温室里管理作物 真人劳作过程",
    "养蚕 采桑 田间日常农作",
    "thu hoạch và trồng trọt ngoài đồng",
    "trabajo agrícola en el campo cosecha y siembra",
    "panen sayur buah di ladang kerja petani",
)

# 确定非渔业牧业劳作（教程/烹饪/游戏/纯风景/口播课等）
DROP_PROTOTYPES = (
    "how to grow tomatoes garden tips tutorial beginners guide lecture",
    "online course webinar planting secrets expert talk",
    "cooking recipe taste test mukbang homemade vinegar cake kitchen",
    "beautiful village travel documentary peaceful scenery paradise postcard",
    "farming simulator gameplay video game let's play cartoon",
    "bbc agriculture news interview podcast success story talking head",
    "official music video kids cartoon nursery rhyme dvd anime",
    "building a wooden house homestead construction kitchen diy furniture",
    "rainwater harvesting system off grid invention product review ad",
    "tractor product advertisement machinery for sale showroom",
    "种植教程 how to 讲解 养护秘诀 口播网课",
    "乡村风景旅拍 最美村庄 纪录片 无人劳作",
    "做菜试吃 食谱 mukbang",
)

# 标题像农作劳作（用于 rescue / prototype few-shot 粗筛，不含动物 blanket ban）
FARM_ACTION_RE = re.compile(
    r"harvest|harvesting|picking|picked|planting|planted|farming|farm(?:er|ing)?|"
    r"field work|greenhouse|orchard|irrigation|drip|watering|tilling|tillage|"
    r"sowing|seed(?:ling|s)?|bed prep|mulch|compost|"
    r"silkworm|mulberry|mushroom|"
    r"cosecha|recolec|r[eé]colte|ernte|thu ho[aạ]ch|panen|"
    r"采摘|收割|收获|种植|播种|整地|灌溉|温室|农作|劳作|装筐|"
    r"\bpick(?:ing)?\b|\bharvesting\b",
    re.I,
)
# 向后兼容旧名
CROP_KEEP_RE = FARM_ACTION_RE

CERTAIN_DROP_RE = re.compile(
    r"how to grow|garden tips|tutorial|germination|online course|webinar|"
    r"masterclass|lecture|tips and tricks|"
    r"recipe|cook(?:ing)?|taste test|mukbang|vinegar|cake|"
    r"beautiful village|paradise|travel vlog|documentary|"
    r"bbc |podcast|interview|success story|talking head|"
    r"kids? dvd|cartoon|gameplay|simulator|let.?s play|"
    r"building.{0,20}(house|kitchen|cabin)|off-grid|"
    r"rainwater harvesting|product review|for sale|showroom|"
    r"official music|music video|anime|"
    r"种植教程|最美村庄|口播|网课|做菜|试吃",
    re.I,
)
FARM_CROP_RE = re.compile(
    r"lychee|mango|strawberr|grape|apple|orange|banana|peach|tomato|potato|"
    r"rice|corn|maize|wheat|chili|pepper|cabbage|lettuce|onion|garlic|"
    r"coffee|cocoa|mulberry|mushroom|silkworm|"
    r"vegetable|crop|farm|greenhouse|orchard|"
    r"白菜|水稻|小麦|玉米|番茄|土豆|草莓|芒果|葡萄|苹果|香蕉|"
    r"辣椒|黄瓜|茄子|花生|甘蔗|咖啡|蚕|菌",
    re.I,
)
RESCUE_BLOCK_RE = re.compile(
    r"how to|tutorial|tips|gameplay|\bgames?\b|simulator|\brecipe\b|"
    r"cook(?:ing)?|mukbang|taste test|podcast|webinar|lecture",
    re.I,
)


def should_rescue_labor(title: str) -> bool:
    """窄救援：明确徒手/撒网劳作句式，避免被低分误杀。"""
    s = str(title or "")
    return bool(
        re.search(
            r"(?i)(catching.{0,20}by\s+hand|cast(?:ing)?\s+net|throw\s+net|bare\s*hand|"
            r"徒手|撒网|鱼笼|挤奶|放牧|喂饲)",
            s,
        )
    )


def build_text(row: pd.Series) -> str:
    title = str(row.get("title", "") or "") if pd.notna(row.get("title")) else ""
    return re.sub(r"\s+", " ", title.strip())


def contrast_score(sim_keep: np.ndarray, sim_drop: np.ndarray, *, tau: float = DEFAULT_TAU) -> np.ndarray:
    keep = np.asarray(sim_keep, dtype=float)
    drop = np.asarray(sim_drop, dtype=float)
    tau = max(float(tau), 1e-6)
    return 1.0 / (1.0 + np.exp(-(keep - drop) / tau))


def is_farm_action_title(title: str) -> bool:
    return bool(FARM_ACTION_RE.search(str(title or "")))


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
        if not used_human and "qc_text_result" in frame.columns:
            frame = frame[frame["qc_text_result"].isin(["T", "F", "U"])].copy()
            frame["label_kind"] = frame["qc_text_result"]
            frame["y"] = (frame["qc_text_result"] != "F").astype(int)
            used_human = True
        if not used_human:
            raise ValueError(f"{path} 需要 qc_result 或 qc_text_result")

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


class LivestockPrototypeClf(BaseEstimator, ClassifierMixin):
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
            raise RuntimeError("LivestockPrototypeClf 尚未 fit")
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
    MiniLMEncoder.__module__ = "exo_livestock_text_classifier"
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
    LivestockPrototypeClf.__module__ = "exo_livestock_text_classifier"
    MiniLMEncoder.__module__ = "exo_livestock_text_classifier"
    sys.modules["exo_livestock_text_classifier"] = sys.modules[__name__]


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
        "task": "livestock_fish_herd_labor",
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
            "监督=人工 T/F；口径=像人标 T（渔牧劳作实拍），非宽泛钓鱼主题",
            "仅 title，不含 channel/keyword",
            "标签解析对齐唱跳：startswith T/F；同 video_id keep=last",
            "默认 recall：早期宽召回；交付/收紧用 calibration.t_like（τ越高越像T）",
            "recall_aggressive：t_hurt≤10% 上限内最大化 f_recall（历史对照）",
            "黑名单只做非渔牧主题闸门，见 docs/exo_livestock_text_governance_v1.md",
            "t_coverage=1-t_hurt_rate；strict 为宁漏勿杀对照；ml_score 不当交付 KPI",
            "t_like 默认在验收金标上扫 τ（越高越像 T）；交付 KPI 须独立新人标",
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
    clf = LivestockPrototypeClf(
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
    parser = argparse.ArgumentParser(description="exo_livestock MiniLM 渔业牧业劳作文本否决器")
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

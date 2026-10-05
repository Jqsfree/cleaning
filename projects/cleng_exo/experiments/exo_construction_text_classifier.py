#!/usr/bin/env python3
"""exo_construction 标题语义否决器：MiniLM + LR。

交付口径：**像人标 T 的「有人的建筑施工」**——真实施工现场/工序教学里
有工人在劳作（砌砖、抹灰、浇筑、支模、铺贴、屋面、焊接、开挖、吊装、装修…）。
设备展示 / 机械表演 / 播客口播 / 展会 / 考试理论 / 房产中介 / 影视游戏 多为 F——
以人工金标为准。

默认方法：人工 qc_result T/F → MiniLM 向量 + LogisticRegression
  ml_score = P(像人标 T)；score < drop_threshold 才自动丢

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
MODEL_PATH = PROJECT / "models/exo_construction_text_clf_f.pkl"
CALIB_PATH = PROJECT / "models/exo_construction_text_clf_f_calibration.json"


def _default_qc_paths() -> tuple[Path, ...]:
    """优先本批次人标入库产物 03_qc/labeled.csv；其次 Downloads 下的 qc_result。"""
    labeled = PROJECT / "data/runs/exo_construction/machine_0923/03_qc/labeled.csv"
    if labeled.is_file():
        return (labeled,)
    root = Path("/Users/muse/Downloads")
    hits = sorted(root.glob("*建筑施工*qc_result*.csv"))
    if hits:
        return tuple(hits)
    raise FileNotFoundError(
        "未找到人工 QC：请用 --qc-snapshot 指定，或放入 ~/Downloads/建筑施工*qc_result*.csv"
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

# 有人的施工劳作（对齐人工 T 口径：现场有工人动手施工）
KEEP_PROTOTYPES = (
    "construction workers plastering wall with cement third person site labor",
    "bricklayer laying bricks building a house wall step by step",
    "workers pouring concrete slab foundation with mixer on construction site",
    "mason building brick wall with trowel mortar real jobsite",
    "worker installing steel rebar and formwork for column before concrete pour",
    "excavator operator digging foundation trench on construction site",
    "roofers installing roof tiles shingles on a house",
    "tiler laying floor tiles and screed in a house renovation",
    "worker welding steel beam on site steel structure erection",
    "carpenter framing timber house roof truss on site",
    "interior renovation workers installing drywall ceiling plumbing electrical",
    "scaffolding workers building high rise concrete building",
    "工人砌砖 抹灰 浇筑混凝土 施工现场实拍",
    "建筑工地 钢筋 支模 打桩 吊装 作业全过程",
    "装修施工现场 铺贴瓷砖 木工 水电 防水 施工过程",
)

# 非「有人的施工」：设备展示/播客/展会/考试/理论/房产/影视游戏…
DROP_PROTOTYPES = (
    "diecast model toy excavator crane review unboxing by model collector",
    "toy fair walkabout model trucks construction vehicles collectibles",
    "podcast interview talk show about construction business coaching",
    "software demo app review bim autodesk cloud docuSign product overview",
    "engineering exam preparation class question paper certification course",
    "mechanics of materials beam buckling theory lecture no site work",
    "real estate agent house tour client review cost estimate consulting",
    "stock footage motion graphics background loop construction site b roll",
    "construction project management lecture lean takt scheduling no labor",
    "funny fails compilation workers idiotic skills montage",
    "movie trailer film couple dance songs entertainment",
    "gameplay minecraft roblox construction simulator let's play",
    "机械模型 玩具工程车 玩具展会 开箱测评",
    "播客 访谈 口播 课程 考试 精讲班 软件演示",
    "房产中介 看房 报价咨询 建筑理论 无施工画面",
)

# 明确非施工的确定噪声（prototype 模式用；lr 模式仅作参考）
CERTAIN_DROP_RE = re.compile(
    r"podcast|talk show|exam|nppe|pyq|精讲班|一级建造师|二级建造师|"
    r"toy fair|diecast|scale model|model kit|rc model|planet ?coaster|beamng|blender|"
    r"autodesk|docusign|\bbim\b|software|stock footage|premium project|"
    r"mechanics of materials|zener|diode|enterprise architecture|"
    r"real estate|realtor|house tour|client review|"
    r"gameplay|minecraft|roblox|simulator|let.?s play|movie|trailer|"
    r"housefull|chargers|roster|poker|"
    r"玩具|模型|展会|考试|精讲班|播客|访谈|房产中介",
    re.I,
)

# 施工动作（有人的劳作）关键词
SITE_ACTION_RE = re.compile(
    r"plaster|bricklay|lay(?:ing)?\s*bricks?|masonry|mason\b|trowel|"
    r"pour(?:ing)?\s*concrete|concrete\s*(?:pour|casting|slab)|slab casting|"
    r"rebar|reinforce|formwork|shuttering|column|beam|foundation|footing|"
    r"excavat|digging|backhoe|bricklayer|welding|weld\b|steel\s*structure|"
    r"roof(?:ing)?|shingle|waterproof|insulat|"
    r"drywall|gypsum|carpent|timber|framing|joinery|"
    r"plumb|pipefit|hvac|electrical|wiring|conduit|"
    r"tile|tiling|flooring|screed|renovat|remodel|"
    r"scaffold|crane|demolish|paint|render|stucco|"
    r"砌|抹灰|浇筑|支模|钢筋|混凝土|铺贴|粉刷|施工过程|工地|装修|防水|水电|木工|瓦工|泥工",
    re.I,
)

# 救援须命中「劳作动词 + 施工对象」二元短语（器材名词如 backhoe/excavator 单独不触发）
RESCUE_ACTION_RE = re.compile(
    r"(?:lay|laying|laid|install|installing|installed|pour|pouring|poured|build|building|built|"
    r"repair|repairing|fix|fixing|mix|mixing|apply|applying|erect|erecting|assemble|assembling|"
    r"dig|digging|drill|drilling|cut|cutting|weld|welding|tile|tiling|plaster|plastering|"
    r"render|rendering|shutter|shuttering|brick|bricklay|bricklaying|mason|masonry|"
    r"renovat|remodel|demolish|paint|painting|screed|skim|skimming|grout|grouting|"
    r"砌|抹灰|浇筑|支模|钢筋|铺贴|粉刷|施工|装修|防水|水电|木工|瓦工|泥工|安装|铺设|翻新|拆除)"
    r"[\s\w,'\-()/&+]{0,40}?"
    r"(?:wall|walls|roof|roofing|concrete|slab|floor|flooring|tile|tiles|brick|bricks|block|blocks|"
    r"cement|mortar|rebar|steel|pipe|pipes|foundation|footing|beam|column|window|door|drywall|"
    r"panel|insulation|plaster|scaffold|site|house|building|ground|trench|hole|kitchen|bathroom|"
    r"ceiling|stair|porch|deck|fence|driveway|patio|basement|墙|屋顶|混凝土|楼板|地面|瓷砖|砖|"
    r"砌块|水泥|砂浆|钢筋|钢结构|管道|基础|梁|柱|门窗|石膏板|保温|脚手架|工地|房屋|基坑|装修|翻新)",
    re.I,
)

# 救援前置否决：命中即不救（工艺/游戏/玩具/播客/软件/SEO/考试/房产/影视/音乐/器材测评）
RESCUE_BLOCK_RE = re.compile(
    r"gameplay|minecraft|roblox|fortnite|beamng|lego|simulator|planet\s*coaster|"
    r"\bmod\b|modded|game\s*engine|engine\s*architecture|unreal|unity|blender|"
    r"\btoy\b|\btoys\b|toy\s*fair|diecast|scale\s*model|model\s*kit|rc\s*model|"
    r"podcast|talk\s*show|interview|webinar|coaching|"
    r"exam|nppe|\bpyq\b|nata\b|preparation\s*of|online\s*class|"
    r"\bseo\b|marketing|rank\s*and\s*rent|leads?|funnel|ai\s*content|"
    r"real\s*estate|realtor|house\s*tour|\bbnb\b|client\s*review|"
    r"\breviews?\b|\bspecs?\b|buyers?\s*guide|top\s*\d|best\s+\d|"
    r"\bkids?\b|children|nursery|tunes|trivia|auction|surplus|"
    r"trade\s*show|\bexpo\b|fabtech|history\s*of|hoax|"
    r"movie|trailer|anime|song|music|lyrics|official\s*(?:audio|video)|"
    r"tutorial.*(?:code|python|cnn|deep\s*learning|neural)|"
    r"quickbooks|autodesk|\bbim\b|docusign|software|app\s*review|"
    r"\bvs\b|competition|compare|comparison|"
    r"announcement|\bnews\b|election|polls|contest|reveal|haul|unbox|"
    r"playdoh|lumibricks|resume|career|franchise|\bsql\b|azure|openai|crewai|"
    r"玩具|模型|展会|考试|精讲班|播客|访谈|房产中介|软件|游戏|测评|对比|发布|新闻",
    re.I,
)


def should_rescue_labor(title: str) -> bool:
    """窄救援：标题明确写了施工劳作短语、且不含确定噪声词。

    注意：exo_construction 金标上 τ=0.25 本就 T_hurt=0，救援只为兜未见 T；
    故须严格（弱单词 foundation/column 等不再单独触发）。
    """
    s = str(title or "")
    return bool(RESCUE_ACTION_RE.search(s)) and not RESCUE_BLOCK_RE.search(s)


def build_text(row: pd.Series) -> str:
    title = str(row.get("title", "") or "") if pd.notna(row.get("title")) else ""
    return re.sub(r"\s+", " ", title.strip())


def contrast_score(sim_keep: np.ndarray, sim_drop: np.ndarray, *, tau: float = DEFAULT_TAU) -> np.ndarray:
    keep = np.asarray(sim_keep, dtype=float)
    drop = np.asarray(sim_drop, dtype=float)
    tau = max(float(tau), 1e-6)
    return 1.0 / (1.0 + np.exp(-(keep - drop) / tau))


def is_site_action_title(title: str) -> bool:
    return bool(SITE_ACTION_RE.search(str(title or "")))


def is_certain_drop_title(title: str) -> bool:
    return bool(CERTAIN_DROP_RE.search(str(title or "")))


def _normalize_tf_label(raw: str) -> str | None:
    """对齐全仓口径：startswith T/F（兼容 ``T|无声音``）；空/U/其他 → None。"""
    s = str(raw or "").strip().upper()
    if s.startswith("T"):
        return "T"
    if s.startswith("F"):
        return "F"
    return None


_HUMAN_MAP = {"pass": "T", "fail": "F", "t": "T", "f": "F"}


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
            mapped = frame["human_label"].astype(str).str.strip().str.lower().map(_HUMAN_MAP)
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


def fewshot_prototypes(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
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
        print(f"[MiniLMEncoder] device={resolved} batch_size={self.batch_size}", flush=True)

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


class ConstructionPrototypeClf(BaseEstimator, ClassifierMixin):
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
            raise RuntimeError("ConstructionPrototypeClf 尚未 fit")
        X = self.encoder.transform(texts)
        return (X @ self.keep_emb_.T).max(axis=1), (X @ self.drop_emb_.T).max(axis=1)

    def predict_proba(self, texts):
        sim_keep, sim_drop = self._sims(list(texts))
        pos = np.clip(contrast_score(sim_keep, sim_drop, tau=self.tau), 1e-6, 1 - 1e-6)
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
    """宁杀不要漏：t_hurt 率上限内最大化 F 召回。"""
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
    MiniLMEncoder.__module__ = "exo_construction_text_classifier"
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
    ConstructionPrototypeClf.__module__ = "exo_construction_text_classifier"
    MiniLMEncoder.__module__ = "exo_construction_text_classifier"
    sys.modules["exo_construction_text_classifier"] = sys.modules[__name__]


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
        strict = {**threshold_metrics(labels, oof, threshold=0.20),
                  "note": "OOF 无满足 strict 门槛；回退 drop<0.20"}

    recall_aggressive = pick_recall_threshold(labels, oof, max_t_hurt_rate=0.10, min_precision=0.90)
    if recall_aggressive is None:
        recall_aggressive = {**threshold_metrics(labels, oof, threshold=0.25),
                             "note": "OOF 无满足 recall 门槛；回退 drop<0.25"}
    n_t = max(int((labels == "T").sum()), 1)
    recall_aggressive["t_hurt_rate"] = recall_aggressive["t_hurt"] / n_t
    recall_aggressive["t_coverage"] = 1.0 - recall_aggressive["t_hurt_rate"]

    # 默认 apply：对齐全仓口径（先保住 T，再最大化 f_recall）
    OLD_T_HURT_CAP = 0.032
    recall = pick_recall_threshold(labels, oof, max_t_hurt_rate=OLD_T_HURT_CAP, min_precision=0.90)
    if recall is None:
        recall = dict(recall_aggressive)
        recall["note"] = f"无法在 t_hurt≤{OLD_T_HURT_CAP:.1%} 内选阈；回退 recall_aggressive"
    else:
        recall["t_hurt_rate"] = recall["t_hurt"] / n_t
        recall["t_coverage"] = 1.0 - recall["t_hurt_rate"]
        recall["grid"] = f"OOF；保住 T（t_hurt_rate≤{OLD_T_HURT_CAP:.1%}）内最大化 f_recall"
        recall["note"] = "默认 apply；更激进见 recall_aggressive（t_hurt≤10%）"

    t_scores = oof[labels == "T"]
    t_score_quantiles = {}
    if len(t_scores):
        for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
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
        "task": "exo_construction_human_labor",
        "feature_fields": list(FEATURE_FIELDS),
        "n_train": int(len(frame)),
        "n_t": int((labels == "T").sum()),
        "n_f": int((labels == "F").sum()),
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_ap": float(average_precision_score(y, oof)),
        "t_score_quantiles": t_score_quantiles,
        "policy": "文本治理：正类=有人的建筑施工（现场劳作/工序）；默认保住 T 内最大化 f_recall；交付用 t_like（更高τ）",
        "recall": recall,
        "recall_aggressive": recall_aggressive,
        "strict": strict,
        "model_path": str(model_path.resolve()),
        "qc_snapshots": [str(Path(p).resolve()) for p in paths],
        "notes": [
            "监督=人工 T/F；口径=像人标 T（有人的建筑施工劳作），非宽泛「建筑/施工」主题词",
            "仅 title，不含 channel/keyword",
            "标签解析对齐全仓：startswith T/F；同 video_id keep=last",
            "默认 recall：保住 T 优先；交付/收紧用 calibration.t_like（τ越高越像T）",
            "recall_aggressive：t_hurt≤10% 上限内最大化 f_recall（历史对照）",
            "黑名单只做非施工主题闸门，见 docs/exo_construction_text_governance_v1.md",
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
                result["t_like"] = {
                    **old["t_like"],
                    "note": (str(old["t_like"].get("note") or "") + "；重训后请在 keep 上重校准 t_like").strip("；"),
                }
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
    X = encoder.transform(texts)
    tmpl_keep = encoder.transform(list(KEEP_PROTOTYPES))
    tmpl_drop = encoder.transform(list(DROP_PROTOTYPES))
    oof = np.zeros(len(texts), dtype=float)
    keep_idx = np.flatnonzero(keep_mask)
    drop_idx = np.flatnonzero(drop_mask)
    for i in range(len(texts)):
        k_idx = keep_idx[keep_idx != i]
        d_idx = drop_idx[drop_idx != i]
        k_emb = np.vstack([tmpl_keep, X[k_idx]]) if len(k_idx) else tmpl_keep
        d_emb = np.vstack([tmpl_drop, X[d_idx]]) if len(d_idx) else tmpl_drop
        oof[i] = float(contrast_score(
            [float((X[i] @ k_emb.T).max())], [float((X[i] @ d_emb.T).max())], tau=tau)[0])

    strict = pick_strict_threshold(human_labels, oof) or {
        **threshold_metrics(human_labels, oof, threshold=0.25),
        "note": "OOF 无满足 strict 门槛；回退 drop<0.25",
    }

    few_keep, few_drop = fewshot_prototypes(frame)
    clf = ConstructionPrototypeClf(
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
        "method": "construction_labor_prototype_contrast",
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
        "notes": ["legacy 原型对比；默认请用 --method lr", "ml_score 不当交付 KPI"],
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
            paths, model_path=model_path, calibration_path=calibration_path, tau=tau)
    if method != "lr":
        raise ValueError(f"未知 method: {method!r}（可用 lr / prototype）")
    return train_lr_and_calibrate(paths, model_path=model_path, calibration_path=calibration_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="exo_construction MiniLM 文本否决器")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--method", choices=("lr", "prototype"), default="lr",
                        help="lr=人工 T/F + LR（默认）；prototype=原型对比 legacy")
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
        train_and_calibrate(paths, method=args.method, model_path=args.model,
                            calibration_path=args.calibration, tau=args.tau),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

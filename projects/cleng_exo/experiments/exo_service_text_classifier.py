#!/usr/bin/env python3
"""
exo_service 文本语义否决器：title + description 的 TF-IDF + LR。

    y=0: F（文本明显非 PDF 目标 → DROP）
    y=1: U/T（KEEP_FOR_VISUAL；T 不当交付）
ERROR 不进训练。禁止 keyword 进特征。旧 QC 不得当新 F 监督。
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

PROJECT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT / "models/exo_service_text_clf_f.pkl"
CALIB_PATH = PROJECT / "models/exo_service_text_clf_f_calibration.json"
RANDOM_SEED = 42
FEATURE_FIELDS = ("title", "description")


def build_text(row: pd.Series) -> str:
    """只拼 title + description；禁止采集词 keyword。"""
    title = str(row.get("title", "") or "") if pd.notna(row.get("title")) else ""
    desc = str(row.get("description", "") or "") if pd.notna(row.get("description")) else ""
    if len(desc) > 800:
        desc = desc[:800]
    return re.sub(r"\s+", " ", f"{title} {desc}".strip())


def load_training_frame(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path_arg in paths:
        path = Path(path_arg)
        if not path.is_file():
            raise FileNotFoundError(f"缺少 QC 快照: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        required = {"video_id", "title", "qc_text_result"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} 缺少列: {sorted(missing)}")
        frame = frame[frame["qc_text_result"].isin(["T", "F", "U"])].copy()
        frame["label_kind"] = frame["qc_text_result"]
        frame["y"] = (frame["qc_text_result"] != "F").astype(int)
        frame["_qc_round"] = path.stem
        frames.append(frame)
    if not frames:
        raise ValueError("至少需要一个 QC 快照")
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("video_id", keep="first").reset_index(drop=True)


def make_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=12000,
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=2,
            max_features=12000,
            lowercase=True,
            sublinear_tf=True,
        )),
    ])
    return Pipeline([
        ("tfidf", features),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_SEED,
        )),
    ])


def threshold_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict:
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
        "u_hurt_rate": u_hurt / max(n_u, 1),
    }


def pick_strict_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    min_precision: float = 0.95,
    max_u_hurt_rate: float = 0.01,
    min_drop: int = 3,
) -> dict | None:
    """旧安全门（宁漏勿杀）。否决器请用 pick_veto_threshold。"""
    candidates = []
    for threshold in np.round(np.arange(0.01, 0.61, 0.01), 2):
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


def pick_veto_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    min_keep_rate: float = 0.08,
    min_drop: int = 5,
) -> dict | None:
    """宁错杀：最大化 F 召回，同时留一层 KEEP_FOR_VISUAL。"""
    best: dict | None = None
    best_key = None
    for threshold in np.round(np.arange(0.05, 0.95, 0.01), 2):
        row = threshold_metrics(labels, scores, threshold=float(threshold))
        keep_rate = 1.0 - row["drop_coverage"]
        if row["n_drop"] < min_drop or keep_rate < min_keep_rate:
            continue
        key = (row["f_recall"], row["n_drop"], row["drop_precision"])
        if best_key is None or key > best_key:
            best_key = key
            best = {**row, "keep_rate": keep_rate}
    return best


def validate_independent_gate(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    min_precision: float = 0.95,
    max_u_hurt_rate: float = 0.01,
) -> dict:
    metrics = threshold_metrics(labels, scores, threshold=threshold)
    reasons: list[str] = []
    if metrics["drop_precision"] < min_precision:
        reasons.append(f"drop precision {metrics['drop_precision']:.3f} < {min_precision:.3f}")
    if metrics["t_hurt"]:
        reasons.append(f"命中 T={metrics['t_hurt']}")
    if metrics["u_hurt_rate"] > max_u_hurt_rate:
        reasons.append(
            f"U 误伤率 {metrics['u_hurt_rate']:.3f} > {max_u_hurt_rate:.3f}"
        )
    if metrics["n_drop"] == 0:
        reasons.append("独立样本无 drop")
    return {**metrics, "passed": not reasons, "reasons": reasons}


def _oof_scores(texts: list[str], y: np.ndarray) -> np.ndarray:
    minority = min(int((y == 0).sum()), int((y == 1).sum()))
    n_splits = min(5, minority)
    if n_splits < 2:
        raise ValueError("F 与非F 各至少需要 2 条")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in cv.split(texts, y):
        pipe = make_pipeline()
        pipe.fit([texts[i] for i in train_idx], y[train_idx])
        oof[test_idx] = pipe.predict_proba([texts[i] for i in test_idx])[:, 1]
    return oof


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
    oof = _oof_scores(texts, y)
    strict = pick_veto_threshold(labels, oof)
    if strict is None:
        strict = {
            **threshold_metrics(labels, oof, threshold=0.35),
            "note": "OOF 无满足否决门槛的阈值；回退 0.35",
        }

    pipe = make_pipeline()
    pipe.fit(texts, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(pipe, fh)

    rounds = {
        str(name): {
            "n": int(len(group)),
            "F": int((group["label_kind"] == "F").sum()),
            "T": int((group["label_kind"] == "T").sum()),
            "U": int((group["label_kind"] == "U").sum()),
        }
        for name, group in frame.groupby("_qc_round")
    }
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_fields": list(FEATURE_FIELDS),
        "n_train": int(len(frame)),
        "n_f": int((labels == "F").sum()),
        "n_t": int((labels == "T").sum()),
        "n_u": int((labels == "U").sum()),
        "oof_auc_non_f": float(roc_auc_score(y, oof)),
        "oof_ap_non_f": float(average_precision_score(y, oof)),
        "strict": strict,
        "rounds": rounds,
        "model_path": str(model_path),
        "qc_snapshots": [str(Path(p)) for p in paths],
        "notes": [
            "否决器：F=DROP，U/T=KEEP_FOR_VISUAL；T 不当交付",
            "特征只有 title+description，不含 keyword",
            "只使用新否决题标签；旧 QC F 不进训练",
        ],
    }
    calibration_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def score_frame(model_path: Path, frame: pd.DataFrame) -> np.ndarray:
    with model_path.open("rb") as fh:
        pipe = pickle.load(fh)
    texts = frame.apply(build_text, axis=1).tolist()
    return pipe.predict_proba(texts)[:, 1]


def validate_snapshot(
    snapshot: Path,
    *,
    model_path: Path = MODEL_PATH,
    calibration_path: Path = CALIB_PATH,
) -> dict:
    frame = load_training_frame([snapshot])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    threshold = float(calibration["strict"]["drop_threshold"])
    scores = score_frame(model_path, frame)
    return validate_independent_gate(
        frame["label_kind"].to_numpy(dtype=str),
        scores,
        threshold=threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="exo_service 文本语义否决器")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--validate", type=Path, help="独立 QC 快照（须新否决题）")
    parser.add_argument(
        "--qc-snapshot",
        action="append",
        type=Path,
        dest="qc_snapshots",
        help="训练快照（新否决题），可重复",
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--calibration", type=Path, default=CALIB_PATH)
    args = parser.parse_args()

    if args.train:
        if not args.qc_snapshots:
            raise SystemExit("[ERROR] --train 必须 --qc-snapshot（只用新否决题，不用旧 F）")
        paths = tuple(args.qc_snapshots)
        result = train_and_calibrate(
            paths,
            model_path=args.model,
            calibration_path=args.calibration,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.validate:
        result = validate_snapshot(
            args.validate,
            model_path=args.model,
            calibration_path=args.calibration,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["passed"]:
            raise SystemExit(3)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


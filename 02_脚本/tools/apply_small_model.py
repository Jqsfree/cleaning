#!/usr/bin/env python3
"""
tools/apply_small_model.py — 生产侧小模型打分钩子

只自动 drop「高置信负例」；高置信正例 → keep_candidate；中间带 → uncertain（人工池）。
勿用 ml_score 冒充人工合格率。

用法:
  02_脚本/tools/apply_small_model.py input.csv -o $BATCH/06_tools/ \\
    --model models/film_tv_text_clf_svm.pkl
  02_脚本/tools/apply_small_model.py input.csv -o out.csv \\
    --drop-threshold 0.15 --keep-threshold 0.85
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import time
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))


def build_text(row: pd.Series) -> str:
    """与 experiments/film_tv_text_classifier.build_text 对齐（title + keyword）。"""
    t = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
    k = str(row.get("keyword", "")) if pd.notna(row.get("keyword")) else ""
    # 若无 keyword，拼 channel 作弱补充（不破坏已训练特征主路径）
    if not k.strip():
        ch = row.get("channel", row.get("channel_title", ""))
        k = str(ch) if pd.notna(ch) else ""
    k = re.sub(r"(^|\s)-[a-zA-Z0-9*?]+", "", k).strip()
    combined = f"{t} {k}".strip()
    combined = re.sub(r"\s+", " ", combined)
    extra = []
    if re.search(r"\(\s*(19|20)\d{2}\s*\)", t):
        extra.append("FILM_YEAR_TOKEN")
    if re.search(r"\b\d{4}\b", t):
        extra.append("HAS_YEAR_TOKEN")
    if extra:
        combined = combined + " " + " ".join(extra)
    return combined


def score_to_action(
    score: float,
    *,
    drop_threshold: float,
    keep_threshold: float,
) -> str:
    """
    ml_score 视为 P(正类/T)。
    drop: 高置信负例；keep_candidate: 高置信正例；其余 uncertain。
    """
    if score < drop_threshold:
        return "drop"
    if score >= keep_threshold:
        return "keep_candidate"
    return "uncertain"


def _ensure_unpickle_helpers() -> None:
    """pickle 可能引用 experiments 里的 DualTfidfVectorizer。"""
    exp = _REPO_ROOT / "experiments"
    if str(exp) not in sys.path:
        sys.path.insert(0, str(exp))
    try:
        import film_tv_text_classifier as _ftc  # noqa: F401
    except Exception:
        pass


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"模型不存在: {model_path}\n"
            "请先用 experiments/film_tv_text_classifier*.py --train 训练，"
            "或指定已有 --model（如 models/film_tv_text_clf_tfidf.pkl）。"
        )
    _ensure_unpickle_helpers()
    with open(model_path, "rb") as f:
        return pickle.load(f)


def apply_text_model(
    df: pd.DataFrame,
    pipe,
    *,
    drop_threshold: float,
    keep_threshold: float,
) -> pd.DataFrame:
    texts = df.apply(build_text, axis=1).tolist()
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(texts)
        # 取正类列：优先 classes_==1 / 'T'
        pos_idx = 1
        classes = getattr(pipe, "classes_", None)
        if classes is not None:
            for i, c in enumerate(classes):
                if c in (1, "1", "T", True, "pass"):
                    pos_idx = i
                    break
        scores = proba[:, pos_idx]
    elif hasattr(pipe, "decision_function"):
        raw = pipe.decision_function(texts)
        # 粗映射到 (0,1)
        import numpy as np
        scores = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=float)))
    else:
        preds = pipe.predict(texts)
        scores = [1.0 if p in (1, "T", "pass", True) else 0.0 for p in preds]

    out = df.copy()
    out["ml_score"] = scores
    out["ml_pred"] = [
        "T" if s >= 0.5 else "F" for s in out["ml_score"]
    ]
    out["ml_action"] = [
        score_to_action(
            float(s),
            drop_threshold=drop_threshold,
            keep_threshold=keep_threshold,
        )
        for s in out["ml_score"]
    ]
    return out


def resolve_output(path_arg: str, input_path: Path) -> Path:
    p = Path(path_arg)
    if p.suffix.lower() in (".csv", ".parquet", ".pq"):
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    p.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    return p / f"{stem}_ml_scored.csv"


def main() -> None:
    default_model = _REPO_ROOT / "models" / "film_tv_text_clf_svm.pkl"
    p = argparse.ArgumentParser(
        description="小模型打分：仅高置信负例可 auto-drop；中间带回流人工",
    )
    p.add_argument("input", help="CSV/Parquet")
    p.add_argument(
        "-o", "--output", required=True,
        help="输出文件或目录（目录则写 *_ml_scored.csv）",
    )
    p.add_argument(
        "--model", default=str(default_model),
        help=f"pickle 模型路径（默认 {default_model.relative_to(_REPO_ROOT)}）",
    )
    p.add_argument(
        "--modality", choices=("text", "vision"), default="text",
        help="text=文本模型；vision=缩略图（若无权重则报错提示）",
    )
    p.add_argument(
        "--drop-threshold", type=float, default=0.15,
        help="score < 此值 → ml_action=drop（高置信负例，默认 0.15）",
    )
    p.add_argument(
        "--keep-threshold", type=float, default=0.85,
        help="score >= 此值 → keep_candidate（默认 0.85）",
    )
    args = p.parse_args()

    if args.drop_threshold >= args.keep_threshold:
        print("[ERROR] --drop-threshold 必须 < --keep-threshold")
        sys.exit(2)

    inp = Path(args.input)
    if not inp.exists():
        print(f"[ERROR] 文件不存在: {inp}")
        sys.exit(1)

    if args.modality == "vision":
        thumb = _REPO_ROOT / "models" / "film_tv_thumb_clf.pth"
        print(
            f"[ERROR] vision 模态尚未接入生产钩子。"
            f"检测到权重: {'存在' if thumb.exists() else '不存在'} ({thumb})。"
            "请使用 --modality text，或后续扩展缩略图推理。"
        )
        sys.exit(2)

    try:
        pipe = load_model(Path(args.model))
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    ext = inp.suffix.lower()
    df = pd.read_parquet(inp) if ext in (".parquet", ".pq") else pd.read_csv(inp)
    print(f"[{time.strftime('%H:%M:%S')}] 输入 {inp}  ({len(df):,} 行)")
    print(f"  模型: {args.model}")
    print(f"  阈值: drop<{args.drop_threshold}  keep>={args.keep_threshold}")

    t0 = time.perf_counter()
    scored = apply_text_model(
        df, pipe,
        drop_threshold=args.drop_threshold,
        keep_threshold=args.keep_threshold,
    )
    elapsed = time.perf_counter() - t0

    out_path = resolve_output(args.output, inp)
    if out_path.suffix.lower() in (".parquet", ".pq"):
        scored.to_parquet(out_path, index=False)
    else:
        scored.to_csv(out_path, index=False)

    n = len(scored)
    counts = scored["ml_action"].value_counts().to_dict()
    print()
    print("=" * 56)
    print("  小模型打分完成（非人工合格率）")
    print("=" * 56)
    print(f"  行数:           {n:>10,}")
    for action in ("drop", "keep_candidate", "uncertain"):
        c = int(counts.get(action, 0))
        print(f"  {action:16s}{c:>10,}  ({c / max(n, 1) * 100:5.1f}%)")
    print(f"  耗时:           {elapsed:.1f}s")
    print(f"  输出:           {out_path}")
    print("  注: ml_action≠交付 KPI；uncertain 应交人工 QC")
    print("=" * 56)


if __name__ == "__main__":
    main()

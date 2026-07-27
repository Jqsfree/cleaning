"""小模型打分阈值 / action 单测（mock pipe，不依赖真实模型文件）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR / "tools"))

import apply_small_model as asm  # noqa: E402


class _FakePipe:
    """predict_proba → 固定分数列。"""

    classes_ = [0, 1]

    def __init__(self, scores):
        self._scores = scores

    def predict_proba(self, texts):
        import numpy as np
        n = len(texts)
        scores = list(self._scores)
        if len(scores) < n:
            scores = (scores * ((n // len(scores)) + 1))[:n]
        pos = np.asarray(scores[:n], dtype=float)
        neg = 1.0 - pos
        return np.column_stack([neg, pos])


def test_score_to_action_thresholds():
    assert asm.score_to_action(0.10, drop_threshold=0.15, keep_threshold=0.85) == "drop"
    assert asm.score_to_action(0.90, drop_threshold=0.15, keep_threshold=0.85) == "keep_candidate"
    assert asm.score_to_action(0.50, drop_threshold=0.15, keep_threshold=0.85) == "uncertain"


def test_apply_text_model_actions():
    df = pd.DataFrame({
        "video_id": ["a", "b", "c"],
        "title": ["film (2020)", "anime clip", "maybe drama"],
        "keyword": ["k1", "k2", "k3"],
    })
    pipe = _FakePipe([0.05, 0.92, 0.40])
    out = asm.apply_text_model(df, pipe, drop_threshold=0.15, keep_threshold=0.85)
    assert list(out["ml_action"]) == ["drop", "keep_candidate", "uncertain"]
    assert "ml_score" in out.columns and "ml_pred" in out.columns


def test_build_text_uses_title_keyword():
    row = pd.Series({"title": "Hello (2019)", "keyword": "drama -x1"})
    text = asm.build_text(row)
    assert "Hello" in text
    assert "FILM_YEAR_TOKEN" in text or "HAS_YEAR_TOKEN" in text


def test_missing_model_message(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as ei:
        asm.load_model(tmp_path / "no_such.pkl")
    assert "experiments" in str(ei.value) or "训练" in str(ei.value)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("sklearn") is None,
    reason="sklearn not installed",
)
def test_sklearn_available_for_real_models():
    import sklearn  # noqa: F401
    assert True

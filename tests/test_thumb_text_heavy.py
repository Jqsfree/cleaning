"""封面文字启发式特征单测。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR / "tools"))

from core.thumb_text_heavy import (  # noqa: E402
    composite_text_score,
    extract_features_from_bgr,
)


def _smooth_photo(h: int = 360, w: int = 640) -> np.ndarray:
    """偏平滑的实拍风格底图。"""
    y, x = np.mgrid[0:h, 0:w]
    base = (80 + 0.08 * x + 0.05 * y).astype(np.uint8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    img = cv2.GaussianBlur(img, (21, 21), 0)
    cv2.rectangle(img, (40, 40), (220, 320), (90, 110, 130), -1)
    return img


def _poster_cover(h: int = 360, w: int = 640) -> np.ndarray:
    """大色块 + 多行白字标题。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, : w // 2] = (40, 30, 200)
    img[:, w // 2 :] = (220, 180, 240)
    for i, text in enumerate(["BIG TITLE", "SUBTITLE LINE", "MORE TEXT"]):
        y = 60 + i * 70
        cv2.putText(
            img,
            text,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
    cv2.rectangle(img, (20, h - 80), (w - 20, h - 20), (0, 255, 255), -1)
    cv2.putText(
        img,
        "BANNER",
        (40, h - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return img


def test_poster_has_higher_text_signals_than_smooth_photo():
    smooth = extract_features_from_bgr(_smooth_photo())
    poster = extract_features_from_bgr(_poster_cover())
    assert poster["stroke_high_ratio"] > smooth["stroke_high_ratio"]
    assert poster["lap_std"] > smooth["lap_std"]
    assert composite_text_score(poster) > composite_text_score(smooth)


def test_cli_report_only_smoke(tmp_path: Path):
    img_dir = tmp_path / "cache"
    img_dir.mkdir()
    vid = "SYNTHPOSTER"
    cv2.imwrite(str(img_dir / f"{vid}_maxresdefault.jpg"), _poster_cover())

    inp = tmp_path / "in.csv"
    pd.DataFrame({"video_id": [vid], "title": ["demo"]}).to_csv(inp, index=False)
    out = tmp_path / "out"
    script = _SCRIPT_DIR / "tools" / "score_thumb_text_heavy.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            str(inp),
            "-o",
            str(out),
            "--cache-dir",
            str(img_dir),
            "--no-download",
            "--drop-threshold",
            "0.5",
            "--report-only",
        ],
        capture_output=True,
        text=True,
        cwd=str(_SCRIPT_DIR.parent),
    )
    assert proc.returncode == 0, proc.stderr
    assert "would_drop" in proc.stdout or "n_input" in proc.stdout

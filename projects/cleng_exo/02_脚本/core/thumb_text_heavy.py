"""YouTube 缩略图「封面文字过多」启发式特征（无 OCR）。

用于 exo_medical 等品类在 VL 前的 certain-noise 前置过滤。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "edge_density",
    "text_cc_area_ratio",
    "text_cc_count",
    "vivid_ratio",
    "vivid_big_ratio",
    "row_var",
    "col_var",
    "row_peak",
    "lap_mean",
    "lap_std",
    "stroke_mean",
    "stroke_high_ratio",
    "band_stroke_ratio",
    "band_vivid",
    "light_text_cand",
    "white_ratio",
    "border_center_stroke_ratio",
    "hstroke_high",
)

_SHORT_SIDE = 360


def extract_features_from_bgr(im: np.ndarray) -> dict[str, float]:
    """从 BGR 图像提取封面文字启发式特征。"""
    h, w = im.shape[:2]
    scale = _SHORT_SIDE / min(h, w)
    im = cv2.resize(
        im,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )
    h, w = im.shape[:2]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(edges.mean() / 255)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        8,
    )
    nlab, _labels, stats, _ = cv2.connectedComponentsWithStats(thr, 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if nlab > 1 else np.array([])
    img_area = h * w
    if len(areas):
        textlike = areas[(areas >= 20) & (areas <= img_area * 0.02)]
        text_cc_area_ratio = float(textlike.sum() / img_area)
        text_cc_count = int(len(textlike))
    else:
        text_cc_area_ratio = 0.0
        text_cc_count = 0

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    vivid = ((sat > 140) & (val > 80)).astype(np.uint8)
    vivid_ratio = float(vivid.mean())
    n2, _, st2, _ = cv2.connectedComponentsWithStats(vivid * 255, 8)
    vivid_big_ratio = 0.0
    if n2 > 1:
        aa = st2[1:, cv2.CC_STAT_AREA]
        vivid_big_ratio = float(aa[aa > img_area * 0.01].sum() / img_area)

    row_sum = edges.mean(axis=1) / 255
    col_sum = edges.mean(axis=0) / 255
    row_var = float(np.var(row_sum))
    col_var = float(np.var(col_sum))
    row_peak = float((row_sum > row_sum.mean() + row_sum.std()).mean())

    lap = cv2.Laplacian(blur, cv2.CV_32F)
    lap_abs = np.abs(lap)
    lap_mean = float(lap_abs.mean())
    lap_std = float(lap_abs.std())

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    stroke = np.maximum(
        cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel),
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel),
    )
    stroke_mean = float(stroke.mean())
    stroke_high_ratio = float((stroke > 40).mean())

    top = slice(0, h // 5)
    bot = slice(4 * h // 5, h)
    band_stroke_ratio = float(
        max(stroke[top].mean(), stroke[bot].mean()) / max(stroke.mean(), 1e-6)
    )
    band_vivid = float(max(vivid[top].mean(), vivid[bot].mean()))

    whiteish = ((val > 200) & (sat < 60)).astype(np.uint8)
    yellowish = (
        (hsv[:, :, 0] >= 15)
        & (hsv[:, :, 0] <= 40)
        & (sat > 80)
        & (val > 120)
    ).astype(np.uint8)
    light_text_cand = float(((whiteish | yellowish) & (edges > 0)).mean())
    white_ratio = float(whiteish.mean())

    cy0, cy1 = h // 4, 3 * h // 4
    cx0, cx1 = w // 4, 3 * w // 4
    center_stroke = float(stroke[cy0:cy1, cx0:cx1].mean())
    border_mask = np.ones((h, w), bool)
    border_mask[cy0:cy1, cx0:cx1] = False
    border_stroke = float(stroke[border_mask].mean())
    border_center_stroke_ratio = border_stroke / max(center_stroke, 1e-6)

    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    hstroke = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k2)
    hstroke_high = float((hstroke > 35).mean())

    return {
        "edge_density": edge_density,
        "text_cc_area_ratio": text_cc_area_ratio,
        "text_cc_count": float(text_cc_count),
        "vivid_ratio": vivid_ratio,
        "vivid_big_ratio": vivid_big_ratio,
        "row_var": row_var,
        "col_var": col_var,
        "row_peak": row_peak,
        "lap_mean": lap_mean,
        "lap_std": lap_std,
        "stroke_mean": stroke_mean,
        "stroke_high_ratio": stroke_high_ratio,
        "band_stroke_ratio": band_stroke_ratio,
        "band_vivid": band_vivid,
        "light_text_cand": light_text_cand,
        "white_ratio": white_ratio,
        "border_center_stroke_ratio": border_center_stroke_ratio,
        "hstroke_high": hstroke_high,
    }


def extract_features_from_path(path: Path | str) -> dict[str, float] | None:
    im = cv2.imread(str(path))
    if im is None:
        return None
    return extract_features_from_bgr(im)


def extract_features_from_bytes(data: bytes) -> dict[str, float] | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        return None
    return extract_features_from_bgr(im)


def features_to_vector(feats: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(feats[name]) for name in FEATURE_NAMES], dtype=float)


def composite_text_score(feats: dict[str, float]) -> float:
    """未标定前的启发式综合分（越高越像花字封面）。"""
    return float(
        2.0 * feats["stroke_high_ratio"]
        + 1.5 * feats["text_cc_area_ratio"]
        + 1.0 * feats["edge_density"]
        + 0.8 * feats["vivid_big_ratio"]
        + 1.2 * feats["light_text_cand"]
        + 0.5 * feats["row_peak"]
        + 0.3 * (feats["lap_std"] / 20.0)
    )

"""enrich_film_meta 规则与 join 冒烟（不调外网）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_脚本"))

from tools.enrich_film_meta import (  # noqa: E402
    REGION_TO_COUNTRY,
    YT_CATEGORY_ZH,
    classify_region_genre,
    enrich,
    infer_country_from_region,
)


def test_classify_region_hk():
    r, g, src = classify_region_genre("高清修复丨香港邵氏经典", "大嘴電影")
    assert r == "港剧"
    assert src == "title_rule"


def test_classify_genre_crime():
    r, g, src = classify_region_genre("刑侦破案全集", "好剧研究所")
    assert g == "刑侦/犯罪"
    assert src == "title_rule"


def test_classify_empty():
    r, g, src = classify_region_genre("asdf qwerty", "xyz")
    assert src == ""


def test_category_map():
    assert YT_CATEGORY_ZH["1"] == "电影与动画"
    assert YT_CATEGORY_ZH["24"] == "娱乐"


def test_region_to_country_infer():
    assert REGION_TO_COUNTRY["韩剧"] == "KR"
    df = pd.DataFrame(
        {
            "频道国家": ["", "US", ""],
            "地区剧种": ["国产剧", "美剧", ""],
            "yt_meta_status": ["", "ok", ""],
            "country_source": ["", "", ""],
        }
    )
    n = infer_country_from_region(df)
    assert n == 1
    assert df.loc[0, "频道国家"] == "CN"
    assert df.loc[0, "country_source"] == "infer_region"
    assert df.loc[1, "频道国家"] == "US"
    assert df.loc[1, "country_source"] == "yt_api"


def test_join_ref_labels(tmp_path: Path):
    tgt = tmp_path / "tgt.csv"
    ref = tmp_path / "ref.csv"
    out = tmp_path / "out.csv"
    pd.DataFrame(
        [
            {"video_id": "aaa", "title": "国产家庭伦理剧高清", "channel": "A"},
            {"video_id": "bbb", "title": "random", "channel": "B"},
        ]
    ).to_csv(tgt, index=False)
    pd.DataFrame(
        [{"video_id": "aaa", "地区剧种": "国产剧", "剧种": "家庭伦理"}]
    ).to_csv(ref, index=False)

    stats = enrich(str(tgt), str(ref), str(out), skip_yt=True, infer_country=True)
    df = pd.read_csv(out, dtype=str).fillna("")
    assert stats["rows"] == 2
    row_a = df[df.video_id == "aaa"].iloc[0]
    assert row_a["地区剧种"] == "国产剧"
    assert row_a["剧种"] == "家庭伦理"
    assert row_a["label_source"] == "ref_join"
    assert row_a["频道国家"] == "CN"
    assert row_a["country_source"] == "infer_region"
    row_b = df[df.video_id == "bbb"].iloc[0]
    assert row_b["label_source"] in ("yt_only", "title_rule")

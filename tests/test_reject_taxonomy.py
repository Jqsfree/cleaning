"""排除类 registry：映射 / 弃用 / provisional。"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR))

from core.reject_taxonomy import (  # noqa: E402
    get_registry,
    is_provisional,
    load_registry,
    normalize_tag,
    normalize_tags,
    resolve_canonical,
)


def test_load_default_registry():
    reg = get_registry(reload=True)
    assert reg.version >= 1
    assert "anime_cartoon" in reg.tags
    assert reg.is_active("anime_cartoon")


def test_alias_and_chinese():
    reg = get_registry(reload=True)
    assert normalize_tag("AI短剧", reg) == "ai_concept_trailer"
    assert normalize_tag("ai_short_drama", reg) == "ai_concept_trailer"
    assert normalize_tag("动画", reg) == "anime_cartoon"


def test_deprecated_maps_to():
    reg = get_registry(reload=True)
    assert normalize_tag("docu_legacy", reg) == "documentary"
    assert normalize_tag("旧纪录片", reg) == "documentary"
    assert resolve_canonical("docu_legacy", reg) == "documentary"
    assert not reg.is_active("docu_legacy")


def test_provisional_unknown():
    reg = get_registry(reload=True)
    t = normalize_tag("brand_new_weird_class", reg)
    assert t == "provisional:brand_new_weird_class"
    assert is_provisional(t)
    assert normalize_tag("x", reg, allow_provisional=False) is None


def test_normalize_tags_multi():
    tags = normalize_tags("动画, AI短剧 | gaming")
    assert tags[0] == "anime_cartoon"
    assert "ai_concept_trailer" in tags
    assert "gaming" in tags


def test_custom_registry_file(tmp_path: Path):
    p = tmp_path / "reg.toml"
    p.write_text(
        'registry_version = 9\n\n'
        '[[tags]]\n'
        'id = "foo"\n'
        'label_zh = "甲"\n'
        'status = "active"\n'
        'aliases = ["甲类"]\n'
        '[[tags]]\n'
        'id = "bar_old"\n'
        'status = "deprecated"\n'
        'aliases = []\n'
        'maps_to = "foo"\n',
        encoding="utf-8",
    )
    reg = load_registry(p)
    assert reg.version == 9
    assert normalize_tag("甲类", reg) == "foo"
    assert normalize_tag("bar_old", reg) == "foo"

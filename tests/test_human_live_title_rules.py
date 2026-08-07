from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "02_脚本"
sys.path.insert(0, str(_SCRIPT_DIR))

from categories.human_live.cleaner import clean  # noqa: E402
from core.rules_loader import load_blacklist_individual  # noqa: E402


def test_human_live_rules_expose_title_only_section():
    rules_dir = _SCRIPT_DIR / "categories" / "human_live" / "rules"

    rules = load_blacklist_individual(rules_dir)

    assert "title_pass2" in rules
    assert any(
        item["category"] == "audio_slides_screen"
        for item in rules["title_pass2"]
    )
    assert "title_r3" in rules
    assert "channel_pass2" in rules


def test_title_rule_does_not_match_channel_or_keyword(tmp_path):
    source = tmp_path / "input.parquet"
    pd.DataFrame([
        {
            "video_id": "title-hit",
            "title": "Desktop screen recording tutorial",
            "channel": "Person Live",
            "keyword": "",
        },
        {
            "video_id": "channel-only",
            "title": "Real person chatting live",
            "channel": "Screen Recording Archive",
            "keyword": "screen recording",
        },
    ]).to_parquet(source, index=False)

    summary = clean(
        str(source),
        output_dir=str(tmp_path / "out"),
        raw_name="title_test",
        run="run01",
    )

    keep = pd.read_parquet(summary["keep_path"])
    drop = pd.read_parquet(summary["drop_path"])
    assert keep["video_id"].tolist() == ["channel-only"]
    assert drop["video_id"].tolist() == ["title-hit"]
    assert drop["drop_step"].tolist() == ["title_blacklist"]


def test_aggressive_title_and_exact_channel_are_separate(tmp_path):
    source = tmp_path / "input.parquet"
    pd.DataFrame([
        {
            "video_id": "aggressive-title",
            "title": "NFL match highlights",
            "channel": "Person Live",
            "keyword": "",
        },
        {
            "video_id": "all-f-channel",
            "title": "Ordinary upload",
            "channel": "Ray William Johnson",
            "keyword": "",
        },
        {
            "video_id": "channel-name-in-title",
            "title": "Ray William Johnson fan chatting live",
            "channel": "Unrelated Creator",
            "keyword": "",
        },
    ]).to_parquet(source, index=False)

    summary = clean(
        str(source),
        output_dir=str(tmp_path / "out"),
        raw_name="aggressive_test",
        run="run01",
    )

    keep = pd.read_parquet(summary["keep_path"])
    drop = pd.read_parquet(summary["drop_path"]).set_index("video_id")
    assert keep["video_id"].tolist() == ["channel-name-in-title"]
    assert drop.at["aggressive-title", "drop_step"] == "title_aggressive_blacklist"
    assert drop.at["all-f-channel", "drop_step"] == "channel_blacklist"

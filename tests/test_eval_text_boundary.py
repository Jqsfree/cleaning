"""文本边界评估：可写规则闸门 + 连续新鲜样本停点。"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_脚本"))

from core.text_boundary import (  # noqa: E402
    declare_boundary,
    evaluate_sample,
    load_qc_rows,
    propose_rules,
    score_pattern,
)


def _write_qc(path: Path, rows: list[dict]) -> None:
    fields = ["video_id", "title", "channel", "qc_text_result"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_addable_requires_three_f_zero_tu(tmp_path):
    rows = [
        {"video_id": "1", "title": "Minecraft gameplay episode", "channel": "G", "qc_text_result": "F"},
        {"video_id": "2", "title": "Roblox let's play farm", "channel": "G", "qc_text_result": "F"},
        {"video_id": "3", "title": "Fortnite gameplay highlights", "channel": "G", "qc_text_result": "F"},
        {"video_id": "4", "title": "Tomato harvest in the field", "channel": "Farm", "qc_text_result": "T"},
        {"video_id": "5", "title": "Maybe a farm vlog", "channel": "U", "qc_text_result": "U"},
    ]
    s = score_pattern(rows, "gameplay", r"(?i)\b(gameplay|roblox|fortnite|lets?\s*play)\b")
    assert s.n_f == 3 and s.n_t == 0 and s.n_u == 0
    assert s.addable is True

    s2 = score_pattern(rows, "farm", r"(?i)\bfarm\b")
    assert s2.n_t >= 1 or s2.n_u >= 1
    assert s2.addable is False


def test_u_blocks_rule():
    rows = [
        {"video_id": "1", "title": "ASMR haircut salon", "channel": "x", "qc_text_result": "U"},
        {"video_id": "2", "title": "ASMR makeup tapping", "channel": "x", "qc_text_result": "F"},
        {"video_id": "3", "title": "ASMR true crime", "channel": "x", "qc_text_result": "F"},
        {"video_id": "4", "title": "ASMR garden rain", "channel": "x", "qc_text_result": "F"},
    ]
    s = score_pattern(rows, "asmr", r"(?i)\basmr\b")
    assert s.n_f >= 3 and s.n_u == 1
    assert s.addable is False


def test_propose_generic_and_ngram(tmp_path):
    p = tmp_path / "qc.csv"
    _write_qc(p, [
        {"video_id": "1", "title": "Official Music Video Harvest", "channel": "L", "qc_text_result": "F"},
        {"video_id": "2", "title": "Official Trailer Movie 2026", "channel": "L", "qc_text_result": "F"},
        {"video_id": "3", "title": "Lyric video of a song", "channel": "L", "qc_text_result": "F"},
        {"video_id": "4", "title": "Workers picking tomatoes", "channel": "Farm", "qc_text_result": "T"},
        {"video_id": "5", "title": "Farming Simulator 22 bakery", "channel": "Gamer", "qc_text_result": "F"},
        {"video_id": "6", "title": "Farming Simulator 25 harvest", "channel": "Gamer", "qc_text_result": "F"},
        {"video_id": "7", "title": "Let's play Farming Simulator", "channel": "Gamer", "qc_text_result": "F"},
    ])
    rows = load_qc_rows([p])
    proposed = propose_rules(rows, rules_dir=None, min_f=3)
    names = {x.name for x in proposed}
    assert "official_mv_trailer" in names
    assert any("farming" in x.name or "farming simulator" in x.pattern.lower() for x in proposed)


def test_skip_how_to_and_no_mine():
    rows = [
        {"video_id": str(i), "title": f"How to squat {i}", "channel": "Gym", "qc_text_result": "F"}
        for i in range(3)
    ]
    full = evaluate_sample(rows, None, min_f=3, mine=True)
    assert full["n_addable"] >= 1
    skipped = evaluate_sample(
        rows, None, min_f=3, mine=False, skip_names={"how_to_tutorial"}
    )
    assert skipped["n_addable"] == 0
    zero = {"n_addable": 0, "labels": {"F": 10}}
    one = {"n_addable": 2, "labels": {"F": 10}}
    v = declare_boundary([one, zero, zero], consecutive=2)
    assert v["status"] == "text_boundary"
    v2 = declare_boundary([zero, one], consecutive=2)
    assert v2["status"] == "not_boundary"
    v3 = declare_boundary([zero], consecutive=2)
    assert v3["status"] == "not_boundary"


def test_evaluate_sample_empty_blacklist(tmp_path):
    p = tmp_path / "qc.csv"
    _write_qc(p, [
        {"video_id": "1", "title": "Minecraft gameplay", "channel": "G", "qc_text_result": "F"},
        {"video_id": "2", "title": "Minecraft let's play", "channel": "G", "qc_text_result": "F"},
        {"video_id": "3", "title": "Minecraft survival", "channel": "G", "qc_text_result": "F"},
        {"video_id": "4", "title": "Field harvest", "channel": "F", "qc_text_result": "T"},
    ])
    ev = evaluate_sample(load_qc_rows([p]), rules_dir=None)
    assert ev["labels"]["F"] == 3
    assert ev["n_addable"] >= 1
    assert ev["residual_f"]["f_unmatched"] == 3

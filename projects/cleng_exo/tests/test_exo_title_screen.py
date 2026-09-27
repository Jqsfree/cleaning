"""exo title certain-noise rules smoke test."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_脚本"))

from core.rules_loader import load_blacklist
from categories.exo.cleaner import clean
import tempfile
import pandas as pd


def test_exo_blacklist_loads():
    rules = load_blacklist(Path("02_脚本/categories/exo/rules"))
    assert "title_pass2" in rules
    assert rules["title_pass2"] != r"\b\B"
    assert rules["r2"] == r"\b\B"


def test_exo_clean_drops_certain_noise(tmp_path):
    rows = [
        {"video_id": "a1", "title": "Tomato harvesting by hand workers", "channel": "Farm Life", "keyword": "tomato"},
        {"video_id": "a2", "title": "Minecraft Let's Play Episode 12", "channel": "Gamer", "keyword": "farm"},
        {"video_id": "a3", "title": "Official Music Video - Harvest Moon", "channel": "Label", "keyword": "x"},
        {"video_id": "a4", "title": "How to cook spinach soup recipe", "channel": "Chef", "keyword": "spinach"},
        {"video_id": "a5", "title": "Potato digging in the field", "channel": "Kindly Keyin", "keyword": "potato"},
        {"video_id": "a6", "title": "Exclusive interview with farm CEO", "channel": "News", "keyword": "farm"},
        {"video_id": "a7", "title": "Online course: agriculture masterclass", "channel": "Edu", "keyword": "agri"},
        {"video_id": "a8", "title": "Makeup tutorial for summer glow", "channel": "Beauty", "keyword": "x"},
        {"video_id": "a9", "title": "TEDx keynote motivational speech", "channel": "TEDx", "keyword": "x"},
        {"video_id": "a10", "title": "Piano lesson for beginners", "channel": "Music", "keyword": "x"},
        {"video_id": "a11", "title": "Street fight caught on camera", "channel": "Viral", "keyword": "x"},
        {"video_id": "a12", "title": "NBA full match highlights tonight", "channel": "Sports", "keyword": "x"},
    ]
    inp = tmp_path / "in.csv"
    pd.DataFrame(rows).to_csv(inp, index=False)
    out = tmp_path / "out"
    summary = clean(str(inp), stem="t", output_dir=str(out), run="t01")
    keep = pd.read_parquet(summary["keep_path"])
    keep_ids = set(keep["video_id"])
    assert keep_ids == {"a1"}
    assert summary["n_keep"] == 1
    assert summary["n_drop"] == 11


def test_exo_v11_categories_present():
    from core.rules_loader import load_blacklist_individual
    rules = load_blacklist_individual(Path("02_脚本/categories/exo/rules"))
    cats = {r["category"] for r in rules["title_pass2"]}
    for need in (
        "interview_talk",
        "teaching_training",
        "beauty_skincare",
        "business_speech",
        "music_teaching",
        "conflict_fight",
        "sports_compete",
    ):
        assert need in cats


def test_exo_v16_textqc_f_drops(tmp_path):
    """title09×1000 文本 QC F 回流 v1.6：ASMR/街边小吃/菲语教程等 T=0。"""
    from core.rules_loader import load_blacklist_individual
    cats = {r["category"] for r in load_blacklist_individual(Path("02_脚本/categories/exo/rules"))["title_pass2"]}
    for need in ("asmr_noise", "clickbait_misc_soft"):
        assert need in cats

    rows = [
        {"video_id": "keep1", "title": "Harvesting tomatoes by hand in the field", "channel": "Farm", "keyword": "tomato"},
        {"video_id": "b1", "title": "Massive Can Meltdown - ASMR Metal Melting", "channel": "ASMR", "keyword": "x"},
        {"video_id": "b2", "title": "Best Street Food in Bangkok Night Market", "channel": "Food", "keyword": "x"},
        {"video_id": "b3", "title": "PAANO MAGTANIM NG SILI SA BOTE NG SOFTDRINKS", "channel": "Tips", "keyword": "x"},
        {"video_id": "b4", "title": "Easy and Delicious Mini Cheesecakes", "channel": "Bake", "keyword": "x"},
        {"video_id": "b5", "title": "Hits Afrobeat 2023 | Detty December Afro Mix", "channel": "DJ", "keyword": "x"},
        {"video_id": "b6", "title": "Start a New Compost Pile with Newspaper", "channel": "Garden", "keyword": "x"},
        {"video_id": "b7", "title": "Why this Brisket did Over ONE MILLION Views", "channel": "Click", "keyword": "x"},
        {"video_id": "b8", "title": "Combine Demolition Derby: Dixie Deere vs Cow-Bine", "channel": "Show", "keyword": "x"},
    ]
    inp = tmp_path / "in.csv"
    pd.DataFrame(rows).to_csv(inp, index=False)
    out = tmp_path / "out"
    summary = clean(str(inp), stem="t", output_dir=str(out), run="v16")
    keep_ids = set(pd.read_parquet(summary["keep_path"])["video_id"])
    assert keep_ids == {"keep1"}
    assert summary["n_drop"] == 8

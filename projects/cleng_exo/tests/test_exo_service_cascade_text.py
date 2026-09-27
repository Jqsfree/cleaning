"""exo_service cascade L1+L2 unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_脚本"))

from categories.exo_service.cascade_text import (  # noqa: E402
    classify_title_l1_l2,
    load_l1,
    load_l2,
)


def test_l1_l2_rules_load():
    assert len(load_l1()) >= 5
    order, routes = load_l2()
    assert "hair" in order
    assert len(routes) == 10


@pytest.mark.parametrize(
    "title,expect_stage,expect_cat",
    [
        ("My Podcast Interview with a CEO", "l1_drop", "podcast"),
        ("Business tips for startup marketing", "l1_drop", "ceo_business"),
        ("How to start a salon business webinar", "l1_drop", "lecture_conf"),
        ("Panel discussion on hospitality", "l1_drop", "talking_head_panel"),
    ],
)
def test_l1_drops(title, expect_stage, expect_cat):
    out = classify_title_l1_l2(title)
    assert out["stage"] == expect_stage
    assert out["drop_category"] == expect_cat


@pytest.mark.parametrize(
    "title,primary",
    [
        ("Barber haircut fade tutorial day", "hair"),
        ("Nail tech manicure for customer", "beauty"),
        ("Restaurant back of house chef plating", "food_service"),
        ("Cashier stocking shelves retail", "retail"),
            ("Plumber pipe repair in bathroom", "repair"),
        ("Housekeeping cleaning hotel room", "cleaning"),  # cleaning before hospitality if both
        ("Front desk hotel check-in", "hospitality"),
        ("Dental clinic nurse assisting", "healthcare"),
        ("Car wash detailing mechanic bay", "automotive"),
        ("Dog grooming pet salon", "pet_service"),
    ],
)
def test_l2_routes(title, primary):
    out = classify_title_l1_l2(title)
    assert out["stage"] == "candidate"
    assert out["industry_primary"] == primary


def test_unrouted():
    out = classify_title_l1_l2("Random scenery drone flight over mountains")
    assert out["stage"] == "unrouted"

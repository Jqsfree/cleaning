#!/usr/bin/env python3
"""兼容 shim：原 categories.exo_tennis → categories.exo_团队协作.tennis。"""
from categories.exo_团队协作.tennis.cleaner import clean  # noqa: F401

__all__ = ["clean"]

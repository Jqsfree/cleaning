#!/usr/bin/env python3
"""
core/config.py — 配置管理

从 config/datasets.toml 加载数据集清单，
提供 DatasetConfig 和 PipelineConfig 数据类。

用法:
    from dataclean.core.config import load_datasets, get_dataset

    datasets = load_datasets()
    ds = get_dataset("film_tv")
    print(ds.raw_dir, ds.runs, ds.raw_files)
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataclean.core.logging import log


# ── 路径常量 ──

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASETS_CONFIG = PROJECT_ROOT / "config" / "datasets.toml"


# ── 配置数据类 ──

@dataclass
class DatasetConfig:
    """单个数据集的配置。"""
    category: str
    raw_dir: str = ""
    runs: list[str] = field(default_factory=list)
    description: str = ""
    raw_files: list[str] = field(default_factory=list)
    status: str = "active"

    @property
    def raw_path(self) -> Path | None:
        if not self.raw_dir:
            return None
        p = PROJECT_ROOT / self.raw_dir
        return p if p.exists() else None


@dataclass
class PipelineConfig:
    """全局管道配置。"""
    datasets: dict[str, DatasetConfig] = field(default_factory=dict)


# ── 加载函数 ──

def load_datasets(config_path: Path | None = None) -> dict[str, DatasetConfig]:
    """从 datasets.toml 加载所有数据集配置。

    Args:
        config_path: TOML 配置文件路径，None 则使用默认路径

    Returns:
        {category_name: DatasetConfig} 字典
    """
    if config_path is None:
        config_path = DATASETS_CONFIG

    if not config_path.exists():
        log(f"配置文件不存在: {config_path}，返回空配置", level="WARN")
        return {}

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    datasets: dict[str, DatasetConfig] = {}
    for item in raw.get("datasets", []):
        ds = DatasetConfig(
            category=item.get("category", ""),
            raw_dir=item.get("raw_dir", ""),
            runs=item.get("runs", []),
            description=item.get("description", ""),
            raw_files=item.get("raw_files", []),
            status=item.get("status", "active"),
        )
        datasets[ds.category] = ds

    log(f"加载 {len(datasets)} 个数据集配置")
    return datasets


def get_dataset(category: str, config_path: Path | None = None) -> DatasetConfig | None:
    """按类别名查找单个数据集配置。

    Returns:
        DatasetConfig 或 None（类别未注册）
    """
    datasets = load_datasets(config_path)
    return datasets.get(category)


def list_categories(config_path: Path | None = None) -> list[str]:
    """列出所有已注册的数据集类别。"""
    datasets = load_datasets(config_path)
    return sorted(datasets.keys())


# ── 环境变量 ──

def require_env(name: str) -> str:
    """读取必需的环境变量，不存在则退出。"""
    val = os.environ.get(name, "")
    if not val:
        log(f"缺少环境变量: {name}", level="ERROR")
        log(f"  请设置: export {name}=...", level="ERROR")
        import sys
        sys.exit(1)
    return val


def get_env(name: str, default: str = "") -> str:
    """读取可选的环境变量。"""
    return os.environ.get(name, default)

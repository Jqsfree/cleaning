#!/usr/bin/env python3
"""
categories/base.py — Category 插件抽象基类

定义所有 category cleaner/scorer 的统一接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol


class CleanerProtocol(Protocol):
    """clean() 函数的协议（结构化类型，不强制继承）。"""

    def __call__(
        self,
        input_path: str,
        stem: str = "clean",
        output_dir: str = "output",
        raw_name: str = "",
        run: str = "run01",
        **kwargs: Any,
    ) -> dict: ...


class BaseCleaner(ABC):
    """所有 category cleaner 的抽象基类。

    子类必须提供:
      - category: str    类别标识
      - engine: str      引擎标识（用于 summary）

    子类必须实现:
      - clean(input_path, ... , **kwargs) -> dict

    **kwargs 用于类别特有参数（如 keep_score、with_r2 等），
    每个 cleaner 自行从 kwargs 中提取所需参数，
    入口脚本无需知道每个类别的具体参数。
    """

    category: str
    engine: str

    @abstractmethod
    def clean(
        self,
        input_path: str,
        stem: str = "clean",
        output_dir: str = "output",
        raw_name: str = "",
        run: str = "run01",
        **kwargs: Any,
    ) -> dict:
        """执行清洗，返回 summary dict。

        summary 必须包含:
          - engine, category, input, total_rows, total_keep, total_drop
          - retention_pct, elapsed_sec, steps
        可选:
          - total_keep_high, total_keep_medium（打分模式）
        """
        ...


class BaseScorer(ABC):
    """打分 UDF 提供者的抽象基类（打分模式的 category 需要）。

    子类必须实现:
      - register_udfs(conn) -> conn
      - get_thresholds() -> dict
    """

    @abstractmethod
    def register_udfs(self, conn) -> Any:
        """向 DuckDB 连接注册打分 UDF，返回 conn。"""
        ...

    @abstractmethod
    def get_thresholds(self) -> dict:
        """返回阈值 dict: {keep_score, gray_score_low, medium_min_score}。"""
        ...

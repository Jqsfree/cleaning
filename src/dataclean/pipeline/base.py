#!/usr/bin/env python3
"""
pipeline/base.py — Phase 基类 + 共享工具

所有管道阶段的抽象基类，定义统一接口和共享行为。
"""

from __future__ import annotations

import time
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from dataclean.core.logging import log, banner


@dataclass
class PhaseResult:
    """管道阶段的标准化返回值。

    Attributes:
        phase_id: 阶段编号（0-7）
        phase_name: 阶段名称
        total_in: 输入行数
        total_out: 输出行数
        stats: 额外统计指标
        output_files: 产出文件列表
        elapsed_sec: 耗时（秒）
    """
    phase_id: int
    phase_name: str
    total_in: int = 0
    total_out: int = 0
    stats: dict[str, Any] = field(default_factory=dict)
    output_files: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def retention_pct(self) -> float:
        if self.total_in > 0:
            return round(self.total_out / self.total_in * 100, 1)
        return 0.0


class Phase(ABC):
    """管道阶段的抽象基类。

    子类必须实现:
      - phase_id: int         阶段编号
      - phase_name: str       阶段名称
      - run(input_path, output_dir, **kwargs) -> PhaseResult

    用法:
        class NormalizePhase(Phase):
            phase_id = 0
            phase_name = "normalize"

            def run(self, input_path, output_dir, **kwargs):
                ...
                return PhaseResult(...)
    """

    phase_id: int
    phase_name: str

    @abstractmethod
    def run(self, input_path: str, output_dir: str, **kwargs) -> PhaseResult:
        """执行阶段逻辑。子类必须实现。"""
        ...

    def execute(self, input_path: str, output_dir: str, **kwargs) -> PhaseResult:
        """带计时和日志包装的执行入口。

        调用 run() 并自动记录开始/结束、耗时、统计。
        """
        self._log_start(input_path, output_dir, kwargs)
        t0 = time.perf_counter()

        try:
            result = self.run(input_path, output_dir, **kwargs)
            result.elapsed_sec = round(time.perf_counter() - t0, 1)
            result.phase_id = self.phase_id
            result.phase_name = self.phase_name
            self._log_done(result)
            return result
        except Exception:
            elapsed = round(time.perf_counter() - t0, 1)
            log(f"阶段 {self.phase_name} 执行失败 (耗时 {elapsed}s)", level="ERROR")
            raise

    def _log_start(self, input_path: str, output_dir: str, kwargs: dict) -> None:
        log(f"Phase {self.phase_id} — {self.phase_name} 开始")
        log(f"  输入: {os.path.abspath(input_path)}")
        log(f"  输出: {os.path.abspath(output_dir)}/")
        if kwargs:
            extras = {k: v for k, v in kwargs.items() if v is not None}
            if extras:
                log(f"  参数: {extras}")

    def _log_done(self, result: PhaseResult) -> None:
        banner(f"Phase {self.phase_id} — {self.phase_name} 完成")
        if result.total_in:
            print(f"  输入:       {result.total_in:>12,}")
        print(f"  输出:       {result.total_out:>12,}" +
              (f"  ({result.retention_pct}%)" if result.total_in else ""))
        if result.stats:
            for k, v in result.stats.items():
                if isinstance(v, int):
                    print(f"  {k}:  {v:>12,}")
                else:
                    print(f"  {k}:  {v}")
        print(f"  耗时:       {result.elapsed_sec:>11.1f}s")
        if result.output_files:
            print(f"  产物:       {len(result.output_files)} 个文件")
        print("=" * 62)


# ── 共享工具函数 ──


def ensure_input_exists(path: str) -> None:
    """校验输入文件存在，不存在则打印错误并退出。

    这是一个便捷包装，委托给 logging.ensure_input_exists。
    """
    from dataclean.core.logging import ensure_input_exists as _ensure
    _ensure(path)

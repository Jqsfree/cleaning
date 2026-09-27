"""QC 结果 T/F/ERROR 统计（全表累计 + 本轮 pass），供进度条与 monitor 复用。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pandas as pd

QC_RESULT_COL = "qc_vision_result"


@dataclass
class QcTfErrCounts:
    """T / F / ERROR 计数快照。"""
    n_t: int = 0
    n_f: int = 0
    n_err: int = 0
    total: int = 0

    @classmethod
    def from_series(cls, col: pd.Series, total: int | None = None) -> QcTfErrCounts:
        if col is None or len(col) == 0:
            tot = total or 0
            return cls(total=tot)
        n_t = int((col == "T").sum())
        n_f = int((col == "F").sum())
        n_err = int((col == "ERROR").sum())
        return cls(n_t=n_t, n_f=n_f, n_err=n_err, total=total if total is not None else len(col))

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        col: str = QC_RESULT_COL,
        total: int | None = None,
    ) -> QcTfErrCounts:
        tot = total if total is not None else len(df)
        if col not in df.columns:
            return cls(total=tot)
        return cls.from_series(df[col], total=tot)

    @property
    def done(self) -> int:
        return self.n_t + self.n_f + self.n_err

    @property
    def pending(self) -> int:
        return max(0, self.total - self.done)

    def t_rate_pct(self) -> float:
        """T 占已判定（T+F）比例。"""
        ok = self.n_t + self.n_f
        return self.n_t / max(ok, 1) * 100

    def done_pct(self) -> float:
        return self.done / max(self.total, 1) * 100

    def format_compact(self, prefix: str = "") -> str:
        tag = f"{prefix} " if prefix else ""
        return f"{tag}T={self.n_t:,} F={self.n_f:,} E={self.n_err:,}"


@dataclass
class QcStatsBoard:
    """
    全表累计 + 当前 pass 的 T/F/ERROR。
    每处理一条后 sync_global(df)，进度条同时显示本轮与 Σ 全表。
    """
    total_rows: int
    global_counts: QcTfErrCounts = field(default_factory=QcTfErrCounts)
    pass_counts: QcTfErrCounts = field(default_factory=QcTfErrCounts)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.global_counts.total = self.total_rows
        self.pass_counts.total = 0

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, total_rows: int | None = None) -> QcStatsBoard:
        total = total_rows if total_rows is not None else len(df)
        board = cls(total_rows=total)
        board.sync_global(df)
        return board

    def reset_pass(self, pass_total: int = 0) -> None:
        with self._lock:
            self.pass_counts = QcTfErrCounts(total=pass_total)

    def record_pass(self, label: str) -> None:
        with self._lock:
            if label == "T":
                self.pass_counts.n_t += 1
            elif label == "F":
                self.pass_counts.n_f += 1
            elif label == "ERROR":
                self.pass_counts.n_err += 1

    def sync_global(self, df: pd.DataFrame) -> QcTfErrCounts:
        with self._lock:
            self.global_counts = QcTfErrCounts.from_dataframe(
                df, total=self.total_rows,
            )
            return self.global_counts

    def record_and_sync(self, label: str, df: pd.DataFrame) -> None:
        self.record_pass(label)
        self.sync_global(df)

    def tqdm_postfix(self, avg_sec: float | None = None) -> dict:
        """tqdm set_postfix：本轮 + 全表 Σ + 可选均速。"""
        with self._lock:
            p, g = self.pass_counts, self.global_counts
            out = {
                "T": p.n_t,
                "F": p.n_f,
                "E": p.n_err,
                "ΣT": g.n_t,
                "ΣF": g.n_f,
                "ΣE": g.n_err,
            }
        if avg_sec is not None:
            out["s"] = f"{avg_sec:.1f}"
        return out

    def status_suffix(self) -> str:
        with self._lock:
            p, g = self.pass_counts, self.global_counts
        return (
            f"本轮 {p.format_compact()} | 全表 {g.format_compact()} "
            f"({g.done_pct():.1f}% 已QC, 待 {g.pending:,})"
        )

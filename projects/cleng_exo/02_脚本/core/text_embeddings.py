#!/usr/bin/env python3
"""文本编码缓存：MiniLM 向量按「内容寻址」落盘，跨 run / 跨批次复用。

背景：MiniLM 编码是打分链路的唯一重活（百万行 ≈ 20 分钟），而 LR 头是瞬时的。
同一份 title 在 v01→v04 会被反复编码，纯属浪费。本模块把 ``sha1(text) → 向量``
存进 DuckDB，命中即直接读，miss 才调编码器。

用法（一般不直接调用，由 MiniLMEncoder 自动接入）::

    cache = TextEmbeddingCache(dir, encoder_key=..., dim=384)
    vecs  = cache.encode(texts, encoder.encode)   # (n, dim) float32

约定：

- 键 = ``sha1(text.encode('utf-8'))``，与顺序、批次无关；同一 title 只编码一次。
- 存 fp16（体积减半）；读出即转 float32。与全量 fp32 重算的 ``ml_score`` 差异 < 1e-4。
- 缓存与模型权重绑定（``encoder_key`` 记 snapshot 路径 + 维度），换编码器自动换表。
- 关闭方式：``TEXT_EMBED_CACHE=0``，或给编码器 ``cache_dir=None``。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

SCHEMA_VERSION = "1"
_ENV_ENABLE = "TEXT_EMBED_CACHE"
_ENV_DIR = "TEXT_EMBED_CACHE_DIR"

# 单次 SQL 往返的键数：太大 DuckDB 绑参慢，太小往返多
_KEY_CHUNK = 50_000


class CacheUnavailable(RuntimeError):
    """缓存库被其它进程独占（读写+只读都进不去）：调用方应回退到直接编码。"""


def default_cache_dir() -> Path:
    """默认缓存根：``<repo>/data/assets/embeddings/text``（可用环境变量覆盖）。"""
    env = os.environ.get(_ENV_DIR)
    if env:
        return Path(env).expanduser()
    # core/ → 02_脚本/ → 仓库根
    return Path(__file__).resolve().parents[2] / "data/assets/embeddings/text"


def cache_enabled() -> bool:
    return str(os.environ.get(_ENV_ENABLE, "1")).strip().lower() not in {"0", "false", "no", "off"}


def text_key(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def encoder_key(*parts: object) -> str:
    """由编码器身份派生缓存表名（换模型 / 换维度自动隔离）。"""
    raw = "|".join(str(p) for p in parts) + f"|schema={SCHEMA_VERSION}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class TextEmbeddingCache:
    """DuckDB 支撑的内容寻址向量缓存。单进程写；跨进程只读是安全的。"""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        encoder_key: str,
        dim: int,
        dtype: np.dtype | str = np.float16,
        log: Callable[[str], None] | None = None,
    ) -> None:
        import duckdb  # 局部导入：未启用缓存时不拖慢冷启动

        self.root = Path(cache_dir).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.key = encoder_key
        self.dim = int(dim)
        self.dtype = np.dtype(dtype)
        self.path = self.root / f"{self.key}.duckdb"
        self._log = log or (lambda m: None)
        # 读写打开；若另一进程（如预热任务）独占该库，则退化为只读；再不行就放弃缓存。
        # DuckDB 的只读连接不能与别的进程的读写锁共存，所以两种都试一遍。
        self.read_only = False
        try:
            self._con = duckdb.connect(str(self.path))
        except duckdb.IOException:
            if not self.path.exists():
                raise
            try:
                self._con = duckdb.connect(str(self.path), read_only=True)
                self.read_only = True
                self._log(
                    f"[embcache] {self.path.name} 被其它进程占用 → 只读模式"
                    "（命中复用，本次 miss 不写回）"
                )
            except duckdb.IOException as exc_ro:
                raise CacheUnavailable(
                    f"{self.path.name} 被其它进程独占（另一打分/预热任务在跑）；"
                    "并发任务结束后自动恢复复用"
                ) from exc_ro
        self._con.execute("SET threads=4")
        self._con.execute("SET preserve_insertion_order=false")
        self._init_schema()
        self.hits = 0
        self.misses = 0

    # ── schema ──────────────────────────────────────────────────────────────
    def _init_schema(self) -> None:
        self._con.execute("CREATE TABLE IF NOT EXISTS meta(k VARCHAR PRIMARY KEY, v VARCHAR)")
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS emb(h VARCHAR PRIMARY KEY, v BLOB)"
        )
        row = self._con.execute("SELECT v FROM meta WHERE k='dim'").fetchone()
        if row is None:
            self._con.execute(
                "INSERT INTO meta VALUES ('dim', ?), ('dtype', ?), ('schema', ?)",
                [str(self.dim), self.dtype.name, SCHEMA_VERSION],
            )
        elif int(row[0]) != self.dim:
            raise ValueError(
                f"缓存维度不符：{self.path} 存的是 {row[0]}，当前编码器为 {self.dim}；"
                "请换缓存目录或删除该文件"
            )

    # ── 查询 ────────────────────────────────────────────────────────────────
    def _fetch(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        """批量取回已有向量（按 key 分块，避免超长 IN 列表）。"""
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), _KEY_CHUNK):
            part = keys[i : i + _KEY_CHUNK]
            self._con.execute("CREATE OR REPLACE TEMP TABLE q(h VARCHAR)")
            self._con.executemany("INSERT INTO q VALUES (?)", [(k,) for k in part])
            rows = self._con.execute(
                "SELECT q.h, e.v FROM q LEFT JOIN emb e USING(h)"
            ).fetchall()
            for h, blob in rows:
                if blob is not None:
                    found[h] = np.frombuffer(blob, dtype=self.dtype)
        return found

    def _store(self, keys: Sequence[str], vecs: np.ndarray) -> None:
        import pandas as pd

        blobs = [
            np.asarray(vecs[i], dtype=self.dtype).tobytes() for i in range(len(keys))
        ]
        frame = pd.DataFrame({"h": list(keys), "v": blobs})
        self._con.register("_new_emb", frame)
        self._con.execute("INSERT INTO emb SELECT h, v FROM _new_emb")
        self._con.unregister("_new_emb")

    # ── 主入口 ──────────────────────────────────────────────────────────────
    def encode(
        self,
        texts: Iterable[str],
        encode_fn: Callable[[list[str]], np.ndarray],
    ) -> np.ndarray:
        """返回 ``(n, dim) float32``；命中缓存的直接复用，miss 交给 ``encode_fn``。"""
        items = ["" if t is None else str(t) for t in texts]
        n = len(items)
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        keys = [text_key(t) for t in items]
        # 同一批内的重复 title 也只编码一次
        uniq: list[str] = []
        seen: set[str] = set()
        for k in keys:
            if k not in seen:
                seen.add(k)
                uniq.append(k)

        found = self._fetch(uniq)
        missing = [k for k in uniq if k not in found]
        self.hits += len(uniq) - len(missing)
        self.misses += len(missing)

        if missing:
            by_key = {k: t for k, t in zip(keys, items)}
            miss_texts = [by_key[k] for k in missing]
            vecs = np.asarray(encode_fn(miss_texts), dtype=np.float32)
            if vecs.shape != (len(missing), self.dim):
                raise ValueError(
                    f"编码器返回 {vecs.shape}，期望 ({len(missing)}, {self.dim})"
                )
            if not self.read_only:
                self._store(missing, vecs)
            for i, k in enumerate(missing):
                found[k] = np.asarray(vecs[i], dtype=self.dtype)

        out = np.empty((n, self.dim), dtype=np.float32)
        for i, k in enumerate(keys):
            out[i] = found[k]
        return out

    def stats(self) -> dict:
        total = self._con.execute("SELECT COUNT(*) FROM emb").fetchone()[0]
        return {
            "path": str(self.path),
            "table_rows": int(total),
            "session_hits": self.hits,
            "session_misses": self.misses,
            "dim": self.dim,
            "dtype": self.dtype.name,
            "read_only": self.read_only,
        }

    def log_stats(self) -> None:
        s = self.stats()
        tail = "（只读：miss 未写回）" if s["read_only"] else ""
        self._log(
            f"[embcache] 本次命中 {s['session_hits']:,} / miss {s['session_misses']:,}"
            f"；表内累计 {s['table_rows']:,} 条 → {s['path']}{tail}"
        )

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass

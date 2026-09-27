#!/usr/bin/env python3
"""
core/reject_taxonomy.py — 可演进排除类 registry 加载与归一化

Registry 是资产桶命名空间（非真理分类器）：
- active：可作为新提案目标
- deprecated：经 maps_to 接到新 id；历史可读，不删
- 未知字符串 → provisional:<slug>，不堵死积累
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCRIPT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = (
    _SCRIPT_ROOT / "categories" / "_shared" / "reject_registry.toml"
)

_PROVISIONAL_PREFIX = "provisional:"
_SLUG_RE = re.compile(r"[^a-z0-9_]+")


@dataclass
class RejectTag:
    id: str
    label_zh: str = ""
    status: str = "active"  # active | deprecated
    aliases: list[str] = field(default_factory=list)
    maps_to: str = ""


@dataclass
class RejectRegistry:
    version: int
    tags: dict[str, RejectTag]
    alias_index: dict[str, str]  # lower(alias|id|label) -> id
    path: Path | None = None

    def get(self, tag_id: str) -> RejectTag | None:
        return self.tags.get(tag_id)

    def is_active(self, tag_id: str) -> bool:
        t = self.tags.get(tag_id)
        return bool(t and t.status == "active")

    def list_active_ids(self) -> list[str]:
        return sorted(t.id for t in self.tags.values() if t.status == "active")


def _slugify(raw: str) -> str:
    s = raw.strip().lower().replace("-", "_").replace(" ", "_")
    s = _SLUG_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def load_registry(path: str | Path | None = None) -> RejectRegistry:
    """加载 reject_registry.toml。"""
    reg_path = Path(path) if path else DEFAULT_REGISTRY_PATH
    if not reg_path.exists():
        raise FileNotFoundError(f"reject registry 不存在: {reg_path}")

    data = tomllib.loads(reg_path.read_text(encoding="utf-8"))
    version = int(data.get("registry_version", 1))
    tags: dict[str, RejectTag] = {}
    alias_index: dict[str, str] = {}

    for item in data.get("tags", []):
        tid = str(item.get("id", "")).strip()
        if not tid:
            continue
        status = str(item.get("status", "active")).strip().lower()
        if status not in ("active", "deprecated"):
            status = "active"
        aliases = [str(a).strip() for a in item.get("aliases", []) if str(a).strip()]
        tag = RejectTag(
            id=tid,
            label_zh=str(item.get("label_zh", "") or ""),
            status=status,
            aliases=aliases,
            maps_to=str(item.get("maps_to", "") or "").strip(),
        )
        tags[tid] = tag
        for key in (tid, tag.label_zh, *aliases):
            if key:
                alias_index[key.strip().lower()] = tid

    return RejectRegistry(
        version=version,
        tags=tags,
        alias_index=alias_index,
        path=reg_path,
    )


_CACHED: RejectRegistry | None = None


def get_registry(path: str | Path | None = None, *, reload: bool = False) -> RejectRegistry:
    """进程内缓存默认 registry；自定义 path 不缓存。"""
    global _CACHED
    if path is not None:
        return load_registry(path)
    if _CACHED is None or reload:
        _CACHED = load_registry()
    return _CACHED


def resolve_canonical(tag_id: str, registry: RejectRegistry | None = None) -> str:
    """
    将 id 经 maps_to 链解析到最终 canonical（防环，最多 8 跳）。
    未知 id 原样返回。
    """
    reg = registry or get_registry()
    seen: set[str] = set()
    cur = tag_id
    for _ in range(8):
        if cur in seen:
            break
        seen.add(cur)
        t = reg.tags.get(cur)
        if not t or not t.maps_to:
            break
        cur = t.maps_to
    return cur


def normalize_tag(
    raw: Any,
    registry: RejectRegistry | None = None,
    *,
    allow_provisional: bool = True,
) -> str | None:
    """
    任意字符串 → canonical id。
    - 命中 alias/id/label_zh → resolve maps_to
    - 已是 provisional:* → 规范化 slug
    - 未知 → provisional:<slug>（allow_provisional=True）或 None
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None

    reg = registry or get_registry()

    if s.lower().startswith(_PROVISIONAL_PREFIX):
        slug = _slugify(s[len(_PROVISIONAL_PREFIX):])
        return f"{_PROVISIONAL_PREFIX}{slug}"

    key = s.lower()
    if key in reg.alias_index:
        return resolve_canonical(reg.alias_index[key], reg)

    # 直接 id（大小写不敏感）
    for tid in reg.tags:
        if tid.lower() == key:
            return resolve_canonical(tid, reg)

    if allow_provisional:
        return f"{_PROVISIONAL_PREFIX}{_slugify(s)}"
    return None


def normalize_tags(
    raw: Any,
    registry: RejectRegistry | None = None,
    *,
    allow_provisional: bool = True,
) -> list[str]:
    """逗号/分号/竖线分隔的多标签 → 去重后的 canonical 列表（保序）。"""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        parts = [str(x) for x in raw]
    else:
        s = str(raw).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            return []
        parts = re.split(r"[,;|/]+", s)

    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        nid = normalize_tag(p, registry, allow_provisional=allow_provisional)
        if nid and nid not in seen:
            seen.add(nid)
            out.append(nid)
    return out


def is_provisional(tag_id: str) -> bool:
    return tag_id.startswith(_PROVISIONAL_PREFIX)


def validate_tags(
    tags: list[str],
    registry: RejectRegistry | None = None,
) -> dict[str, Any]:
    """返回 active / deprecated_resolved / provisional / unknown 统计。"""
    reg = registry or get_registry()
    active, deprecated, provisional, unknown = [], [], [], []
    for t in tags:
        if is_provisional(t):
            provisional.append(t)
            continue
        info = reg.get(t)
        if not info:
            unknown.append(t)
        elif info.status == "deprecated":
            deprecated.append(t)
        else:
            active.append(t)
    return {
        "active": active,
        "deprecated": deprecated,
        "provisional": provisional,
        "unknown": unknown,
    }

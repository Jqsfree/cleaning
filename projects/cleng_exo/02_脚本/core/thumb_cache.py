"""YouTube 缩略图本地缓存路径解析。"""

from __future__ import annotations

from pathlib import Path

THUMB_SUFFIXES = (
    "maxresdefault",
    "hqdefault",
    "mqdefault",
    "sddefault",
    "0",
)


def resolve_thumbnail_path(
    video_id: str,
    cache_dir: Path | str,
    *,
    min_bytes: int = 1500,
) -> Path | None:
    cache = Path(cache_dir)
    vid = str(video_id).strip()
    if not vid:
        return None
    for suffix in THUMB_SUFFIXES:
        path = cache / f"{vid}_{suffix}.jpg"
        if path.is_file() and path.stat().st_size >= min_bytes:
            return path
    flat = cache / f"{vid}.jpg"
    if flat.is_file() and flat.stat().st_size >= min_bytes:
        return flat
    return None

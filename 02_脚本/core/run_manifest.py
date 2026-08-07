#!/usr/bin/env python3
"""
core/run_manifest.py — 批次索引（快速定位输入/阶段/交付路径）

落在批次根目录 manifest.json：
  data/runs/{category}/{source}_{batch}/manifest.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


MANIFEST_NAME = "manifest.json"


def manifest_path(batch_root: str | Path) -> Path:
    return Path(batch_root) / MANIFEST_NAME


def load_manifest(batch_root: str | Path) -> dict[str, Any]:
    path = manifest_path(batch_root)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"manifest JSON 损坏: {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"manifest 根须为 object: {path}")
    return data


def save_manifest(batch_root: str | Path, data: dict[str, Any]) -> Path:
    root = Path(batch_root)
    root.mkdir(parents=True, exist_ok=True)
    path = manifest_path(root)
    data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def init_manifest(
    batch_root: str | Path,
    *,
    category: str,
    source: str,
    batch: str,
    input_path: str = "",
    notes: str = "",
    reinit: bool = False,
) -> Path:
    """创建批次清单；默认 merge 保留已有 stages / deliver_path。

    reinit=True 时清空 stages（显式重建）。
    """
    source = source.strip().lower()
    if source not in ("human", "machine"):
        raise ValueError("source 必须是 human 或 machine")

    existing: dict[str, Any] = {}
    if not reinit:
        try:
            existing = load_manifest(batch_root)
        except ValueError:
            existing = {}

    if existing and not reinit:
        data = dict(existing)
        data["category"] = category
        data["source"] = source
        data["batch"] = batch
        if input_path:
            data["input"] = input_path
        if notes:
            data["notes"] = notes
        data.setdefault("stages", {})
        data.setdefault("deliver_path", data.get("deliver_path") or "")
        data.setdefault(
            "created_at",
            data.get("created_at") or time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        return save_manifest(batch_root, data)

    data = {
        "category": category,
        "source": source,
        "batch": batch,
        "input": input_path,
        "notes": notes,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stages": {},
        "deliver_path": "",
    }
    return save_manifest(batch_root, data)


def update_stage(
    batch_root: str | Path,
    stage: str,
    *,
    paths: dict[str, str] | None = None,
    stats: dict[str, Any] | None = None,
    deliver_path: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """追加/更新某一 stage；paths/stats/provenance 与已有条目 merge，不整段抹掉。"""
    data = load_manifest(batch_root)
    if not data:
        raise FileNotFoundError(
            f"无 manifest: {manifest_path(batch_root)}；请先 run_manifest.py init"
        )
    stages = data.setdefault("stages", {})
    prev = dict(stages.get(stage) or {})
    entry: dict[str, Any] = {
        **prev,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if paths:
        merged_paths = dict(prev.get("paths") or {})
        merged_paths.update(paths)
        entry["paths"] = merged_paths
    if stats:
        merged_stats = dict(prev.get("stats") or {})
        merged_stats.update(stats)
        entry["stats"] = merged_stats
    if provenance:
        merged_prov = dict(prev.get("provenance") or {})
        merged_prov.update(provenance)
        entry["provenance"] = merged_prov
    stages[stage] = entry
    if deliver_path:
        data["deliver_path"] = deliver_path
    return save_manifest(batch_root, data)


def infer_source_batch(batch_root: str | Path) -> tuple[str, str] | None:
    """从目录名 {source}_{batch} 解析；失败返回 None。"""
    name = Path(batch_root).name
    if "_" not in name:
        return None
    source, batch = name.split("_", 1)
    source = source.strip().lower()
    if source not in ("human", "machine") or not batch:
        return None
    return source, batch


def maybe_update_stage(
    output_path: str | Path,
    stage: str,
    *,
    paths: dict[str, str] | None = None,
    stats: dict[str, Any] | None = None,
    deliver_path: str | None = None,
    category: str | None = None,
    provenance: dict[str, Any] | None = None,
    soft_init: bool = True,
) -> bool:
    """
    从阶段输出路径推断批次根并 update_stage。

    无批次根 → False；无 manifest 时 soft_init 从目录名创建。
    失败不抛（打到调用方可选日志），返回 False。
    """
    from core.batch_layout import infer_batch_root

    root = infer_batch_root(output_path)
    if root is None:
        return False
    try:
        data = load_manifest(root)
    except ValueError:
        return False
    if not data:
        if not soft_init:
            return False
        parsed = infer_source_batch(root)
        if not parsed:
            return False
        source, batch = parsed
        cat = (category or root.parent.name or "unknown").strip() or "unknown"
        try:
            init_manifest(root, category=cat, source=source, batch=batch)
        except Exception:
            return False
    try:
        update_stage(
            root,
            stage,
            paths=paths,
            stats=stats,
            deliver_path=deliver_path,
            provenance=provenance,
        )
    except Exception:
        return False
    return True


def format_summary(data: dict[str, Any]) -> str:
    if not data:
        return "(empty manifest)"
    lines = [
        f"category={data.get('category')}  source={data.get('source')}  batch={data.get('batch')}",
        f"input={data.get('input')}",
        f"deliver={data.get('deliver_path') or '(未登记)'}",
        f"updated_at={data.get('updated_at')}",
    ]
    lot = data.get("lot")
    if isinstance(lot, dict) and lot:
        lines.append(
            "lot: "
            f"frame={lot.get('sample_frame')} size={lot.get('lot_size')} "
            f"method={lot.get('method')} n={lot.get('n')} "
            f"pass_rate={lot.get('pass_rate')} decision={lot.get('decision')}"
        )
    lines.append("stages:")
    for name, meta in (data.get("stages") or {}).items():
        paths = meta.get("paths") or {}
        lines.append(f"  - {name}: {paths or meta}")
    return "\n".join(lines)


def format_paths_only(data: dict[str, Any]) -> str:
    """仅列出各 stage 的文件路径。"""
    if not data:
        return "(empty manifest)"
    lines = [
        f"category={data.get('category')} source={data.get('source')} batch={data.get('batch')}",
    ]
    if data.get("deliver_path"):
        lines.append(f"deliver={data['deliver_path']}")
    for name, meta in (data.get("stages") or {}).items():
        paths = meta.get("paths") or {}
        if not paths:
            continue
        for k, v in paths.items():
            lines.append(f"{name}.{k}={v}")
    return "\n".join(lines)


def iter_manifests(runs_root: str | Path) -> list[dict[str, Any]]:
    """
    扫描 runs_root 下 */*/manifest.json（约定 data/runs/{category}/{source}_{batch}/）。
    返回带 _batch_root 的 dict 列表。
    """
    root = Path(runs_root)
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*/manifest.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not data:
            continue
        data = dict(data)
        data["_batch_root"] = str(path.parent)
        data["_manifest"] = str(path)
        rows.append(data)
    return rows


def find_deliver_paths(
    runs_root: str | Path,
    *,
    category: str,
    batch: str | None = None,
    source: str | None = None,
) -> list[dict[str, str]]:
    """
    按品类（及可选 batch/source）查找交付路径。
    优先 manifest.deliver_path；否则 glob 批次 07_deliver/*。
    """
    root = Path(runs_root)
    cat_dir = root / category
    results: list[dict[str, str]] = []
    if not cat_dir.is_dir():
        return results

    for path in sorted(cat_dir.glob("*/manifest.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        batch_root = path.parent
        src = (data.get("source") or "").lower()
        bat = str(data.get("batch") or "")
        # 目录名兜底：human_0724
        dirname = batch_root.name
        if not src and "_" in dirname:
            src = dirname.split("_", 1)[0].lower()
        if not bat and "_" in dirname:
            bat = dirname.split("_", 1)[1]

        if source and src != source.lower():
            continue
        if batch and bat != batch and batch not in dirname:
            continue

        deliver = (data.get("deliver_path") or "").strip()
        paths: list[str] = []
        if deliver:
            paths.append(deliver)
        else:
            ddir = batch_root / "07_deliver"
            if ddir.is_dir():
                paths.extend(
                    str(p) for p in sorted(ddir.iterdir())
                    if p.is_file() and not p.name.startswith(".")
                )

        for dp in paths:
            results.append({
                "category": category,
                "source": src,
                "batch": bat,
                "batch_root": str(batch_root),
                "deliver_path": dp,
            })
    return results


def format_list_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no manifests)"
    header = f"{'category':<16} {'source':<8} {'batch':<12} {'updated':<20} deliver"
    lines = [header, "-" * len(header)]
    for r in rows:
        deliver = r.get("deliver_path") or ""
        if len(deliver) > 48:
            deliver = "…" + deliver[-47:]
        lines.append(
            f"{str(r.get('category') or ''):<16} "
            f"{str(r.get('source') or ''):<8} "
            f"{str(r.get('batch') or ''):<12} "
            f"{str(r.get('updated_at') or ''):<20} "
            f"{deliver or '(未登记)'}"
        )
    return "\n".join(lines)

#!/usr/bin/env python3
"""
tools/export_reject_assets.py — 排除类资产分层导出

将批次内 proposed / human_validated 按 canonical id 写入:

  data/assets/rejects/{id}/proposed.csv
  data/assets/rejects/{id}/human_validated.csv
  data/assets/rejects/{id}/manifest.json

不进默认 pipeline。训练时 validated 优先，proposed 仅弱监督。

用法:
  02_脚本/tools/export_reject_assets.py --runs-root data/runs
  02_脚本/tools/export_reject_assets.py --batch-root data/runs/film_tv/human_0724/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.reject_taxonomy import (  # noqa: E402
    get_registry,
    normalize_tag,
    normalize_tags,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ASSETS = _REPO_ROOT / "data" / "assets" / "rejects"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, low_memory=False)


def _expand_rows(df: pd.DataFrame, *, layer: str, batch_hint: str = "") -> pd.DataFrame:
    """按 reject_tags 展开为每行一个 canonical tag。"""
    if df.empty or "reject_tags" not in df.columns:
        return pd.DataFrame()

    reg = get_registry()
    rows: list[dict] = []
    id_col = "video_id" if "video_id" in df.columns else None
    if not id_col:
        return pd.DataFrame()

    for _, r in df.iterrows():
        tags = normalize_tags(r.get("reject_tags", ""))
        for tag in tags:
            rows.append({
                "video_id": str(r[id_col]).strip(),
                "reject_tag": tag,
                "layer": layer,
                "modality": r.get("modality", ""),
                "confidence_band": r.get("confidence_band", ""),
                "title": r.get("title", ""),
                "channel": r.get("channel", r.get("channel_title", "")),
                "category": r.get("category", r.get("pipeline_category", "")),
                "batch": r.get("batch", batch_hint),
                "labeled_at": r.get("labeled_at", ""),
                "propose_source": r.get("propose_source", ""),
                "reject_action": r.get("reject_action", ""),
                "registry_version": r.get("registry_version", str(reg.version)),
                "label_source": r.get("label_source", layer),
            })
    return pd.DataFrame(rows)


def _merge_write(path: Path, new_df: pd.DataFrame) -> int:
    """按 video_id+reject_tag+modality 去重合并写入。返回最终行数。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path, dtype=str, low_memory=False)
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = new_df
    if merged.empty:
        merged.to_csv(path, index=False)
        return 0
    subset = ["video_id", "reject_tag"]
    if "modality" in merged.columns:
        subset.append("modality")
    merged = merged.drop_duplicates(subset=subset, keep="last")
    merged.to_csv(path, index=False)
    return len(merged)


def _n_by_modality(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if df.empty or "modality" not in df.columns:
        return {"_": len(df)}
    return {str(k): int(v) for k, v in df["modality"].fillna("").value_counts().items()}


def _update_tag_manifest(
    tag_dir: Path,
    *,
    tag_id: str,
    n_proposed: int,
    n_validated: int,
    sources: list[str],
) -> None:
    reg = get_registry()
    pp = tag_dir / "proposed.csv"
    vv = tag_dir / "human_validated.csv"
    meta = {
        "reject_tag": tag_id,
        "canonical_id": normalize_tag(tag_id, reg) or tag_id,
        "registry_version": reg.version,
        "n_proposed": n_proposed,
        "n_human_validated": n_validated,
        "n_by_modality_proposed": _n_by_modality(pp),
        "n_by_modality_validated": _n_by_modality(vv),
        "sources": sources,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (tag_dir / "manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def collect_from_batch(batch_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    qc = batch_root / "03_qc"
    batch_name = batch_root.name
    proposed = _expand_rows(
        _read_csv(qc / "reject_proposed.csv"),
        layer="proposed",
        batch_hint=batch_name,
    )
    # validated: prefer reject_validated.csv, else train_export / labeled with action
    validated_raw = _read_csv(qc / "reject_validated.csv")
    if validated_raw.empty:
        for name in ("train_export.csv", "labeled.csv"):
            cand = _read_csv(qc / name)
            if not cand.empty and "reject_action" in cand.columns:
                mask = cand["reject_action"].isin(("confirm", "correct"))
                validated_raw = cand.loc[mask]
                break
    validated = _expand_rows(
        validated_raw,
        layer="human_validated",
        batch_hint=batch_name,
    )
    return proposed, validated


def export_assets(
    *,
    batch_roots: list[Path],
    assets_root: Path,
) -> dict[str, dict[str, int]]:
    assets_root.mkdir(parents=True, exist_ok=True)
    all_prop: list[pd.DataFrame] = []
    all_val: list[pd.DataFrame] = []
    sources: list[str] = []

    for root in batch_roots:
        p, v = collect_from_batch(root)
        if not p.empty:
            all_prop.append(p)
        if not v.empty:
            all_val.append(v)
        sources.append(str(root))

    prop_df = pd.concat(all_prop, ignore_index=True) if all_prop else pd.DataFrame()
    val_df = pd.concat(all_val, ignore_index=True) if all_val else pd.DataFrame()

    stats: dict[str, dict[str, int]] = {}
    tag_ids = set()
    if not prop_df.empty:
        tag_ids.update(prop_df["reject_tag"].unique())
    if not val_df.empty:
        tag_ids.update(val_df["reject_tag"].unique())

    for tag in sorted(tag_ids):
        tag_dir = assets_root / tag
        n_p = n_v = 0
        if not prop_df.empty:
            sub = prop_df[prop_df["reject_tag"] == tag]
            if len(sub):
                n_p = _merge_write(tag_dir / "proposed.csv", sub)
        if not val_df.empty:
            sub = val_df[val_df["reject_tag"] == tag]
            if len(sub):
                n_v = _merge_write(tag_dir / "human_validated.csv", sub)
        # recount from files
        pp = tag_dir / "proposed.csv"
        vv = tag_dir / "human_validated.csv"
        n_p = len(pd.read_csv(pp)) if pp.exists() else 0
        n_v = len(pd.read_csv(vv)) if vv.exists() else 0
        _update_tag_manifest(
            tag_dir,
            tag_id=tag,
            n_proposed=n_p,
            n_validated=n_v,
            sources=sources,
        )
        stats[tag] = {"proposed": n_p, "human_validated": n_v}

    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="导出排除类资产 proposed / human_validated")
    p.add_argument("--batch-root", default=None, help="单个批次根目录")
    p.add_argument(
        "--runs-root", default=None,
        help="扫描其下 **/03_qc/reject_proposed.csv 与 reject_validated",
    )
    p.add_argument(
        "--assets-root", default=str(DEFAULT_ASSETS),
        help="资产根目录（默认 data/assets/rejects）",
    )
    args = p.parse_args()

    roots: list[Path] = []
    if args.batch_root:
        roots.append(Path(args.batch_root))
    if args.runs_root:
        runs = Path(args.runs_root)
        for qc in runs.glob("**/03_qc"):
            roots.append(qc.parent)
    roots = sorted({r.resolve() for r in roots})

    if not roots:
        print("[ERROR] 请指定 --batch-root 或 --runs-root")
        sys.exit(2)

    stats = export_assets(batch_roots=roots, assets_root=Path(args.assets_root))
    print()
    print("=" * 56)
    print("  排除类资产导出")
    print("=" * 56)
    print(f"  批次数:     {len(roots)}")
    print(f"  资产根:     {args.assets_root}")
    print(f"  tag 数:     {len(stats)}")
    for tag, st in sorted(stats.items(), key=lambda x: -x[1]["proposed"])[:15]:
        print(
            f"    {tag:40s}  proposed={st['proposed']:>6,}  "
            f"validated={st['human_validated']:>5,}"
        )
    if len(stats) > 15:
        print(f"    … 另有 {len(stats) - 15} 个 tag")
    print("=" * 56)


if __name__ == "__main__":
    main()

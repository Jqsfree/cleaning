#!/usr/bin/env python3
"""
tools/propose_reject_tags.py — 自动提案排除类（非金标）

文本 blacklist / 小模型，或缩略图 vision_thumb → reject_proposed.csv。
列含 modality、confidence_band。不进默认 pipeline。

用法:
  02_脚本/tools/propose_reject_tags.py drop.csv -o $BATCH/ --category film_tv
  02_脚本/tools/propose_reject_tags.py thumb_qc.csv -o $BATCH/ --category film_tv --modality thumb
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.human_qc import detect_video_id_col  # noqa: E402
from core.reject_modality import (  # noqa: E402
    confidence_band_for_ml_score,
    confidence_band_for_rule,
    confidence_band_for_thumb,
    load_cascade_cfg,
    load_modality_map,
    map_vision_thumb_row,
    source_status,
)
from core.reject_taxonomy import get_registry, normalize_tag  # noqa: E402
from core.rules_loader import load_blacklist_individual  # noqa: E402
from core.run_manifest import maybe_update_stage  # noqa: E402

_CATEGORIES_DIR = Path(__file__).resolve().parent.parent / "categories"


def _read_table(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def _compiled_rules(category: str) -> list[tuple[str, re.Pattern]]:
    rules_dir = _CATEGORIES_DIR / category / "rules"
    ind = load_blacklist_individual(rules_dir)
    out: list[tuple[str, re.Pattern]] = []
    for section in ("pass2", "r2"):
        for item in ind.get(section, []):
            cat = item.get("category") or ""
            pat = item.get("pattern") or ""
            if not cat or not pat:
                continue
            try:
                out.append((cat, re.compile(pat, re.IGNORECASE)))
            except re.error:
                continue
    return out


def _text_row(row: pd.Series) -> str:
    parts = []
    for col in ("title", "channel", "channel_title", "keyword", "description"):
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))
    if "title_channel" in row.index and pd.notna(row["title_channel"]):
        parts.append(str(row["title_channel"]))
    return " ".join(parts)


def match_blacklist_category(
    text: str,
    drop_reason: str,
    rules: list[tuple[str, re.Pattern]],
) -> tuple[str | None, str]:
    """返回 (blacklist_category, how)。优先用 drop_reason 定位规则。"""
    reason = (drop_reason or "").strip()
    if reason:
        for cat, cre in rules:
            if cre.search(reason):
                return cat, "drop_reason"
    blob = text or ""
    if blob:
        for cat, cre in rules:
            if cre.search(blob):
                return cat, "title_channel_rematch"
    return None, "unmatched"


def propose_from_frame(
    df: pd.DataFrame,
    *,
    category: str,
    from_ml: bool = False,
    registry_path: str | None = None,
    modality: str = "text",
) -> pd.DataFrame:
    reg = get_registry(registry_path) if registry_path else get_registry()
    cfg = load_cascade_cfg()
    id_col = detect_video_id_col(df.columns) or "video_id"
    if id_col not in df.columns:
        raise ValueError(f"缺少 video_id 列；实际: {list(df.columns)}")

    rules = _compiled_rules(category)
    rows: list[dict] = []

    direct_cols = [
        c for c in ("drop_category", "blacklist_category", "reject_tags", "reject_tag")
        if c in df.columns
    ]

    for _, row in df.iterrows():
        vid = str(row[id_col]).strip()
        if not vid:
            continue

        tags: list[str] = []
        source = ""
        confidence = ""
        band = confidence_band_for_rule(cfg)

        if direct_cols:
            raw = row[direct_cols[0]]
            nid = normalize_tag(raw, reg)
            if nid:
                tags = [nid]
                source = f"column:{direct_cols[0]}"
                confidence = "rule"
                band = confidence_band_for_rule(cfg)

        if not tags and from_ml and "ml_action" in df.columns:
            action = str(row.get("ml_action", "")).strip().lower()
            if action == "drop":
                tags = [normalize_tag("ml_text_drop", reg) or "provisional:ml_text_drop"]
                source = "ml_action"
                try:
                    score = float(row.get("ml_score", 0.5))
                except (TypeError, ValueError):
                    score = 0.5
                confidence = f"ml_score={score}"
                band = confidence_band_for_ml_score(score, cfg)
                if band == "low":
                    continue  # 不像负例，不提案

        if not tags and rules:
            reason = str(row.get("drop_reason", "") or "")
            text = _text_row(row)
            cat, how = match_blacklist_category(text, reason, rules)
            if cat:
                nid = normalize_tag(cat, reg)
                if nid:
                    tags = [nid]
                    source = f"blacklist:{how}"
                    confidence = "rule"
                    band = confidence_band_for_rule(cfg)

        if not tags:
            continue
        if source_status(source, cfg) == "paused":
            continue

        rows.append({
            "video_id": vid,
            "reject_tags": ",".join(tags),
            "propose_source": source,
            "confidence": confidence,
            "confidence_band": band,
            "modality": modality,
            "label_source": "proposed",
            "registry_version": reg.version,
            "pipeline_category": category,
        })

    return pd.DataFrame(rows)


def propose_from_thumb(
    df: pd.DataFrame,
    *,
    category: str,
    registry_path: str | None = None,
) -> pd.DataFrame:
    """vision_thumb F → proposed，modality=thumb。"""
    reg = get_registry(registry_path) if registry_path else get_registry()
    cfg = load_cascade_cfg()
    mmap = load_modality_map()
    if source_status("vision_thumb", cfg) == "paused":
        return pd.DataFrame()

    id_col = detect_video_id_col(df.columns) or "video_id"
    if id_col not in df.columns:
        raise ValueError(f"缺少 video_id 列；实际: {list(df.columns)}")

    band = confidence_band_for_thumb(cfg)
    rows: list[dict] = []
    for _, row in df.iterrows():
        vid = str(row[id_col]).strip()
        if not vid:
            continue
        tag = map_vision_thumb_row(row, modality_map=mmap)
        if not tag:
            continue
        rows.append({
            "video_id": vid,
            "reject_tags": tag,
            "propose_source": "vision_thumb",
            "confidence": "thumb_fail",
            "confidence_band": band,
            "modality": "thumb",
            "label_source": "proposed",
            "registry_version": reg.version,
            "pipeline_category": category,
        })
    return pd.DataFrame(rows)


def merge_proposed(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """按 video_id+modality+reject_tags 去重合并。"""
    if existing is None or existing.empty:
        return new.copy() if new is not None else pd.DataFrame()
    if new is None or new.empty:
        return existing.copy()
    merged = pd.concat([existing, new], ignore_index=True)
    keys = [c for c in ("video_id", "modality", "reject_tags") if c in merged.columns]
    return merged.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)


def write_proposed(batch_root: Path, df: pd.DataFrame, *, merge: bool = True) -> Path:
    qc_dir = batch_root / "03_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    out_path = qc_dir / "reject_proposed.csv"
    if merge and out_path.exists():
        old = pd.read_csv(out_path, dtype=str, low_memory=False)
        df = merge_proposed(old, df)
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="自动提案排除类 → 03_qc/reject_proposed.csv")
    p.add_argument("input", help="drop.csv / scored.csv / thumb_qc.csv")
    p.add_argument("-o", "--batch-root", required=True, help="批次根目录")
    p.add_argument("--category", required=True, help="品类（用于加载 blacklist）")
    p.add_argument(
        "--modality", choices=("text", "thumb"), default="text",
        help="text=blacklist/ml；thumb=vision_thumb 结果",
    )
    p.add_argument(
        "--from-ml", action="store_true",
        help="同时读取 ml_action=drop 作为 provisional 提案",
    )
    p.add_argument("--no-merge", action="store_true", help="覆盖而非合并已有 proposed")
    p.add_argument("--registry", default=None, help="自定义 registry 路径")
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"[ERROR] 不存在: {inp}")
        sys.exit(1)

    batch_root = Path(args.batch_root)
    df = _read_table(str(inp))
    print(f"[{time.strftime('%H:%M:%S')}] 读取 {inp}  ({len(df):,} 行)  modality={args.modality}")

    try:
        if args.modality == "thumb":
            out = propose_from_thumb(
                df, category=args.category, registry_path=args.registry,
            )
        else:
            out = propose_from_frame(
                df,
                category=args.category,
                from_ml=args.from_ml,
                registry_path=args.registry,
                modality="text",
            )
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(2)

    out_path = write_proposed(batch_root, out, merge=not args.no_merge)

    print()
    print("=" * 56)
    print("  排除类自动提案（非金标）")
    print("=" * 56)
    print(f"  输入行:     {len(df):,}")
    print(f"  提案行:     {len(out):,}")
    if len(out) and "modality" in out.columns:
        print(f"  modality:   {out['modality'].value_counts().to_dict()}")
    if len(out):
        top = out["reject_tags"].value_counts().head(8)
        print("  Top tags:")
        for k, v in top.items():
            print(f"    {k:40s}  {v:,}")
    print(f"  输出:       {out_path}")
    print("=" * 56)

    if maybe_update_stage(
        batch_root,
        "reject_proposed",
        paths={"reject_proposed": str(out_path)},
        stats={"n_proposed": len(out), "modality": args.modality, "input": str(inp.resolve())},
    ):
        print("  manifest 已更新 stage=reject_proposed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
tools/reject_source_metrics.py — 按 (tag, modality, source) 统计 overturn / 可信度

用 human_validated 相对 proposed 校准来源，而非相信 raw score。

用法:
  02_脚本/tools/reject_source_metrics.py --assets-root data/assets/rejects
  02_脚本/tools/reject_source_metrics.py --batch-root $BATCH/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.reject_modality import load_cascade_cfg  # noqa: E402
from tools.export_reject_assets import DEFAULT_ASSETS  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent.parent


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, low_memory=False)


def _source_key(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("blacklist"):
        return "blacklist"
    if s.startswith("column:"):
        return "column"
    if s.startswith("fusion"):
        return "fusion"
    if s in ("ml_action", "vision_thumb"):
        return s
    return s or "unknown"


def compute_metrics_from_assets(assets_root: Path) -> dict:
    cfg = load_cascade_cfg()
    min_n = int(cfg.get("metrics", {}).get("min_n_trusted", 30))
    max_ot = float(cfg.get("metrics", {}).get("max_overturn_rate", 0.35))

    # key -> counters
    stats: dict[tuple[str, str, str], dict] = defaultdict(lambda: {
        "n_proposed": 0,
        "n_confirm": 0,
        "n_correct": 0,
        "n_reject_proposal": 0,
        "n_validated": 0,
    })

    deadlock = {
        "n_proposed_total": 0,
        "n_validated_total": 0,
        "tags_with_prop_no_val": [],
    }

    for tag_dir in sorted(assets_root.iterdir()):
        if not tag_dir.is_dir() or tag_dir.name.startswith("_"):
            continue
        tag = tag_dir.name
        prop = _read(tag_dir / "proposed.csv")
        val = _read(tag_dir / "human_validated.csv")
        deadlock["n_proposed_total"] += len(prop)
        deadlock["n_validated_total"] += len(val)
        if len(prop) and not len(val):
            deadlock["tags_with_prop_no_val"].append(tag)

        for _, r in prop.iterrows():
            mod = str(r.get("modality", "") or "")
            src = _source_key(str(r.get("propose_source", "")))
            stats[(tag, mod, src)]["n_proposed"] += 1

        for _, r in val.iterrows():
            mod = str(r.get("modality", "") or "")
            src = _source_key(str(r.get("propose_source", "")))
            # validated 可能无 propose_source：用 reject_action
            action = str(r.get("reject_action", "") or "").lower()
            key = (tag, mod, src if src != "unknown" else "human")
            st = stats[key]
            st["n_validated"] += 1
            if action == "confirm":
                st["n_confirm"] += 1
            elif action == "correct":
                st["n_correct"] += 1
                st["n_reject_proposal"] += 1  # 纠正原提案
            else:
                # 有 validated 行但无 action → 视为 confirm
                st["n_confirm"] += 1

    rows = []
    for (tag, mod, src), st in sorted(stats.items()):
        n_decided = st["n_confirm"] + st["n_correct"]
        overturn = st["n_correct"] + st["n_reject_proposal"]
        # 避免 double count：overturn 用 n_correct
        overturn = st["n_correct"]
        precision = (
            st["n_confirm"] / n_decided if n_decided else None
        )
        overturn_rate = (
            overturn / n_decided if n_decided else None
        )
        if n_decided < min_n:
            trust = "untrusted"
        elif overturn_rate is not None and overturn_rate > max_ot:
            trust = "degraded"
        else:
            trust = "trusted"

        rows.append({
            "reject_tag": tag,
            "modality": mod,
            "propose_source": src,
            "n_proposed": st["n_proposed"],
            "n_sampled_validated": n_decided,
            "n_confirm": st["n_confirm"],
            "n_correct": st["n_correct"],
            "precision_proxy": None if precision is None else round(precision, 4),
            "overturn_rate": None if overturn_rate is None else round(overturn_rate, 4),
            "trust_status": trust,
        })

    # 死局信号
    alerts = []
    if deadlock["n_proposed_total"] > 100 and deadlock["n_validated_total"] == 0:
        alerts.append("proposed_without_validation")
    if len(deadlock["tags_with_prop_no_val"]) > 10:
        alerts.append("many_tags_unvalidated")
    ratio = (
        deadlock["n_validated_total"] / deadlock["n_proposed_total"]
        if deadlock["n_proposed_total"] else 1.0
    )
    if deadlock["n_proposed_total"] > 200 and ratio < 0.01:
        alerts.append("validated_stalled")

    return {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "min_n_trusted": min_n,
        "max_overturn_rate": max_ot,
        "deadlock_alerts": alerts,
        "totals": deadlock,
        "by_source": rows,
    }


def compute_from_batch(batch_root: Path) -> dict:
    """也可直接从批次 03_qc 粗算（无 assets 时）。"""
    qc = batch_root / "03_qc"
    # 临时写到内存结构：构造 mini assets 风格
    # 简化：把 proposed / validated 当单一 tag 展开后走同一逻辑
    from core.reject_taxonomy import normalize_tags

    tmp = batch_root / ".tmp_reject_metrics"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    prop = _read(qc / "reject_proposed.csv")
    val = _read(qc / "reject_validated.csv")
    for _, r in prop.iterrows():
        for tag in normalize_tags(r.get("reject_tags", "")):
            d = tmp / tag
            d.mkdir(exist_ok=True)
            path = d / "proposed.csv"
            row = {
                "video_id": r.get("video_id", ""),
                "reject_tag": tag,
                "modality": r.get("modality", ""),
                "propose_source": r.get("propose_source", ""),
            }
            pd.DataFrame([row]).to_csv(
                path, mode="a", header=not path.exists(), index=False,
            )
    for _, r in val.iterrows():
        for tag in normalize_tags(r.get("reject_tags", "")):
            d = tmp / tag
            d.mkdir(exist_ok=True)
            path = d / "human_validated.csv"
            row = {
                "video_id": r.get("video_id", ""),
                "reject_tag": tag,
                "modality": r.get("modality", ""),
                "propose_source": r.get("propose_source", ""),
                "reject_action": r.get("reject_action", "confirm"),
            }
            pd.DataFrame([row]).to_csv(
                path, mode="a", header=not path.exists(), index=False,
            )

    out = compute_metrics_from_assets(tmp)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="排除来源准确度 / overturn 账本")
    p.add_argument("--assets-root", default=str(DEFAULT_ASSETS))
    p.add_argument("--batch-root", default=None)
    p.add_argument(
        "-o", "--output", default=None,
        help="默认 data/assets/rejects/_metrics/by_source.json",
    )
    args = p.parse_args()

    if args.batch_root:
        report = compute_from_batch(Path(args.batch_root))
    else:
        assets = Path(args.assets_root)
        if not assets.exists():
            print(f"[ERROR] 资产目录不存在: {assets}")
            sys.exit(1)
        report = compute_metrics_from_assets(assets)

    out = Path(args.output) if args.output else Path(args.assets_root) / "_metrics" / "by_source.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print()
    print("=" * 56)
    print("  排除来源准确度")
    print("=" * 56)
    print(f"  输出: {out}")
    print(f"  条目: {len(report.get('by_source', []))}")
    print(f"  告警: {report.get('deadlock_alerts') or '(无)'}")
    trusted = sum(1 for r in report["by_source"] if r["trust_status"] == "trusted")
    untrusted = sum(1 for r in report["by_source"] if r["trust_status"] == "untrusted")
    degraded = sum(1 for r in report["by_source"] if r["trust_status"] == "degraded")
    print(f"  trusted={trusted}  untrusted={untrusted}  degraded={degraded}")
    print("=" * 56)


if __name__ == "__main__":
    main()

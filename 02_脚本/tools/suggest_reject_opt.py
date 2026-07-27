#!/usr/bin/env python3
"""
tools/suggest_reject_opt.py — 排除优化建议（默认只写建议，不改生产配置）

自动优化 ≠ 自动改规则。无 --apply --i-understand 绝不改 cascade toml。

用法:
  02_脚本/tools/suggest_reject_opt.py --metrics data/assets/rejects/_metrics/by_source.json
  02_脚本/tools/suggest_reject_opt.py --assets-root data/assets/rejects -o $BATCH/04_rules/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.reject_modality import DEFAULT_CASCADE, load_cascade_cfg  # noqa: E402
from tools.export_reject_assets import DEFAULT_ASSETS  # noqa: E402
from tools.reject_source_metrics import compute_metrics_from_assets  # noqa: E402


def build_suggestions(report: dict) -> list[dict]:
    cfg = load_cascade_cfg()
    min_prec = float(cfg.get("metrics", {}).get("min_precision_to_relax", 0.85))
    max_ot = float(cfg.get("metrics", {}).get("max_overturn_rate", 0.35))
    min_n = int(cfg.get("metrics", {}).get("min_n_trusted", 30))

    suggestions: list[dict] = []

    for row in report.get("by_source", []):
        tag = row["reject_tag"]
        src = row["propose_source"]
        mod = row["modality"]
        status = row["trust_status"]
        n = row["n_sampled_validated"] or 0
        ot = row.get("overturn_rate")
        prec = row.get("precision_proxy")

        if status == "untrusted":
            suggestions.append({
                "level": "info",
                "action": "collect_more_validation",
                "target": f"{tag}/{mod}/{src}",
                "detail": f"n_sampled={n} < min_n={min_n}；不可收紧自动阈值",
                "shadow": True,
            })
        elif status == "degraded":
            suggestions.append({
                "level": "warn",
                "action": "pause_or_tighten",
                "target": f"{tag}/{mod}/{src}",
                "detail": (
                    f"overturn_rate={ot} > {max_ot}；建议 pause sources.{src} "
                    f"或加严 cascade 阈值（勿自动加规则）"
                ),
                "shadow": True,
            })
        elif (
            status == "trusted"
            and prec is not None
            and prec >= min_prec
            and n >= min_n
        ):
            suggestions.append({
                "level": "info",
                "action": "consider_relax_high_band",
                "target": f"{tag}/{mod}/{src}",
                "detail": (
                    f"precision_proxy={prec}；可考虑略放宽 high 带 "
                    f"（须人工改 reject_cascade.toml，先 shadow）"
                ),
                "shadow": True,
            })

    for alert in report.get("deadlock_alerts", []):
        suggestions.append({
            "level": "critical",
            "action": "deadlock_alert",
            "target": alert,
            "detail": "见 metrics.deadlock_alerts；停止扩大 auto-propose，先补抽样验证",
            "shadow": False,
        })

    if not suggestions:
        suggestions.append({
            "level": "info",
            "action": "none",
            "target": "-",
            "detail": "暂无建议；继续累计 proposed + 抽样 validated",
            "shadow": False,
        })
    return suggestions


def render_md(suggestions: list[dict], report: dict) -> str:
    lines = [
        "# 排除类优化建议（自动生成）",
        "",
        f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 告警: {', '.join(report.get('deadlock_alerts') or ['(无)'])}",
        "",
        "**重要：** 本文件只是建议。自动优化不保证准确；",
        "准确度来自 overturn 校准 + 人工闸门。",
        "禁止：无 validated 自训、用 proposed 当金标扩训（确认偏误）。",
        "",
        "| level | action | target | detail | shadow |",
        "|-------|--------|--------|--------|--------|",
    ]
    for s in suggestions:
        detail = str(s["detail"]).replace("|", "\\|")
        lines.append(
            f"| {s['level']} | {s['action']} | `{s['target']}` | {detail} | {s['shadow']} |"
        )
    lines.append("")
    lines.append("## 应用方式")
    lines.append("")
    lines.append("1. 人工审阅上表")
    lines.append("2. 手动改 `categories/_shared/reject_cascade.toml` 的 sources / 阈值")
    lines.append("3. 或：`suggest_reject_opt.py --apply --i-understand` 仅写入 **shadow 副本**")
    lines.append("")
    return "\n".join(lines)


def write_shadow_cascade(suggestions: list[dict], dest: Path) -> None:
    """复制 cascade toml 到 shadow，并追加注释建议（不改生产文件）。"""
    src = DEFAULT_CASCADE
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    notes = ["", "# --- shadow suggestions (NOT applied to production) ---"]
    for s in suggestions:
        if s["action"] in ("pause_or_tighten", "consider_relax_high_band"):
            notes.append(f"# [{s['level']}] {s['action']}: {s['target']} — {s['detail']}")
    with dest.open("a", encoding="utf-8") as f:
        f.write("\n".join(notes) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="排除优化建议（默认不改配置）")
    p.add_argument("--metrics", default=None, help="by_source.json")
    p.add_argument("--assets-root", default=str(DEFAULT_ASSETS))
    p.add_argument(
        "-o", "--output-dir", default=None,
        help="写 reject_opt_suggestions.md 的目录（默认 assets/_metrics）",
    )
    p.add_argument(
        "--apply", action="store_true",
        help="写入 shadow 副本（仍不改生产 cascade）",
    )
    p.add_argument(
        "--i-understand", action="store_true",
        help="确认了解：不会自动改生产规则/重训",
    )
    args = p.parse_args()

    if args.metrics:
        report = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    else:
        assets = Path(args.assets_root)
        if not assets.exists():
            print(f"[ERROR] {assets} 不存在；先 export / metrics")
            sys.exit(1)
        report = compute_metrics_from_assets(assets)

    suggestions = build_suggestions(report)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.assets_root) / "_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "reject_opt_suggestions.md"
    json_path = out_dir / "reject_opt_suggestions.json"
    md_path.write_text(render_md(suggestions, report), encoding="utf-8")
    json_path.write_text(
        json.dumps({"suggestions": suggestions, "report_totals": report.get("totals")},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"建议已写: {md_path}")
    print(f"JSON:     {json_path}")
    print(f"条数:     {len(suggestions)}")

    if args.apply:
        if not args.i_understand:
            print("[ERROR] --apply 必须同时传 --i-understand（仍只写 shadow，不改生产）")
            sys.exit(2)
        shadow = out_dir / "reject_cascade.shadow.toml"
        write_shadow_cascade(suggestions, shadow)
        print(f"shadow 副本: {shadow}（生产 {DEFAULT_CASCADE} 未修改）")
    else:
        print("未 --apply：生产配置未改动（推荐）")


if __name__ == "__main__":
    main()

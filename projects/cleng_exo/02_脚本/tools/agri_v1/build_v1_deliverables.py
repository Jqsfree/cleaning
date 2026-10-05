#!/usr/bin/env python3
"""Phase 5 §8 交付物汇总：各 Stage 统计 + HALT 清单 + Phase0 结果。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _load_json(path: Path) -> dict | None:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def build_deliverables(batch_root: Path, out_dir: Path) -> dict:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    tools = batch_root / "06_tools"

    stages = {
        "stage_minus1": tools / "stage_minus1_metadata/summary.json",
        "stage0": tools / "stage0_blocklist/summary.json",
        "stage1": tools / "stage1_clip",
        "phase0": tools / "v1_risk_audit/phase0_summary.json",
        "stage2": tools / "stage2_scene/summary.json",
        "stage3": tools / "stage3_interaction/summary.json",
        "stage4": tools / "stage4_vlm/summary.json",
        "stage5_sample": batch_root / "02_sample/v1_stratified_c90",
    }

    report: dict = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}
    halt_items: list[str] = []
    deferred_items: list[str] = []

    policy_path = out_dir / "human_qc_after_pipeline.md"
    report["human_qc_policy"] = {
        "status": "deferred_until_pipeline_complete",
        "note": "Phase0 / interim Stage5 本轮不标；Stage1 完成后 clip_pass c90 → 人工合格率 KPI",
        "doc": str(policy_path.resolve()) if policy_path.is_file() else None,
    }

    for name, path in stages.items():
        if path.is_file():
            report["stages"][name] = _load_json(path)
        elif path.is_dir():
            summaries = sorted(path.glob("*summary.json"))
            clip_sum = sorted(path.glob("*_clip_summary.json"))
            picked = summaries + clip_sum
            report["stages"][name] = (
                _load_json(picked[-1]) if picked else {"status": "dir_empty", "path": str(path)}
            )
        else:
            report["stages"][name] = {"status": "missing", "path": str(path)}

    p0 = report["stages"].get("phase0") or {}
    if isinstance(p0, dict):
        clip = p0.get("clip_reject") or {}
        if clip.get("status") == "pending":
            halt_items.append("Phase0 CLIP reject 审计样本待生成（需 clip_fail 或 pilot）")
        hist = p0.get("historical_t") or {}
        deferred_items.append(
            "Phase0 历史 T / CLIP reject 审计：本轮暂不标，流程结束后再做（见 human_qc_after_pipeline.md）"
        )
        deferred_items.append(
            f"interim Stage5（Stage0 174 条）本轮不用于 KPI；Stage1 完成后对 clip_pass 重新 c90"
        )

    s1 = report["stages"].get("stage1") or {}
    if isinstance(s1, dict) and s1.get("status") == "missing":
        halt_items.append("Stage1 CLIP 未跑完或 embedding/thumbnail 资源不足")

    s3 = report["stages"].get("stage3") or {}
    if isinstance(s3, dict):
        halt_items.append(s3.get("halt_note", "Stage3 IoU/接触阈值须人工校准"))

    report["halt_checklist"] = halt_items
    report["deferred_human_qc"] = deferred_items
    report["embedding_store"] = _load_json(
        Path("data/assets/embeddings/exo_agriculture_0814_semantic_remain/store_manifest.json")
    )

    json_path = out_dir / "v1_deliverables.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# exo 农业 v1 交付物",
        "",
        f"生成时间: {report['created_at']}",
        "",
        "## 各 Stage 通过/拒绝",
        "",
    ]
    for name, data in report["stages"].items():
        md_lines.append(f"### {name}")
        if isinstance(data, dict):
            for k in ("n_in", "n_pass", "n_reject", "n_metadata_pass", "n_clip_pass", "n_clip_fail", "n_run"):
                if k in data:
                    md_lines.append(f"- {k}: {data[k]:,}" if isinstance(data[k], int) else f"- {k}: {data[k]}")
            if data.get("status"):
                md_lines.append(f"- status: {data['status']}")
        md_lines.append("")

    md_lines.extend(["## §5 HALT 待确认", ""])
    for item in halt_items:
        md_lines.append(f"- {item}")
    if deferred_items:
        md_lines.extend(["", "## 人工 QC（延后至本轮机检结束）", ""])
        for item in deferred_items:
            md_lines.append(f"- {item}")
        md_lines.append("")
        md_lines.append("详见 `human_qc_after_pipeline.md`。")

    md_path = out_dir / "v1_deliverables.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    report["outputs"] = {"json": str(json_path), "md": str(md_path)}
    report["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Build v1 deliverables")
    ap.add_argument(
        "--batch-root",
        type=Path,
        default=Path("data/runs/exo_agriculture/machine_0818"),
    )
    ap.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("data/runs/exo_agriculture/machine_0818/06_tools/v1_deliverables"),
    )
    args = ap.parse_args()
    rep = build_deliverables(args.batch_root, args.out_dir)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

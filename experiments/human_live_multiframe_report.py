#!/usr/bin/env python3
"""汇总真人直播多帧 pilot 的 Go/No-Go、成本与全池小时估计。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="真人直播多帧最终报告")
    parser.add_argument("--pilot-dir", required=True)
    parser.add_argument("--benchmark-report", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--v2-summary", required=True)
    parser.add_argument("--v2-acceptance", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    pilot = Path(args.pilot_dir)
    benchmark = _load(args.benchmark_report)
    fetch = _load(pilot / "storyboard_fetch_summary.json")
    feature = _load(pilot / "feature_summary.json")
    v2_summary = _load(args.v2_summary)
    v2_acceptance = _load(args.v2_acceptance)
    pool = pd.read_parquet(args.pool, columns=["video_id", "duration_seconds"])
    pool_hours = float(
        pd.to_numeric(pool["duration_seconds"], errors="coerce").fillna(0).sum()
        / 3600
    )
    index = pd.read_csv(pilot / "storyboard_index.csv")
    attempted = len(index)
    successful = int(index["storyboard_status"].eq("ok").sum())
    aggregate_fetch_sec = float(
        pd.to_numeric(index["elapsed_sec"], errors="coerce").fillna(0).sum()
    )
    avg_fetch_sec = aggregate_fetch_sec / max(attempted, 1)
    workers = 3
    full_fetch_hours = avg_fetch_sec * len(pool) / workers / 3600
    feature_sec_per_video = float(feature["elapsed_sec"]) / max(feature["videos"], 1)
    full_feature_hours = feature_sec_per_video * len(pool) / 3600
    fusion = benchmark["fusion"]
    action = fusion["machine_holdout_actions"]
    projected_keep_hours = pool_hours * float(action["keep_hour_share"])
    goal_hours = 80_000.0
    direct_vlm = benchmark["vlm"]
    go = bool(benchmark["go_full_pool"])

    result = {
        "decision": "go" if go else "no_go",
        "reason": (
            "machine grouped holdout 同时通过全部门槛"
            if go else "machine grouped holdout 未同时通过 AUC、keep 与 drop 支持数门槛"
        ),
        "storyboard": {
            "attempted": attempted,
            "successful": successful,
            "success_rate": successful / max(attempted, 1),
            "avg_fetch_sec_per_video": avg_fetch_sec,
            "full_pool_wall_hours_at_3_workers": full_fetch_hours,
            "paid_api_cost": 0,
        },
        "local_features": {
            **feature,
            "full_pool_gpu_hours_linear_projection": full_feature_hours,
        },
        "models": {
            "rule": benchmark["rule"],
            "direct_vlm": direct_vlm,
            "fusion": fusion,
        },
        "baseline_v2": {
            "keep_hours": v2_summary["stats"]["keep_candidate"]["hours"],
            "human_pass_rate": v2_acceptance["pass_rate"],
            "human_pass_wilson_lower": v2_acceptance["pass_lower"],
            "decision": v2_acceptance["decision"],
        },
        "full_pool": {
            "rows": len(pool),
            "hours": pool_hours,
            "projected_fusion_keep_hours": projected_keep_hours,
            "projection_note": "仅按非代表性 machine holdout 的 keep 小时占比线性外推",
            "goal_hours": goal_hours,
            "goal_reachable_by_projection": projected_keep_hours >= goal_hours,
        },
        "blind_sample": {
            "path": str((pilot / "blind_sample_385.csv").resolve()),
            "rows": 385,
            "status": "prepared_unlabeled",
        },
        "error_clusters": benchmark["clusters"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    action = fusion["machine_holdout_actions"]
    markdown = f"""# 真人直播多帧有效性试验

## 结论

**{result["decision"].upper()}**：{result["reason"]}。

- Storyboard 成功率：{result["storyboard"]["success_rate"]:.1%}（{successful}/{attempted}）
- Fusion machine holdout AUC：{fusion["machine_holdout_auc"]:.3f}
- Keep：n={action["keep_n"]}，precision={action["keep_precision"]:.1%}
- Drop：n={action["drop_n"]}，overturn={action["drop_overturn"]:.1%}
- 全池保留小时启发式估计：{projected_keep_hours:,.0f} / 目标 {goal_hours:,.0f}

## 与 v2 对照

v2 验收通过率 {v2_acceptance["pass_rate"]:.1%}，90% Wilson 下界
{v2_acceptance["pass_lower"]:.1%}，keep {v2_summary["stats"]["keep_candidate"]["hours"]:,.0f}
小时，结论为 {v2_acceptance["decision"]}。

## 成本

- Storyboard 本地抓取无需付费 API；按本次均速和 3 workers 线性外推，
  30 万池约 {full_fetch_hours:.1f} 小时。
- 人体检测 + SigLIP 2 按本次吞吐线性外推约 {full_feature_hours:.1f} GPU 小时。
- Qwen 路径状态：{direct_vlm.get("status")}；
  {direct_vlm.get("reason", direct_vlm.get("note", ""))}。

## 盲样

已生成 385 条、排除全部 pilot 标签、按时长五等分并优先频道去重的盲样；
当前尚未人工标注，不把它作为合格率证据。
"""
    output.write_text(markdown, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

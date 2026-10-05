#!/usr/bin/env python3
"""Phase 0 §6 风险核查：生成人工复核抽样表。

产出（默认 machine_0818/06_tools/v1_risk_audit/）：
  - historical_tf_relabel_sample.csv  历史 T 样本 30~50 条
  - clip_reject_audit_sample.csv      CLIP reject 分层抽样（或 pilot 重放）
  - thumb_lr_audit_report.md          旧 thumb_lr 本机可用性说明
  - phase0_summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).resolve().parent.parent.parent
_REPO = _SCRIPT.parent
sys.path.insert(0, str(_SCRIPT))

from core.agri_v1_schema import audit_sample_columns, merge_agri_v1_context, normalize_agri_v1_frame  # noqa: E402

DEFAULT_QC_DIR = Path("/Users/muse/Downloads/农业_人工")
DEFAULT_OUT = (
    _REPO / "data/runs/exo_agriculture/machine_0818/06_tools/v1_risk_audit"
)
DEFAULT_CLEAN = (
    _REPO
    / "data/runs/exo_agriculture/machine_0818/05_clean/run01"
    / "农业采集_0813-0818_clean_0828.csv"
)
THUMB_LR_GLOB = "**/machine_0814/**/v04_thumb_lr_v2/*thumb_lr_pass*.csv"


def load_historical_qc(qc_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(qc_dir / "*_qc_result.csv")))
    if not files:
        raise FileNotFoundError(f"未找到 qc_result CSV: {qc_dir}")
    parts = [pd.read_csv(f, dtype=str, low_memory=False) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset="video_id", keep="first")
    qc = df["qc_result"].astype(str).str.strip().str.upper()
    df = df[qc == "T"].copy()
    return df


def sample_historical_t(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    n_take = min(n, len(df))
    part = df.sample(n=n_take, random_state=seed).copy()
    part["review_route"] = "historical_t_relabel"
    part["legacy_qc_result"] = "T"
    base = normalize_agri_v1_frame(part)
    merged = merge_agri_v1_context(base, part)
    for col in audit_sample_columns():
        if col not in merged.columns:
            merged[col] = ""
    merged["review_notes"] = "按 v1 §2 填写 human_present / crop_interaction / scene_type"
    return merged[audit_sample_columns()]


def sample_clip_reject(
    *,
    clip_fail_csv: Path | None,
    clean_csv: Path,
    n: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    meta: dict = {"source": None, "n_pool": 0, "n_sample": 0}
    pool: pd.DataFrame | None = None

    if clip_fail_csv and clip_fail_csv.is_file():
        pool = pd.read_csv(clip_fail_csv, dtype=str, low_memory=False)
        meta["source"] = str(clip_fail_csv.resolve())
    else:
        pilot_dir = DEFAULT_OUT / "clip_pilot"
        pilot_fail = sorted(pilot_dir.glob("*_clip_fail_*.csv"))
        if pilot_fail:
            pool = pd.read_csv(pilot_fail[-1], dtype=str, low_memory=False)
            meta["source"] = str(pilot_fail[-1].resolve())
            meta["note"] = "来自本机 pilot CLIP 重放"

    if pool is None or pool.empty:
        meta["status"] = "pending"
        meta["note"] = (
            "无历史 clip_fail 且未跑 pilot；"
            "同步 embedding store 后重跑 cascade，或执行 run_clip_pilot.sh"
        )
        empty = pd.DataFrame(columns=audit_sample_columns())
        return empty, meta

    meta["n_pool"] = len(pool)
    if "keyword" in pool.columns:
        groups = pool.groupby(pool["keyword"].fillna("").astype(str), dropna=False)
        parts: list[pd.DataFrame] = []
        per = max(1, n // max(len(groups), 1))
        for _, sub in groups:
            parts.append(sub.sample(n=min(per, len(sub)), random_state=seed))
        part = pd.concat(parts, ignore_index=True).drop_duplicates("video_id")
        if len(part) < n:
            extra = pool[~pool["video_id"].isin(part["video_id"])].sample(
                n=min(n - len(part), len(pool) - len(part)),
                random_state=seed + 1,
            )
            part = pd.concat([part, extra], ignore_index=True)
    else:
        part = pool.sample(n=min(n, len(pool)), random_state=seed)

    part = part.head(n).copy()
    part["review_route"] = "clip_reject_overturn"
    if "clip_decision" not in part.columns:
        part["clip_decision"] = "clip_fail"
    base = normalize_agri_v1_frame(part)
    merged = merge_agri_v1_context(base, part)
    for col in audit_sample_columns():
        if col not in merged.columns:
            merged[col] = ""
    merged["review_notes"] = "CLIP reject 误杀复核：v1 valid 是否应为 1"
    meta["n_sample"] = len(merged)
    meta["status"] = "ready"
    return merged[audit_sample_columns()], meta


def thumb_lr_report(repo: Path) -> dict:
    matches = sorted(repo.glob(THUMB_LR_GLOB))
    report_path = DEFAULT_OUT / "thumb_lr_audit_report.md"
    lines = [
        "# Phase 0.3 thumb_lr 10.8 万 h 本机核查",
        "",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    if matches:
        lines.append(f"找到 {len(matches)} 个 thumb_lr_pass 文件：")
        for p in matches[:10]:
            lines.append(f"- `{p}`")
        status = "found_local"
    else:
        lines.extend([
            "## 结论：本机无法验证",
            "",
            "未找到 `machine_0814/06_tools/v04_thumb_lr_v2/*thumb_lr_pass.csv`。",
            "该 10.8 万 h 产出**不能**当作 v1 交付结果；需重新过 Stage 2~4。",
            "",
            "若远程有数据，可 rsync 到：",
            "`data/runs/exo_agriculture/machine_0814/06_tools/v04_thumb_lr_v2/`",
        ])
        status = "not_available_local"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": status, "matches": [str(p) for p in matches], "report": str(report_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 0 v1 风险核查抽样表")
    ap.add_argument("--qc-dir", type=Path, default=DEFAULT_QC_DIR)
    ap.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--clean-csv", type=Path, default=DEFAULT_CLEAN)
    ap.add_argument("--clip-fail-csv", type=Path, default=None)
    ap.add_argument("--historical-n", type=int, default=50)
    ap.add_argument("--clip-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    hist_df = load_historical_qc(args.qc_dir)
    hist_sample = sample_historical_t(hist_df, args.historical_n, args.seed)
    hist_path = args.out_dir / "historical_tf_relabel_sample.csv"
    hist_sample.to_csv(hist_path, index=False)

    clip_sample, clip_meta = sample_clip_reject(
        clip_fail_csv=args.clip_fail_csv,
        clean_csv=args.clean_csv,
        n=args.clip_n,
        seed=args.seed,
    )
    clip_path = args.out_dir / "clip_reject_audit_sample.csv"
    clip_sample.to_csv(clip_path, index=False)

    thumb = thumb_lr_report(_REPO)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "historical_t": {
            "n_pool_t": len(hist_df),
            "n_sample": len(hist_sample),
            "path": str(hist_path.resolve()),
            "halt_note": "人工完成标注后计算 v1 一致率；明显偏低则 HALT Stage -1/3",
        },
        "clip_reject": {**clip_meta, "path": str(clip_path.resolve())},
        "thumb_lr": thumb,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    summary_path = args.out_dir / "phase0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

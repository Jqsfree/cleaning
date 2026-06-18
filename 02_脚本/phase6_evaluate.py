#!/usr/bin/env python3
"""
phase6_evaluate.py -- SOP Phase 6: 效果验证 (v2)

评估 Phase 5 完整决策系统（非单独评估 blacklist）。

逻辑:
  1. 加载 audit_sample (video_id → audit_label) 作为 Ground Truth
  2. 加载 Phase 5 clean_all.parquet → KEEP 集
  3. 加载 Phase 5 clean_dropped.parquet → DROP 集
  4. 对每个标注样本，查 video_id 在哪个集合 → 预测 KEEP/DROP
  5. 对比预测 vs Ground Truth → TP/FP/TN/FN
  6. 输出报告 + false_positive.csv + false_negative.csv

用法:
  python3 phase6_evaluate.py \
    --audit data/runs/pingpong/002_audit/audit_sample_v1.parquet \
    --clean-dir data/runs/pingpong/005_clean/run01/ \
    -o data/runs/pingpong/006_eval/
"""

import sys, os, time, argparse, csv
from collections import Counter
from pathlib import Path
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.sop import load_sop, print_banner

COVERAGE_WARN = 0.80


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_ids(parquet_path: str, con: duckdb.DuckDBPyConnection) -> set:
    """加载 parquet 中所有 video_id。文件不存在返回空集。"""
    if not os.path.exists(parquet_path):
        return set()
    rows = con.execute(
        f"SELECT video_id FROM read_parquet('{parquet_path}')"
    ).fetchall()
    return {r[0] for r in rows}


def load_ids_with_reason(parquet_path: str, con) -> dict:
    """加载 parquet 中 video_id → drop_reason (normalized)。"""
    if not os.path.exists(parquet_path):
        return {}
    result = {}
    rows = con.execute(
        f"SELECT video_id, drop_reason FROM read_parquet('{parquet_path}')"
    ).fetchall()
    for vid, reason in rows:
        r = (reason or "").strip()
        if not r:
            r = "unknown"
        # Normalize reasons into buckets
        if "blacklist" in r:
            bucket = "blacklist"
        elif "no_signal" in r or "no_align" in r:
            bucket = "no_signal"
        elif "playlist" in r:
            bucket = "playlist_filter"
        elif r in ("high_score", "gray_aligned"):
            bucket = "score_keep"
        elif r == "medium_strong_signal":
            bucket = "medium_recovery"
        elif "score" in r or "default_drop" in r or "low_score" in r or "aligned_low" in r or "medium_" in r:
            bucket = "score_threshold"
        else:
            bucket = r[:40] if r else "unknown"
        result[vid] = bucket
    return result


def load_labels(audit_path: str, con) -> dict:
    """加载 audit_sample → {video_id: {label, title, keyword, channel}}。"""
    if not os.path.exists(audit_path):
        return {}
    ext = Path(audit_path).suffix.lower()
    reader = "read_parquet" if ext == ".parquet" else "read_csv_auto"
    rows = con.execute(f"""
        SELECT video_id, audit_label, title, keyword, channel
        FROM {reader}('{audit_path}')
        WHERE video_id IS NOT NULL AND audit_label IS NOT NULL
    """).fetchall()
    return {
        r[0]: {"label": str(r[1]).strip().upper(),
               "title": r[2] or "", "keyword": r[3] or "", "channel": r[4] or ""}
        for r in rows
    }


def main():
    print_banner(6)

    parser = argparse.ArgumentParser(description="SOP Phase 6: 效果验证 v2")
    parser.add_argument("--audit", required=True,
                        help="标注样本 (audit_sample_v1.parquet)")
    parser.add_argument("--clean-dir", required=True,
                        help="Phase 5 输出目录 (含 {base}_{run}_keep.parquet, {base}_{run}_drop.parquet)")
    parser.add_argument("-o", "--output-dir", default="data/runs/006_eval",
                        help="输出目录")
    args = parser.parse_args()

    if not os.path.exists(args.audit):
        print(f"[ERROR] audit 文件不存在: {args.audit}")
        sys.exit(1)

    # Auto-detect: try new naming first, fallback to old
    import glob as _glob
    keep_files = _glob.glob(os.path.join(args.clean_dir, "*_keep.parquet"))
    drop_files = _glob.glob(os.path.join(args.clean_dir, "*_drop.parquet"))
    clean_all_path = keep_files[0] if keep_files else os.path.join(args.clean_dir, "clean_all.parquet")
    clean_drop_path = drop_files[0] if drop_files else os.path.join(args.clean_dir, "clean_dropped.parquet")
    if not os.path.exists(clean_all_path) and not os.path.exists(clean_drop_path):
        print(f"[ERROR] Phase 5 产物不存在于: {args.clean_dir}")
        sys.exit(1)

    out_dir = args.output_dir.rstrip("/")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    con = duckdb.connect()

    # 1. Load Phase 5 outputs
    log("加载 Phase 5 产物...")
    keep_ids = load_ids(clean_all_path, con)
    drop_ids = load_ids(clean_drop_path, con)
    keep_reasons = load_ids_with_reason(clean_all_path, con)
    drop_reasons = load_ids_with_reason(clean_drop_path, con)
    n_keep = len(keep_ids)
    n_drop = len(drop_ids)
    n_total_phase5 = n_keep + n_drop

    # Run statistics
    keep_rate = n_keep / max(n_total_phase5, 1) * 100
    drop_rate = n_drop / max(n_total_phase5, 1) * 100

    # 2. Load Ground Truth
    log("加载 Ground Truth...")
    labels = load_labels(args.audit, con)
    n_audit = len(labels)

    con.close()

    # 3. Match & classify
    log("匹配 & 分类...")
    tp = fp = tn = fn = 0
    skipped = 0
    fp_rows = []  # (video_id, title, reason)
    fn_rows = []  # (video_id, title, reason)
    skipped_rows = []

    fp_reasons = Counter()
    fn_reasons = Counter()

    # Determine KEEP reason from Phase 5: we don't have explicit keep reasons stored
    # Use a simple approach: if score >= threshold or entity matched
    # For now, bucket as "scoring_pass" since Phase 5 doesn't store keep reasons separately

    for vid, info in labels.items():
        gt = info["label"]
        if gt not in ("T", "F"):
            skipped += 1
            skipped_rows.append((vid, info["title"], f"invalid_label:{gt}"))
            continue

        in_keep = vid in keep_ids
        in_drop = vid in drop_ids

        if in_keep and not in_drop:
            pred = "KEEP"
            reason = keep_reasons.get(vid, "score_keep")
        elif in_drop and not in_keep:
            pred = "DROP"
            reason = drop_reasons.get(vid, "unknown")
        elif in_keep and in_drop:
            # Shouldn't happen, but pick DROP as conservative
            pred = "DROP"
            reason = drop_reasons.get(vid, "unknown")
        else:
            # Not in Phase 5 output → skipped (filtered out by Phase 0 or not processed)
            skipped += 1
            skipped_rows.append((vid, info["title"], "not_in_phase5_output"))
            continue

        if pred == "KEEP" and gt == "T":
            tp += 1
        elif pred == "KEEP" and gt == "F":
            fp += 1
            fp_rows.append((vid, info["title"], reason))
            fp_reasons[reason] += 1
        elif pred == "DROP" and gt == "F":
            tn += 1
        elif pred == "DROP" and gt == "T":
            fn += 1
            fn_rows.append((vid, info["title"], reason))
            fn_reasons[reason] += 1

    matched = tp + fp + tn + fn
    total = matched + skipped
    coverage = matched / max(total, 1) * 100
    precision = tp / max(tp + fp, 1) * 100
    recall = tp / max(tp + fn, 1) * 100
    f1 = 2 * precision * recall / max(precision + recall, 1)
    accuracy = (tp + tn) / max(matched, 1) * 100
    retention = (tp + fp) / max(matched, 1) * 100

    # 4. Output files
    log("写出结果...")
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    # false_positive.csv
    fp_csv = os.path.join(out_dir, "false_positive.csv")
    with open(fp_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "title", "decision_reason"])
        w.writerows(fp_rows)

    # false_negative.csv
    fn_csv = os.path.join(out_dir, "false_negative.csv")
    with open(fn_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "title", "decision_reason"])
        w.writerows(fn_rows)

    # skipped.csv
    skip_csv = os.path.join(out_dir, "skipped.csv")
    with open(skip_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "title", "skip_reason"])
        w.writerows(skipped_rows)

    # 5. Report
    lines = [
        f"# Evaluation Report v2",
        f"",
        f"**Audit:** `{args.audit}`",
        f"**Phase 5:** `{args.clean_dir}`",
        f"**Generated:** {ts}",
        f"",
        f"## 1. Run Statistics",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Phase 5 Input Rows | {n_total_phase5:,} |",
        f"| Phase 5 Keep Rows | {n_keep:,} |",
        f"| Phase 5 Drop Rows | {n_drop:,} |",
        f"| Keep Rate | {keep_rate:.1f}% |",
        f"| Drop Rate | {drop_rate:.1f}% |",
        f"",
        f"## 2. Coverage",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Audit Samples | {n_audit:,} |",
        f"| Matched Samples | {matched:,} |",
        f"| Skipped Samples | {skipped:,} |",
        f"| Coverage | {coverage:.1f}% |",
        f"",
    ]
    if coverage < COVERAGE_WARN * 100:
        lines.append(f"⚠️ Coverage < {COVERAGE_WARN*100:.0f}%，评估结果可能不具代表性。")
        lines.append(f"")

    lines += [
        f"## 3. Confusion Matrix",
        f"",
        f"| | GT=T (体育) | GT=F (非体育) |",
        f"|---|---|---|",
        f"| Phase5=KEEP | {tp} | {fp} |",
        f"| Phase5=DROP | {fn} | {tn} |",
        f"",
        f"## 4. Metrics",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Precision | {precision:.1f}% |",
        f"| Recall | {recall:.1f}% |",
        f"| F1 | {f1:.1f}% |",
        f"| Accuracy | {accuracy:.1f}% |",
        f"| Retention | {retention:.1f}% |",
        f"",
        f"## 5. Reason Breakdown",
        f"",
        f"### FP Breakdown (漏拦截: Phase5=KEEP, GT=F, n={fp})",
        f"",
        f"| Reason | Count | Pct |",
        f"|---|---|---|",
    ]
    for reason, cnt in fp_reasons.most_common():
        lines.append(f"| {reason} | {cnt} | {cnt/max(fp,1)*100:.1f}% |")

    lines += [
        f"",
        f"### FN Breakdown (误杀: Phase5=DROP, GT=T, n={fn})",
        f"",
        f"| Reason | Count | Pct |",
        f"|---|---|---|",
    ]
    for reason, cnt in fn_reasons.most_common():
        lines.append(f"| {reason} | {cnt} | {cnt/max(fn,1)*100:.1f}% |")

    lines += [
        f"",
        f"## 6. Skipped",
        f"",
        f"{skipped} 条样本未匹配到 Phase 5 产物。",
        f"",
        f"原因: 被 Phase 0 过滤 (null/dedup/damaged)，或 video_id 不存在于 Phase 5 输入。",
        f"",
        f"详细见: {os.path.basename(skip_csv)}",
        f"",
        f"## 7. Outputs",
        f"",
        f"| File | Description |",
        f"|---|---|",
        f"| {os.path.basename(fp_csv)} | 漏拦截: Phase5=KEEP, GT=F ({fp} rows) |",
        f"| {os.path.basename(fn_csv)} | 误杀: Phase5=DROP, GT=T ({fn} rows) |",
        f"| {os.path.basename(skip_csv)} | 未匹配 ({skipped} rows) |",
        f"",
    ]

    # Judgment
    if precision >= 80 and recall >= 70:
        judgment = "✅ 规则达标，建议交付。"
    elif precision < 60:
        judgment = "❌ Precision < 60%，污染率过高。→ 返回 Phase 2 补充 FP 样本分析。"
    elif recall < 50:
        judgment = "⚠️ Recall < 50%，误杀严重。→ 检查 whitelist / scoring 是否过严。"
    else:
        judgment = "⚠️ 中等效果。→ 返回 Phase 2 迭代。"

    lines += [
        f"## 8. Judgment",
        f"",
        judgment,
    ]

    out_md = os.path.join(out_dir, "evaluation_report_v1.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))

    elapsed = time.perf_counter() - t0
    print()
    print("=" * 62)
    print(f"  Phase 6 — 效果验证 完成")
    print("=" * 62)
    print(f"  Audit:         {n_audit:,}")
    print(f"  Matched:       {matched:,}  (coverage {coverage:.1f}%)")
    if skipped:
        print(f"  Skipped:       {skipped:,}")
    print(f"  ┌──────────┬────────┬────────┐")
    print(f"  │          │ GT=T   │ GT=F   │")
    print(f"  ├──────────┼────────┼────────┤")
    print(f"  │ Phase5=K │ {tp:>6} │ {fp:>6} │")
    print(f"  │ Phase5=D │ {fn:>6} │ {tn:>6} │")
    print(f"  └──────────┴────────┴────────┘")
    print(f"  Precision: {precision:.1f}%  Recall: {recall:.1f}%  F1: {f1:.1f}%")
    print(f"  Keep Rate: {keep_rate:.1f}%  耗时: {elapsed:.1f}s")
    print(f"  {out_md}")
    print("=" * 62)
    print()
    print(f"  {judgment}")


if __name__ == "__main__":
    main()

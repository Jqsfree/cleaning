#!/usr/bin/env python3
"""
batch_vision_qc.py — 批量视觉 QC 运行器

遍历 data/data_review_ship_model/ 下所有样本 CSV，依次调用 qc_vision_standalone.py。
支持断点续跑（已有 qc_status 列的文件自动跳过）。

用法:
  # storyboard 模式 (快，推荐)
  python3 batch_vision_qc.py --mode storyboard --backend api

  # 本地 Ollama
  python3 batch_vision_qc.py --mode storyboard --backend local --threads 1

  # 仅跑前 5 个数据集
  python3 batch_vision_qc.py --mode storyboard --backend api --limit 5
"""

import os, sys, csv, subprocess, argparse, time
from pathlib import Path

SHIP_DIR = Path(__file__).resolve().parent.parent / "data" / "data_review_ship_model"
QC_SCRIPT = Path(__file__).resolve().parent / "qc_vision_standalone.py"


def is_completed(csv_path: Path) -> bool:
    """检查 CSV 是否已有 vision QC 结果。"""
    if not csv_path.exists():
        return False
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if "qc_status" not in fieldnames:
                return False
            rows = list(reader)
            if not rows:
                return False
            done = sum(1 for r in rows if r.get("qc_status", "").strip())
            print(f"  {csv_path.name}: {done}/{len(rows)} 已有结果")
            return done >= len(rows) * 0.9  # 90% 以上算完成
    except Exception:
        return False


def discover_csvs(limit: int = 0) -> list[Path]:
    """发现所有待处理的样本 CSV。"""
    if not SHIP_DIR.exists():
        print(f"[错误] 目录不存在: {SHIP_DIR}")
        return []

    all_csvs = sorted(SHIP_DIR.glob("*.csv"))
    print(f"[发现] {len(all_csvs)} 个样本 CSV")

    pending = []
    for p in all_csvs:
        if is_completed(p):
            continue
        pending.append(p)
        if limit and len(pending) >= limit:
            break

    print(f"[待跑] {len(pending)} 个")
    return pending


def run_qc(csv_path: Path, mode: str, backend: str, threads: int) -> bool:
    """对单个 CSV 运行 vision QC。"""
    cmd = [
        sys.executable, str(QC_SCRIPT),
        "--csv", str(csv_path),
        "--mode", mode,
        "--backend", backend,
        "--threads", str(threads),
    ]
    print(f"\n{'='*60}")
    print(f"[运行] {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0

    if result.returncode == 0:
        print(f"✅ 完成 ({elapsed:.0f}s): {csv_path.name}")
        return True
    else:
        print(f"❌ 失败 (exit={result.returncode}): {csv_path.name}")
        return False


def aggregate_results():
    """汇总所有已跑完的 CSV 的 QC 结果。"""
    print(f"\n{'='*60}")
    print("汇总结果")
    print(f"{'='*60}")

    totals = []
    for csv_path in SHIP_DIR.glob("*.csv"):
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "qc_status" not in reader.fieldnames:
                    continue
                rows = list(reader)
                t_count = sum(1 for r in rows if r.get("qc_status", "").strip().upper() == "T")
                f_count = sum(1 for r in rows if r.get("qc_status", "").strip().upper() == "F")
                total = t_count + f_count
                if total == 0:
                    continue
                rate = t_count / total * 100
                totals.append((csv_path.stem[:40], t_count, f_count, total, rate))
        except Exception:
            continue

    totals.sort(key=lambda x: x[4])  # 按通过率排序
    print(f"{'数据集':<42} {'T':>5} {'F':>5} {'总计':>5} {'通过率':>7}")
    print("-" * 67)
    for name, t, f, n, rate in totals:
        print(f"{name:<42} {t:>5} {f:>5} {n:>5} {rate:>6.1f}%")

    if totals:
        avg = sum(t[4] for t in totals) / len(totals)
        print(f"\n平均视觉 QC 通过率: {avg:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="批量视觉 QC 运行器")
    parser.add_argument("--mode", default="storyboard",
                        choices=["storyboard", "video_frames"],
                        help="storyboard=快(默认), video_frames=下载视频抽帧")
    parser.add_argument("--backend", default="api",
                        choices=["api", "local"],
                        help="api=DashScope(默认), local=Ollama")
    parser.add_argument("--threads", type=int, default=1,
                        help="并发线程数 (storyboard+api 可用 2-4)")
    parser.add_argument("--limit", type=int, default=0,
                        help="最多跑几个数据集 (0=全部)")
    args = parser.parse_args()

    pending = discover_csvs(args.limit)
    if not pending:
        print("没有待处理的数据集，退出")
        aggregate_results()
        return

    success = 0
    for i, csv_path in enumerate(pending):
        print(f"\n[{i+1}/{len(pending)}] {csv_path.name}")
        if run_qc(csv_path, args.mode, args.backend, args.threads):
            success += 1

    print(f"\n{'='*60}")
    print(f"批量完成: {success}/{len(pending)} 成功")
    aggregate_results()


if __name__ == "__main__":
    main()

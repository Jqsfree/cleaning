#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import pandas as pd

# 让脚本可 import 项目模块
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "02_脚本").resolve()))
import qc_vision_welding as qw  # noqa: E402


def percentile(vals, p):
    if not vals:
        return 0.0
    vals = sorted(vals)
    k = (len(vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[f]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def load_video_ids(path, sample_n, seed):
    df = pd.read_parquet(path)
    vids = df["video_id"].dropna().astype(str).unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(vids)
    if sample_n > 0:
        vids = vids[:sample_n]
    return vids


def build_auth(output_dir):
    auth = qw.resolve_yt_dlp_auth(None, "chrome")
    auth, cookie_cache, browser_spec = qw.prepare_auth_for_run(
        auth=auth,
        run_id="meta_bench",
        output_dir=output_dir,
        prefetch=qw.PREFETCH_COOKIES,
    )
    if cookie_cache and browser_spec:
        return qw.AuthManager(auth, cookie_cache, browser_spec)
    return auth


def bench_once(video_ids, workers, output_dir):
    auth_mgr = build_auth(output_dir)
    js_runtimes = qw.detect_js_runtimes()
    lock = threading.Lock()

    latencies = []
    ok = 0
    err_counts = {}
    done = 0
    start = time.perf_counter()

    # 模拟真实配置：meta_sleep + 分类错误码
    def one(vid):
        t0 = time.perf_counter()
        try:
            if isinstance(auth_mgr, qw.AuthManager):
                auth = auth_mgr.get_auth()
                ctx = auth_mgr.yt_dlp_section()  # 与线上一致：序列化关键段
            else:
                auth = auth_mgr
                ctx = qw.nullcontext()

            with ctx:
                sb, err = qw.get_storyboard_info(
                    vid,
                    auth=auth,
                    sb_prefer_order=qw.SB_PREFER_ORDER,
                    js_runtimes=js_runtimes,
                    meta_sleep_sec=qw.META_SLEEP_SEC,
                )

            dt = time.perf_counter() - t0
            with lock:
                latencies.append(dt)
                if sb is not None:
                    return ("ok", "")
                return ("err", err or "unknown")

        except Exception as e:
            dt = time.perf_counter() - t0
            with lock:
                latencies.append(dt)
            return ("err", f"exception:{type(e).__name__}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, vid) for vid in video_ids]
        for fut in as_completed(futs):
            status, reason = fut.result()
            if status == "ok":
                ok += 1
            else:
                err_counts[reason] = err_counts.get(reason, 0) + 1
            done += 1
            if done % 20 == 0 or done == len(video_ids):
                elapsed = time.perf_counter() - start
                qps_now = done / elapsed if elapsed > 0 else 0.0
                print(f"[workers={workers}] progress {done}/{len(video_ids)} qps={qps_now:.3f}", flush=True)

    elapsed = time.perf_counter() - start
    total = len(video_ids)
    qps = total / elapsed if elapsed > 0 else 0.0

    result = {
        "workers": workers,
        "total": total,
        "ok": ok,
        "ok_rate": ok / total if total else 0.0,
        "err": total - ok,
        "err_counts": err_counts,
        "rate_limited": err_counts.get("rate_limited", 0),
        "rate_limited_ratio": err_counts.get("rate_limited", 0) / total if total else 0.0,
        "elapsed_sec": elapsed,
        "qps": qps,
        "lat_p50": percentile(latencies, 50),
        "lat_p95": percentile(latencies, 95),
        "lat_avg": statistics.mean(latencies) if latencies else 0.0,
    }
    return result


def main():
    ap = argparse.ArgumentParser(description="yt-dlp metadata benchmark for storyboard extraction")
    ap.add_argument("--input", required=True, help="parquet path with video_id column")
    ap.add_argument("--sample", type=int, default=200, help="sample size")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--workers", default="1,2,3,4", help="comma-separated worker levels")
    ap.add_argument("--output-dir", default="data/runs/welding/005_clean/run02", help="cookie cache/output dir")
    ap.add_argument("--json-out", default="", help="optional output json file")
    ap.add_argument("--write-each", action="store_true", help="write json incrementally after each worker level")
    args = ap.parse_args()

    video_ids = load_video_ids(args.input, args.sample, args.seed)
    worker_levels = [int(x.strip()) for x in args.workers.split(",") if x.strip()]

    print(f"[meta-bench] sample={len(video_ids)} workers={worker_levels}")
    all_results = []

    for w in worker_levels:
        print(f"\n=== workers={w} ===")
        res = bench_once(video_ids, w, args.output_dir)
        all_results.append(res)
        print(
            f"ok={res['ok']}/{res['total']} ({res['ok_rate']:.2%}) "
            f"429={res['rate_limited']} ({res['rate_limited_ratio']:.2%}) "
            f"qps={res['qps']:.3f} p50={res['lat_p50']:.2f}s p95={res['lat_p95']:.2f}s"
        )
        if res["err_counts"]:
            print("err_counts:", res["err_counts"])
        if args.json_out and args.write_each:
            Path(args.json_out).write_text(
                json.dumps(all_results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[partial] Saved JSON -> {args.json_out}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON -> {args.json_out}")


if __name__ == "__main__":
    main()

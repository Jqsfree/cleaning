#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
# -*- coding: utf-8 -*-
"""vlog 片头层 1：读层 0 JSONL 帧路径，调 qwen3-vl 判 handheld vs 固定机位。"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import re
import signal
import sys
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from core.adaptive_api import AdaptiveConcurrencyGate  # noqa: E402
from core.io import resolve_output_dir  # noqa: E402

API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-vl-flash"
MAX_RETRIES = 3
WINDOW_RE = re.compile(r"w(\d+)_", re.IGNORECASE)

_API_GATE: AdaptiveConcurrencyGate | None = None
SYSTEM_PROMPT = ""
USER_PROMPT_TMPL = ""


def load_vision_sb(category: str) -> tuple[str, str]:
    p = _SCRIPT_DIR / "categories" / category / "rules" / "vision_sb.toml"
    if not p.exists():
        print(f"[ERROR] vision_sb.toml 不存在: {p}")
        sys.exit(1)
    cfg = tomllib.loads(p.read_text("utf-8"))
    return cfg["prompts"]["system_prompt"], cfg["prompts"]["user_prompt_tmpl"]


def parse_vision_label(raw: str) -> str | None:
    text = (raw or "").strip().upper()
    if re.fullmatch(r"[TFU]", text):
        return text
    found = set(re.findall(r"[TFU]", text))
    if len(found) == 1:
        return found.pop()
    return None


def pick_vl_frames(frame_paths: list[str], max_frames: int = 5) -> list[str]:
    """每窗取中间 1 张；无窗号则均匀抽 max_frames。"""
    existing = [p for p in frame_paths if p and Path(p).exists()]
    if not existing:
        return []
    by_window: dict[int, list[str]] = {}
    unmatched: list[str] = []
    for p in existing:
        m = WINDOW_RE.search(Path(p).name)
        if m:
            by_window.setdefault(int(m.group(1)), []).append(p)
        else:
            unmatched.append(p)
    if by_window:
        picked = []
        for w in sorted(by_window):
            frames = by_window[w]
            picked.append(frames[len(frames) // 2])
        if len(picked) <= max_frames:
            return picked
        idxs = [
            round(i * (len(picked) - 1) / (max_frames - 1))
            for i in range(max_frames)
        ]
        return [picked[i] for i in idxs]
    if len(existing) <= max_frames:
        return existing
    idxs = [
        round(i * (len(existing) - 1) / (max_frames - 1))
        for i in range(max_frames)
    ]
    return [existing[i] for i in idxs]


def encode_jpeg(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode()


def call_vision(
    frame_paths: list[str],
    model: str,
    api_key: str,
) -> tuple[str, str]:
    if not frame_paths:
        return "ERROR", "no_frames"
    content = []
    for p in frame_paths:
        try:
            b64 = encode_jpeg(p)
        except OSError as exc:
            return "ERROR", f"read_frame:{exc}"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({
        "type": "text",
        "text": USER_PROMPT_TMPL.format(n=len(frame_paths)),
    })
    messages = []
    if SYSTEM_PROMPT and SYSTEM_PROMPT.strip():
        messages.append({"role": "system", "content": SYSTEM_PROMPT.strip()})
    messages.append({"role": "user", "content": content})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 8,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    gate = _API_GATE
    if gate:
        gate.acquire()
    try:
        resp = requests.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        if resp.status_code == 429:
            if gate:
                gate.on_rate_limit()
                gate.record_outcome(transient_error=True)
            return "ERROR", "rate_limited_429"
        if resp.status_code != 200:
            return "ERROR", f"HTTP {resp.status_code}:{resp.text[:80]}"
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        label = parse_vision_label(raw)
        if gate:
            gate.record_outcome(ok=True)
        if label in ("T", "F", "U"):
            return label, raw[:40]
        return "ERROR", f"invalid_response:{raw[:50]}"
    except Exception as exc:
        if gate:
            gate.record_outcome(transient_error=True)
        return "ERROR", f"{type(exc).__name__}:{exc}"[:120]
    finally:
        if gate:
            gate.release()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_done(csv_path: Path) -> tuple[set[str], list[dict]]:
    done: set[str] = set()
    kept: list[dict] = []
    if not csv_path.exists():
        return done, kept
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            vid = (row.get("video_id") or "").strip()
            st = (row.get("qc_vlog_result") or "").strip()
            if not vid or not st:
                continue
            if st == "ERROR":
                continue
            done.add(vid)
            kept.append(row)
    return done, kept


def flatten_record(rec: dict, extra: dict) -> dict:
    out = dict(rec)
    paths = out.get("frame_paths")
    if isinstance(paths, list):
        out["frame_paths"] = json.dumps(paths, ensure_ascii=False)
    out.update(extra)
    return out


def main() -> int:
    global SYSTEM_PROMPT, USER_PROMPT_TMPL, _API_GATE

    parser = argparse.ArgumentParser(
        description="vlog 片头层 1：JSONL 帧 → qwen3-vl T/F/U",
    )
    parser.add_argument("--jsonl", required=True, help="层 0 JSONL")
    parser.add_argument("-c", "--category", default="vlog")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-w", "--workers", type=int, default=8)
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-frames", type=int, default=5, help="送 VL 的最大帧数")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not api_key:
        print("[ERROR] 未设置 DASHSCOPE_API_KEY")
        return 1

    SYSTEM_PROMPT, USER_PROMPT_TMPL = load_vision_sb(args.category)
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"[ERROR] JSONL 不存在: {jsonl_path}")
        return 1

    if args.output:
        out_csv = args.output
        if os.path.isdir(out_csv) or not os.path.splitext(out_csv)[1]:
            out_csv = str(Path(out_csv) / f"{jsonl_path.stem}_vl.csv")
    else:
        out_csv = str(jsonl_path.with_name(jsonl_path.stem + "_vl.csv"))
    out_dir = resolve_output_dir(out_csv, str(jsonl_path))
    os.makedirs(out_dir, exist_ok=True)

    records = load_jsonl(jsonl_path)
    by_id: dict[str, dict] = {}
    for rec in records:
        vid = str(rec.get("video_id") or "").strip()
        if vid:
            by_id[vid] = rec
    records = list(by_id.values())
    if args.limit > 0:
        records = records[: args.limit]
    print(f"[输入] jsonl={len(records)}  model={args.model}  workers={args.workers}")

    done, kept = (set(), [])
    if args.resume:
        done, kept = load_done(Path(out_csv))
        print(f"[resume] 已完成 {len(done)}，ERROR 将重试")

    pending = []
    passthrough = []
    for rec in records:
        vid = str(rec.get("video_id") or "").strip()
        if not vid or vid in done:
            continue
        if rec.get("status") != "READY_FOR_VL":
            continue  # 后面同 video_id 的更新行会覆盖；层0失败不送 VL
        pending.append(rec)

    print(f"[待 VL] {len(pending)}  [层0非READY直写] {len(passthrough)}")

    fieldnames = [
        "video_id", "url", "status", "reason", "duration",
        "width", "height", "sample_count", "valid_sample_count",
        "visual_change_rate", "mean_change", "median_change",
        "static_window_ratio", "dynamic_window_ratio", "frame_paths",
        "qc_vlog_result", "qc_vlog_evidence", "qc_vlog_model", "qc_vlog_n_frames",
    ]
    for rec in records + kept + passthrough:
        for k in rec:
            if k not in fieldnames:
                fieldnames.append(k)

    partial = out_csv + ".partial"
    out_f = open(partial, "w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in kept:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    for row in passthrough:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    out_f.flush()

    stats = {"T": 0, "F": 0, "U": 0, "ERROR": 0}
    for row in kept + passthrough:
        st = (row.get("qc_vlog_result") or "").strip()
        if st in stats:
            stats[st] += 1

    _API_GATE = AdaptiveConcurrencyGate(args.workers)
    write_lock = threading.Lock()
    stop = threading.Event()
    finalized = {"done": False}

    def finalize(reason: str) -> None:
        if finalized["done"]:
            return
        finalized["done"] = True
        try:
            out_f.flush()
            out_f.close()
        except Exception:
            pass
        try:
            os.replace(partial, out_csv)
            print(f"\n[落盘] {reason} → {out_csv}", flush=True)
        except OSError as e:
            print(f"\n[ERROR] 无法替换 {partial} → {out_csv}: {e}", flush=True)

    def on_exit(*_):
        stop.set()
        with write_lock:
            finalize("中断")
        print(f"[中断] T:{stats['T']} F:{stats['F']} U:{stats['U']} E:{stats['ERROR']}")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)

    def work(rec: dict) -> dict:
        paths = rec.get("frame_paths") or []
        if isinstance(paths, str):
            try:
                paths = json.loads(paths)
            except json.JSONDecodeError:
                paths = []
        picked = pick_vl_frames(list(paths), max_frames=args.max_frames)
        label, evidence = "ERROR", "no_frames"
        if picked:
            for _ in range(MAX_RETRIES):
                label, evidence = call_vision(picked, args.model, api_key)
                if label != "ERROR":
                    break
                time.sleep(random.uniform(0.05, 0.3))
        return flatten_record(rec, {
            "qc_vlog_result": label,
            "qc_vlog_evidence": evidence,
            "qc_vlog_model": args.model,
            "qc_vlog_n_frames": len(picked),
        })

    t0 = time.time()
    done_n = 0
    n_pending = len(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, rec): rec for rec in pending}
        for fut in as_completed(futs):
            if stop.is_set():
                break
            row = fut.result()
            st = row.get("qc_vlog_result") or "ERROR"
            with write_lock:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
                out_f.flush()
                if st in stats:
                    stats[st] += 1
                done_n += 1
            elapsed = time.time() - t0
            rate = done_n / (elapsed / 3600) if elapsed else 0
            print(
                f"[{done_n}/{n_pending}] {row.get('video_id')} {st} "
                f"T:{stats['T']} F:{stats['F']} U:{stats['U']} E:{stats['ERROR']} "
                f"~{rate:.0f}/h",
                flush=True,
            )

    finalize("done")
    print(
        f"[完成] T:{stats['T']} F:{stats['F']} U:{stats['U']} "
        f"E:{stats['ERROR']} → {out_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

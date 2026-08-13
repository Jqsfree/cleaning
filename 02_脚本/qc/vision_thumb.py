#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
"""
缩略图视觉 QC — 直接从 i.ytimg.com 下载缩略图 + Qwen VL 判断
用法:
  python3 qc_vision_thumb.py input.parquet --category lila_outdoor --sample 300 -t 12
  python3 qc_vision_thumb.py input.csv --category ego_repair --resume -t 8
"""

import csv, os, sys, time, base64, signal, threading, random, tomllib
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))
from core.io import resolve_output_dir
from core.progress import ThrottledProgress, mark_done
from core.sop import write_run_log
from core.adaptive_api import AdaptiveConcurrencyGate

# ============================================================
CONFIG = {
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "qwen3-vl-flash",
    "api_timeout": 30,
    "threads": 12,
    "http_retries": 3,
    "http_backoff_base": 2.0,
    "http_jitter_max": 0.05,
    "cache_dir": "qc_thumb_cache",
}

_API_GATE: AdaptiveConcurrencyGate | None = None

_CATEGORIES_DIR = _SCRIPT_DIR / "categories"

# ============================================================
# 提示词 — 从 TOML 加载
# ============================================================

def load_vision_prompts(category: str):
    p = _CATEGORIES_DIR / category / "rules" / "vision_thumb.toml"
    if not p.exists():
        print(f"[ERROR] vision_thumb.toml 不存在: {p}")
        sys.exit(1)
    cfg = tomllib.loads(p.read_text("utf-8"))
    return cfg["prompts"]["system_prompt"], cfg["prompts"]["user_prompt_tmpl"]

SYSTEM_PROMPT = ""
USER_PROMPT_TMPL = ""

# ============================================================
# 缩略图下载
# ============================================================

THUMB_SUFFIXES = ['maxresdefault', 'hqdefault', 'mqdefault', 'sddefault', '0']

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
]

def download_thumbnail(video_id, cache_dir=None):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        for suffix in THUMB_SUFFIXES:
            cache_path = os.path.join(cache_dir, f'{video_id}_{suffix}.jpg')
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    data = f.read()
                if len(data) >= 1500:
                    return data, suffix

    for suffix in THUMB_SUFFIXES:
        url = f'https://img.youtube.com/vi/{video_id}/{suffix}.jpg'
        for attempt in range(CONFIG['http_retries']):
            try:
                time.sleep(random.uniform(0, CONFIG['http_jitter_max']))
                ua = random.choice(USER_AGENTS)
                resp = requests.get(url, headers={'User-Agent': ua}, timeout=12)
                if resp.status_code == 429:
                    wait = CONFIG['http_backoff_base'] ** attempt + random.uniform(0, 1)
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    break
                data = resp.content
                if len(data) < 1500:
                    break
                if cache_dir:
                    cache_path = os.path.join(cache_dir, f'{video_id}_{suffix}.jpg')
                    with open(cache_path, 'wb') as f:
                        f.write(data)
                return data, suffix
            except Exception:
                if attempt < CONFIG['http_retries'] - 1:
                    time.sleep(CONFIG['http_backoff_base'] + random.uniform(0, 1))
                    continue
                break
    return None, None

# ============================================================
# Vision API
# ============================================================

def call_vision(image_data, model=None):
    if model is None:
        model = CONFIG["model"]
    b64 = base64.b64encode(image_data).decode()
    user_text = USER_PROMPT_TMPL.format(n=1)
    # Qwen-VL：非 Agent 场景不设 system，规则放在 user（百炼文档建议）
    messages = []
    if SYSTEM_PROMPT and SYSTEM_PROMPT.strip():
        messages.append({"role": "system", "content": SYSTEM_PROMPT.strip()})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": user_text},
        ],
    })
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 5,
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY')}",
        "Content-Type": "application/json",
    }
    gate = _API_GATE
    if gate:
        gate.acquire()
    try:
        resp = requests.post(
            f"{CONFIG['api_base']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=CONFIG["api_timeout"],
        )
        if resp.status_code == 429:
            if gate:
                gate.on_rate_limit()
                gate.record_outcome(transient_error=True)
            return "ERROR", "rate_limited_429"
        if resp.status_code != 200:
            return "ERROR", f"HTTP {resp.status_code}: {resp.text[:100]}"
        answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
        if gate:
            gate.record_outcome(ok=True)
        if "T" in answer:
            return "T", answer
        return "F", answer
    except Exception as e:
        if gate:
            gate.record_outcome(transient_error=True)
        return "ERROR", str(e)[:100]
    finally:
        if gate:
            gate.release()

# ============================================================
# 单条质检
# ============================================================

def check_one(video_id, title="", channel=""):
    img_data, suffix = download_thumbnail(video_id, CONFIG['cache_dir'])
    if not img_data:
        return "ERROR", "no_thumbnail"
    for _ in range(3):
        result, raw = call_vision(img_data)
        if result != "ERROR":
            return result, raw
        # 429 时靠 gate 限流；仅极短 jitter 后重试
        time.sleep(random.uniform(0.05, 0.25))
    return "ERROR", "max_retries"

def check_row(row):
    def _s(v, n=None):
        if v is None or (isinstance(v, float) and v != v):  # NaN
            return ""
        s = str(v).strip()
        if s.lower() in ("nan", "none", "<na>"):
            return ""
        return s[:n] if n is not None else s

    vid = _s(row.get("video_id", ""))
    title = _s(row.get("title"), 60)
    ch = _s(row.get("channel"))
    result, evidence = check_one(vid, title, ch)
    return row, result, evidence

# ============================================================
# 批量执行
# ============================================================

def batch_qc(input_path, output_csv=None, sample=None, resume=False, threads=12, seed=42,
             category: str = ""):
    if output_csv is None:
        output_csv = input_path.replace(".parquet", "_thumb_qc.csv").replace(".csv", "_thumb_qc.csv")
    elif os.path.isdir(output_csv):
        # 与 welding 一致：-o 可传目录，自动用输入 stem 生成 CSV
        stem = Path(input_path).stem
        output_csv = str(Path(output_csv) / f"{stem}_thumb_qc.csv")

    out_dir = resolve_output_dir(output_csv, input_path)
    os.makedirs(out_dir, exist_ok=True)

    print("[加载] 读取输入...")
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path, dtype=str, low_memory=False).fillna("")
    N = len(df)
    print(f"[输入] {N:,} 条")

    if sample and sample < N:
        df = df.sample(n=sample, random_state=seed)
        print(f"[抽样] {sample} 条")

    completed = set()
    kept_rows = []  # 续跑时保留的终态行（不含 ERROR）
    stats = {"T": 0, "F": 0, "ERROR": 0}
    n_retry_err = 0
    if resume and os.path.exists(output_csv):
        print("[恢复] 读取已完成的记录（ERROR 可重试）...")
        for row in csv.DictReader(open(output_csv, "r", encoding="utf-8-sig")):
            vid = (row.get("video_id") or "").strip()
            st = (row.get("qc_thumb_result") or "").strip()
            if not vid or not st:
                continue
            if st == "ERROR":
                n_retry_err += 1
                continue
            if st in stats:
                stats[st] += 1
            completed.add(vid)
            kept_rows.append(row)
        print(
            f"[恢复] done:{len(completed)} T:{stats['T']} F:{stats['F']} "
            f"| 将重试 ERROR:{n_retry_err}"
        )

    out_cols = list(df.columns)
    for c in ["qc_thumb_result", "qc_thumb_evidence"]:
        if c not in out_cols:
            out_cols.append(c)

    pending_rows = []
    for idx, row in df.iterrows():
        if str(row.get("video_id", "")).strip() not in completed:
            pending_rows.append((idx, row))
    n_pending = len(pending_rows)
    print(f"[待检] {n_pending}  (线程: {threads})")
    print()

    global _API_GATE
    _API_GATE = AdaptiveConcurrencyGate(threads)

    # 续跑一律重写：丢掉旧 ERROR，保留 T/F，避免 append 重复
    out = open(output_csv, "w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(out, fieldnames=out_cols, extrasaction="ignore")
    writer.writeheader()
    for row in kept_rows:
        writer.writerow({k: row.get(k, "") for k in out_cols})
    out.flush()
    stats["ERROR"] = 0


    write_lock = threading.Lock()
    stop_event = threading.Event()

    def on_exit(*_):
        stop_event.set()
        out.close()
        s = stats
        print(f"\n[中断] T:{s['T']} F:{s['F']} E:{s['ERROR']}")
        sys.exit(0)
    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)

    t0 = time.time()
    done_count = 0
    CHUNK = 2000
    idx_ptr = 0
    prog = ThrottledProgress(
        out_dir, "qc_vision_thumb",
        interval_sec=5.0, every_n=50,
        input=input_path, output=output_csv, category=category,
        total=n_pending,
    )
    prog.tick(force=True, done=0, T=stats["T"], F=stats["F"], ERROR=stats["ERROR"])

    def progress_line():
        elapsed = time.time() - t0
        rate = done_count / (elapsed / 3600) if elapsed and done_count else 0
        s = stats
        return f"[{done_count}/{n_pending}] T:{s['T']} F:{s['F']} E:{s['ERROR']} ~{rate:.0f}/h"

    with ThreadPoolExecutor(max_workers=threads) as executor:
        while idx_ptr < len(pending_rows) and not stop_event.is_set():
            chunk = pending_rows[idx_ptr:idx_ptr + CHUNK]
            idx_ptr += CHUNK
            futures = {executor.submit(check_row, row): (orig_idx, row) for orig_idx, row in chunk}

            for future in as_completed(futures):
                if stop_event.is_set():
                    for f in futures: f.cancel()
                    break

                orig_idx, row = futures[future]
                try:
                    _, result, evidence = future.result()
                except Exception as e:
                    result, evidence = "ERROR", f"future_exception:{type(e).__name__}:{e}"[:200]

                stats[result] = stats.get(result, 0) + 1

                out_row = {k: str(row.get(k, "")) for k in out_cols}
                out_row["qc_thumb_result"] = result
                out_row["qc_thumb_evidence"] = evidence

                with write_lock:
                    writer.writerow(out_row)
                    out.flush()

                done_count += 1
                prog.tick(done=done_count, T=stats.get("T", 0), F=stats.get("F", 0),
                          ERROR=stats.get("ERROR", 0))
                print(f"\r{progress_line()}", end="", flush=True)

            futures.clear()

    out.close()
    total = stats.get("T", 0) + stats.get("F", 0)
    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"[完成] {elapsed/3600:.1f}h  {total/(elapsed/3600):.0f}/h")
    print(f"  T (通过):  {stats.get('T', 0):>8,}")
    print(f"  F (不通过): {stats.get('F', 0):>8,}")
    if stats.get("ERROR", 0) > 0:
        print(f"  ERROR:     {stats.get('ERROR', 0):>8,}")
    if total > 0:
        print(f"  T rate:    {stats.get('T', 0)/total*100:.0f}%")
    print(f"  输出:      {output_csv}")
    print(f"{'='*50}")

    mark_done(
        out_dir, "qc_vision_thumb",
        input=input_path, output=output_csv, category=category,
        done=done_count, total=n_pending,
        T=stats.get("T", 0), F=stats.get("F", 0), ERROR=stats.get("ERROR", 0),
        elapsed_sec=round(elapsed, 1),
    )
    write_run_log(
        "qc_vision_thumb", input_path, out_dir,
        stats={
            "category": category,
            "T": stats.get("T", 0),
            "F": stats.get("F", 0),
            "ERROR": stats.get("ERROR", 0),
            "elapsed_sec": round(elapsed, 1),
            "output_csv": output_csv,
        },
        command=f"vision_thumb.py {input_path} -c {category} -o {out_dir}",
        category=category or None,
    )

# ============================================================
# CLI
# ============================================================

def main():
    global SYSTEM_PROMPT, USER_PROMPT_TMPL

    import argparse
    p = argparse.ArgumentParser(description="缩略图视觉 QC — i.ytimg.com + Qwen VL")
    p.add_argument("input", help="输入 parquet 或 CSV")
    p.add_argument("-c", "--category", required=True, help="类别名（加载 vision_thumb.toml）")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("-n", "--sample", type=int, default=0, help="抽样 N 条（0=全量）")
    p.add_argument("--resume", action="store_true")
    p.add_argument("-t", "--threads", type=int, default=12)
    p.add_argument("-m", "--model", default="qwen3-vl-flash")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default="qc_thumb_cache")
    args = p.parse_args()

    SYSTEM_PROMPT, USER_PROMPT_TMPL = load_vision_prompts(args.category)
    CONFIG["model"] = args.model
    CONFIG["threads"] = args.threads
    CONFIG["cache_dir"] = args.cache_dir

    print(f"[配置] category={args.category}  model={args.model}  threads={args.threads}")
    if SYSTEM_PROMPT and SYSTEM_PROMPT.strip():
        print(f"[提示] system: {SYSTEM_PROMPT.strip()[:50]}...")
    else:
        print(f"[提示] user-only（无 system）: {USER_PROMPT_TMPL[:60].replace(chr(10), ' ')}...")

    batch_qc(args.input, args.output,
             args.sample if args.sample > 0 else None,
             args.resume, threads=args.threads, seed=args.seed,
             category=args.category)

if __name__ == "__main__":
    main()

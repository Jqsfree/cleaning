#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
"""
视频画面质检脚本 — mqdefault + Qwen VL API
用法:
  python3 qc_vision.py input.csv --sample 100
  python3 qc_vision.py input.csv --resume -t 8
"""

import csv, json, os, re, sys, time, base64, signal, threading, random
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from core.io import resolve_output_dir
from core.progress import ThrottledProgress, mark_done
from core.sop import write_run_log
from core.adaptive_api import AdaptiveConcurrencyGate

# ============================================================
CONFIG = {
    "api_key": "",  # 运行时从 DASHSCOPE_API_KEY 读取，禁止硬编码
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "qwen3-vl-flash",
    "local_model": "qwen3-vl:8b",
    "ollama_base": "http://localhost:11434",
    "api_timeout": 30,
    "delay": 0.05,
    "max_retries": 2,
    "cache_dir": "qc_thumb_cache",
    "threads": 12,
    "http_retries": 3,
    "http_backoff_base": 2.0,
    "http_jitter_max": 0.05,
    "api_jitter_max": 0.05,
    "use_local": False,
}

_API_GATE: AdaptiveConcurrencyGate | None = None

# 频道黑名单 — 零误杀，直接跳过vision
CHANNEL_BLACKLIST = {
    "ABS-CBN Entertainment", "Vijay Television", "TEDx Talks", "MeidasTouch",
    "Flamingo", "The Organic Chemistry Tutor", "StarPlus", "The Indy Author",
    "Skywalker Pan", "Dangal Ki Prem Kahaaniyan", "Daily Dose Of Internet",
    "Hoshyarian", "FOX 2 Detroit", "FailArmy", "TS4U", "Two Players One Console",
    "HL GX", "Kuma - Turkish Series in English", "Susie Larson Live", "Zee Tamil",
    "Pheri Wala", "Ajay Yadav Official Channel", "Pinoy Big Brother",
    "The Tonight Show Starring Jimmy Fallon", "Examपुर", "Flowers Comedy",
    "Thressa Sweat", "Number1TrendsVideos", "JianHao Tan", "SONU PATEL LLB",
    "MUSLIM TV. PRESS", "Washington State Health Care Authority", "Shemaroo",
    "Judge Mathis", "Dil Aur Pyar Ki Baatein", "Unspeakable",
    "The Late Show with Stephen Colbert", "CookieSwirlC", "90 Day Fiancé",
    "WWE", "Mel Robbins", "Aphmau", "TVC News Nigeria",
    "Zee TV", "SET India", "Colors TV", "Star Plus", "Sony SAB",
    "Star Vijay", "Sun TV", "Zee Telugu", "Star Maa", "Dangal TV",
}

# ============================================================
# ============================================================
# L0: 标题硬规则 — patterns moved to core/regex_patterns.py
# ============================================================
from core.regex_patterns import (
    ANIME_PATTERNS, MUSIC_PATTERNS, PLATFORM_PATTERNS, VARIETY_PATTERNS,
    DRAMA_PATTERNS, NEWS_PATTERNS, SPORTS_PATTERNS, LECTURE_PATTERNS,
    LIVE_POSTER_PATTERNS, FAN_IDOL_PATTERNS, SOLO_PATTERNS,
)

def title_classify(title):
    t = (title or "").lower()
    for p in ANIME_PATTERNS:
        if re.search(p, t):
            return "fail", "anime/cartoon"
    for p in MUSIC_PATTERNS:
        if re.search(p, t):
            return "fail", "music/mtv"
    for p in PLATFORM_PATTERNS:
        if re.search(p, t):
            return "fail", "platform/watermark"
    for p in VARIETY_PATTERNS:
        if re.search(p, t):
            return "fail", "variety/bts"
    for p in DRAMA_PATTERNS:
        if re.search(p, t):
            return "fail", "drama/clips"
    for p in NEWS_PATTERNS:
        if re.search(p, t):
            return "fail", "news/press"
    for p in SPORTS_PATTERNS:
        if re.search(p, t):
            return "fail", "sports"
    for p in LECTURE_PATTERNS:
        if re.search(p, t):
            return "fail", "lecture/seminar"
    for p in LIVE_POSTER_PATTERNS:
        if re.search(p, t):
            return "fail", "live/poster"
    for p in FAN_IDOL_PATTERNS:
        if re.search(p, t):
            return "fail", "fan/idol"
    for p in SOLO_PATTERNS:
        if re.search(p, t):
            return "uncertain", "solo/vlog_pattern"
    # Regex only does NEGATIVE filtering; positive judgment is left to vision
    return "maybe", "no_signal"

# ============================================================
# 缩略图下载 — 多分辨率降级: maxresdefault → hqdefault → mqdefault → sddefault → 0
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
    """多级降级下载 YouTube 缩略图，返回 (image_bytes, suffix_used) 或 (None, None)"""
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
                    break  # 此分辨率不可用，试下一个
                data = resp.content
                if len(data) < 1500:
                    break  # 占位图，试下一个
                if cache_dir:
                    cache_path = os.path.join(cache_dir, f'{video_id}_{suffix}.jpg')
                    with open(cache_path, 'wb') as f:
                        f.write(data)
                return data, suffix
            except requests.Timeout:
                if attempt < CONFIG['http_retries'] - 1:
                    time.sleep(CONFIG['http_backoff_base'] + random.uniform(0, 1))
                    continue
                break
            except Exception:
                if attempt < CONFIG['http_retries'] - 1:
                    time.sleep(CONFIG['http_backoff_base'] + random.uniform(0, 1))
                    continue
                break

    return None, None

# ============================================================
# Qwen VL API
# ============================================================

VISION_PROMPT = (
    "You are a strict classifier for video thumbnails. Answer exactly ONE WORD: YES or NO.\n\n"
    "FIRST, check for automatic disqualifiers — answer NO immediately if ANY apply:\n"
    "- 3 or more real people physically present in the frame.\n"
    "- 0 or 1 real person physically present.\n"
    "- Any person appearing only on a screen, monitor, or as a mirror reflection does NOT count as a real person. If removing such people leaves fewer than 2 real people, answer NO.\n"
    "- Variety show stage, talent show, idol/fan content, or scripted entertainment setting.\n"
    "- Drama, movie, or web series still/poster.\n"
    "- News broadcast desk (with chyrons, tickers, or studio news backdrop).\n"
    "- Sports commentary, cooking competition, or game show set.\n"
    "- Concert, live performance, music video, or DJ booth.\n"
    "- Anime, CGI, cartoon, or illustrated content.\n"
    "- One person teaching/presenting toward camera (lecture, seminar, product demo).\n"
    "- Variety-show text overlays, stage spotlights, or dramatic performance lighting.\n"
    "- Thumbnail is primarily text/title card (text covers >50% of frame).\n\n"
    "ONLY IF none of the above apply, answer YES if ALL THREE are true:\n"
    "1. COUNT: Exactly two real people are physically present in the frame.\n"
    "2. FACES: Both people's faces must be clearly visible in the frame. "
    "Partially obscured, turned away, or unidentifiable faces = NO.\n"
    "3. SCENE: The two people are in a conversation, interview, podcast, or talk show setting — "
    "clearly engaged with each other in dialogue, not performing for an audience.\n\n"
    "If uncertain, answer NO.\n\n"
    "ONE WORD: YES or NO"
)

def call_qwen_vision(image_data, model=None):
    if model is None:
        model = CONFIG["model"]
    b64 = base64.b64encode(image_data).decode()
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "temperature": 0,
        "max_tokens": 5,
    }
    headers = {
        "Authorization": f"Bearer {CONFIG['api_key']}",
        "Content-Type": "application/json",
    }
    gate = _API_GATE
    if gate:
        gate.acquire()
    try:
        time.sleep(random.uniform(0, CONFIG["api_jitter_max"]))
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
            return {"result": "error", "raw": "rate_limited_429"}
        if resp.status_code != 200:
            return {"result": "error", "raw": f"HTTP {resp.status_code}: {resp.text[:100]}"}
        data = resp.json()
        answer = data["choices"][0]["message"]["content"].strip().upper()
        if gate:
            gate.record_outcome(ok=True)
        if answer.startswith("YES"):
            return {"result": "pass", "raw": answer}
        elif answer.startswith("NO"):
            return {"result": "fail", "raw": answer}
        elif "YES" in answer:
            return {"result": "pass", "raw": answer}
        else:
            return {"result": "fail", "raw": answer}
    except requests.Timeout:
        if gate:
            gate.record_outcome(transient_error=True)
        return {"result": "error", "raw": "timeout"}
    except Exception as e:
        if gate:
            gate.record_outcome(transient_error=True)
        return {"result": "error", "raw": str(e)[:100]}
    finally:
        if gate:
            gate.release()


def call_qwen_vision_local(image_data):
    """调用本地 Ollama qwen3-vl 模型 (原生 /api/generate)"""
    b64 = base64.b64encode(image_data).decode()
    prompt = (
        "Look at this YouTube video thumbnail. "
        "Answer EXACTLY ONE WORD: YES if 2 real humans appear together in conversation, "
        "NO otherwise. One word:"
    )
    payload = {
        "model": CONFIG["local_model"],
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 50,
            "num_ctx": 4096,
        },
    }
    try:
        time.sleep(random.uniform(0, CONFIG["api_jitter_max"]))
        resp = requests.post(
            f"{CONFIG['ollama_base']}/api/generate",
            json=payload,
            timeout=120,
        )
        if resp.status_code != 200:
            return {"result": "error", "raw": f"HTTP {resp.status_code}: {resp.text[:100]}"}
        data = resp.json()
        answer = data.get("response", "").strip().upper()
        if answer.startswith("YES"):
            return {"result": "pass", "raw": answer}
        elif answer.startswith("NO"):
            return {"result": "fail", "raw": answer}
        elif "YES" in answer:
            return {"result": "pass", "raw": answer}
        else:
            return {"result": "fail", "raw": answer}
    except requests.Timeout:
        return {"result": "error", "raw": "local_timeout"}
    except Exception as e:
        return {"result": "error", "raw": str(e)[:100]}


# ============================================================
# 单条质检
# ============================================================

def check_one(video_id, title="", duration_str="", channel=""):
    try:
        dur = float(duration_str) if duration_str else 0
    except:
        dur = 0
    # 频道黑名单 — 零成本跳过vision
    if channel.strip() in CHANNEL_BLACKLIST:
        return {"qc_status": "fail", "qc_result": "channel_blacklist", "evidence": "channel"}
    if dur > 0 and dur < 90:
        return {"qc_status": "fail", "qc_result": f"too_short:{dur:.0f}s", "evidence": "duration"}
    if dur > 0 and dur > 21600:
        return {"qc_status": "fail", "qc_result": f"too_long:{dur:.0f}h", "evidence": "duration"}
    t_class, t_reason = title_classify(title)
    if t_class == "fail":
        return {"qc_status": "fail", "qc_result": f"title:{t_reason}", "evidence": "title"}
    img_data, thumb_suffix = download_thumbnail(video_id, CONFIG['cache_dir'])
    if not img_data:
        if t_class == 'pass':
            return {'qc_status': 'pass', 'qc_result': 'no_thumb:title_pass', 'evidence': 'title_only'}
        return {'qc_status': 'uncertain', 'qc_result': 'no_thumbnail', 'evidence': 'none'}
    if CONFIG['use_local']:
        vision = call_qwen_vision_local(img_data)
    else:
        vision = call_qwen_vision(img_data)
    v_result = vision.get("result", "error")
    tag = 'local' if CONFIG.get('use_local') else (thumb_suffix if thumb_suffix else 'thumb')
    if t_class == "pass":
        if v_result == "pass":
            return {"qc_status": "pass", "qc_result": f"title_pass+ai_pass [{tag}]", "evidence": "title+ai"}
        elif v_result == "fail":
            return {"qc_status": "uncertain", "qc_result": f"conflict:title_pass_ai_fail [{tag}]", "evidence": "conflict"}
        else:
            return {"qc_status": "fail", "qc_result": f"ai_fail [{tag}]", "evidence": "ai"}
    elif t_class == "maybe":
        if v_result == "pass":
            return {"qc_status": "pass", "qc_result": f"ai_pass [{tag}]", "evidence": "ai"}
        elif v_result == "fail":
            return {"qc_status": "fail", "qc_result": f"ai_fail [{tag}]", "evidence": "ai"}
        else:
            return {"qc_status": "uncertain", "qc_result": f"maybe+ai_error [{tag}]", "evidence": "ai_weak"}
    elif t_class == "uncertain":
        if v_result == "pass":
            return {"qc_status": "pass", "qc_result": f"ai_pass_despite [{tag}]", "evidence": "ai"}
        return {"qc_status": "uncertain", "qc_result": f"both_uncertain [{tag}]", "evidence": "none"}
    return {"qc_status": "uncertain", "qc_result": "fallback", "evidence": "none"}

def check_row(row):
    vid = row.get("video_id", "").strip()
    title = (row.get("title") or "")[:60]
    dur_str = (row.get("duration_seconds") or "").strip()
    ch = (row.get("channel") or "").strip()
    for _ in range(CONFIG["max_retries"]):
        try:
            result = check_one(vid, title, dur_str, ch)
            if result and result.get("qc_status") not in ("error", None):
                time.sleep(random.uniform(0, 0.15))
                return row, result
        except Exception:
            pass
        time.sleep(0.3)
    return row, {"qc_status": "error", "qc_result": "exception", "evidence": ""}

# ============================================================
# 批量执行
# ============================================================

OUTPUT_FIELDS = [
    "keyword", "title", "video_id", "url", "channel",
    "duration_seconds", "view_count", "upload_date",
    "description", "keywords", "source_type", "source_ref",
    "qc_status", "qc_result", "qc_evidence", "qc_updated_at",
]

def batch_qc(input_csv, output_csv=None, sample=None, resume=False, threads=4, seed=42):
    if output_csv is None:
        output_csv = input_csv.replace(".csv", "_qc.csv")

    out_dir = resolve_output_dir(output_csv, input_csv)
    os.makedirs(out_dir, exist_ok=True)

    # --- 加载输入 ---
    print("[加载] 读取输入...")
    with open(input_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    N = len(all_rows)
    print(f"[输入] {N:,} 条")

    if sample and sample < N:
        import random as rnd
        rnd.seed(seed)
        all_rows = rnd.sample(all_rows, sample)
        print(f"[抽样] {sample} 条")

    # --- 恢复进度 (从输出 CSV, 逐行读, 不排序) ---
    completed = set()
    stats = {"pass": 0, "fail": 0, "uncertain": 0, "error": 0}

    if resume and os.path.exists(output_csv):
        print("[恢复] 读取已完成的记录...")
        for row in csv.DictReader(open(output_csv, "r", encoding="utf-8-sig")):
            vid = row.get("video_id", "").strip()
            st = row.get("qc_status", "").strip()
            if vid and st:
                completed.add(vid)
                if st in stats:
                    stats[st] += 1
        print(f"[恢复] done:{len(completed)} P:{stats['pass']} F:{stats['fail']} U:{stats['uncertain']}")

    pending = [r for r in all_rows if r.get("video_id", "").strip() not in completed]
    n_pending = len(pending)
    print(f"[待检] {n_pending}  (线程: {threads})")
    print()

    global _API_GATE
    if not CONFIG.get("use_local"):
        _API_GATE = AdaptiveConcurrencyGate(threads)
    else:
        _API_GATE = None

    # --- 输出 ---
    mode = "a" if (resume and os.path.exists(output_csv)) else "w"
    out = open(output_csv, mode, encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(out, fieldnames=OUTPUT_FIELDS)
    if mode == "w":
        writer.writeheader()

    write_lock = threading.Lock()
    stop_event = threading.Event()

    def on_exit(*_):
        stop_event.set()
        out.close()
        s = stats
        print(f"\n[中断] P:{s['pass']} F:{s['fail']} U:{s['uncertain']} E:{s['error']}")
        sys.exit(0)
    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)

    t0 = time.time()
    done_count = 0
    CHUNK = 2000
    idx = 0
    prog = ThrottledProgress(
        out_dir, "qc_two_person",
        interval_sec=5.0, every_n=50,
        input=input_csv, output=output_csv, total=n_pending,
    )
    prog.tick(force=True, done=0, **{k: stats[k] for k in ("pass", "fail", "uncertain", "error")})

    def progress_line():
        elapsed = time.time() - t0
        rate = done_count / (elapsed / 3600) if elapsed and done_count else 0
        s = stats
        return f"[{done_count}/{n_pending}] P:{s['pass']} F:{s['fail']} U:{s['uncertain']} ~{rate:.0f}/h"

    with ThreadPoolExecutor(max_workers=threads) as executor:
        while idx < len(pending) and not stop_event.is_set():
            chunk = pending[idx:idx + CHUNK]
            idx += CHUNK

            futures = {executor.submit(check_row, row): row for row in chunk}

            for future in as_completed(futures):
                if stop_event.is_set():
                    for f in futures:
                        f.cancel()
                    break

                row = futures[future]
                try:
                    row, result = future.result()
                except Exception:
                    result = {"qc_status": "error", "qc_result": "future_exception", "evidence": ""}

                if result is None:
                    stats["error"] += 1
                    result = {"qc_status": "error", "qc_result": "empty", "evidence": ""}

                st = result["qc_status"]
                if st in stats:
                    stats[st] += 1
                else:
                    stats["error"] += 1

                out_row = {k: row.get(k, "") for k in OUTPUT_FIELDS}
                out_row["qc_status"] = result.get("qc_status", "error")
                out_row["qc_result"] = result.get("qc_result", "")
                out_row["qc_evidence"] = result.get("evidence", "")
                out_row["qc_updated_at"] = datetime.now().isoformat()

                with write_lock:
                    writer.writerow(out_row)
                    out.flush()

                done_count += 1
                print(f"\r{progress_line()}", end="", flush=True)
                prog.tick(
                    done=done_count,
                    n_pass=stats["pass"], n_fail=stats["fail"],
                    n_uncertain=stats["uncertain"], n_error=stats["error"],
                )

            # 每批写完释放 futures
            futures.clear()

    out.close()
    total = stats["pass"] + stats["fail"] + stats["uncertain"]
    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"[完成] {elapsed/3600:.1f}h  {total/(elapsed/3600):.0f}/h")
    print(f"  通过:   {stats['pass']:>8,}  ({stats['pass']/max(1,total)*100:.0f}%)")
    print(f"  不通过: {stats['fail']:>8,}  ({stats['fail']/max(1,total)*100:.0f}%)")
    print(f"  不确定: {stats['uncertain']:>8,}")
    print(f"  错误:   {stats['error']:>8,}")
    print(f"  输出:   {output_csv}")
    print(f"{'='*50}")

    mark_done(
        out_dir, "qc_two_person",
        input=input_csv, output=output_csv,
        done=done_count, total=n_pending,
        n_pass=stats["pass"], n_fail=stats["fail"],
        n_uncertain=stats["uncertain"], n_error=stats["error"],
        elapsed_sec=round(elapsed, 1),
    )
    write_run_log(
        "qc_two_person", input_csv, out_dir,
        stats={
            "pass": stats["pass"],
            "fail": stats["fail"],
            "uncertain": stats["uncertain"],
            "error": stats["error"],
            "elapsed_sec": round(elapsed, 1),
            "output_csv": output_csv,
        },
        command=f"vision_two_person.py {input_csv} -o {output_csv}",
    )


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="视频QC — mqdefault + Qwen VL")
    p.add_argument("input")
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--sample", "-n", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--threads", "-t", type=int, default=12)
    p.add_argument("--seed", type=int, default=42, help="随机种子，用于抽取不同样本")
    p.add_argument("--delay", type=float, default=0.05)
    p.add_argument("--model", default="qwen-vl-plus",
                   choices=["qwen-vl-max", "qwen-vl-plus", "qwen3-vl-plus", "qwen3-vl-flash"])
    p.add_argument("--local", action="store_true", help="使用本地 Ollama qwen3-vl:8b")
    p.add_argument("--local-model", default="qwen3-vl:8b")
    args = p.parse_args()

    CONFIG["delay"] = args.delay
    CONFIG["model"] = args.model
    CONFIG["threads"] = args.threads
    CONFIG["use_local"] = args.local
    if args.local:
        CONFIG["local_model"] = args.local_model

    if not args.local:
        api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
        if not api_key:
            print("[ERROR] 未设置 DASHSCOPE_API_KEY 环境变量")
            sys.exit(1)
        CONFIG["api_key"] = api_key

    if args.local:
        print(f"[配置] local={CONFIG['local_model']}  threads={args.threads}  thumb=mqdefault")
    else:
        print(f"[配置] model={CONFIG['model']}  threads={args.threads}  thumb=mqdefault")

    if args.local:
        print("[检查] Ollama...", end=" ", flush=True)
        try:
            r = requests.get(f"{CONFIG['ollama_base']}/api/tags", timeout=10)
            if r.status_code == 200:
                models_data = r.json().get("models", [])
                names = [m["name"] for m in models_data if "vl" in m["name"].lower()]
                print(f"OK ({len(names)} VL models: {names})")
            else:
                print(f"HTTP {r.status_code}: {r.text[:80]}")
        except Exception as e:
            print(f"FAIL: {e}")
    else:
        print("[检查] API...", end=" ", flush=True)
        try:
            r = requests.get(
                f"{CONFIG['api_base']}/models",
                headers={"Authorization": f"Bearer {CONFIG['api_key']}"},
                timeout=10,
            )
            if r.status_code == 200:
                models = r.json().get("data", [])
                names = [m["id"] for m in models if "vl" in m["id"].lower()]
                print(f"OK ({len(names)} VL models)")
            else:
                print(f"HTTP {r.status_code}: {r.text[:80]}")
        except Exception as e:
            print(f"FAIL: {e}")
    print()

    batch_qc(args.input, args.output,
             args.sample if args.sample > 0 else None,
             args.resume, threads=args.threads, seed=args.seed)

if __name__ == "__main__":
    main()

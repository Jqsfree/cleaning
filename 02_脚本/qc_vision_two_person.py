#!/usr/bin/env python3
"""
视频画面质检脚本 — mqdefault + Qwen VL API
用法:
  python3 qc_vision.py input.csv --sample 100
  python3 qc_vision.py input.csv --resume -t 8
"""

import csv, json, os, re, sys, time, base64, signal, threading, random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# ============================================================
CONFIG = {
    "api_key": "sk-8a0564dcc1c64e35ba2ef70039af2864",
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "qwen-vl-max",
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

# ============================================================
# L0: 标题硬规则
# ============================================================

ANIME_PATTERNS = [
    r'\banime\b', r'\bcartoon\b', r'\u30de\u30f3\u30ac', r'\u30a2\u30cb\u30e1',
    r'\banimated\b', r'\b3d\s*animation\b', r'\bmanga\b',
    r'\bpixar\b', r'\btoon\b', r'\bstop\s*motion\b',
    r'\bdonghua\b', r'\uc560\ub2c8', r'\ub9cc\ud654',
]

MUSIC_PATTERNS = [
    r'\bofficial\s+(music\s+)?video\b',
    r'\bmusic\s+video\b', r'\blyric\b', r'\bofficial\s+audio\b',
    r'\bvevo\b', r'\bmtv\b', r'\bconcert\b',
    r'\blive\s+performance\b',
    r'\bMV\b', r'\bM/V\b', r'\[MV\]', r'\(MV\)', r'[\uff08\uff09(MV\uff09\uff09]',
]

SOLO_PATTERNS = [
    r'\bgaming\b', r'\bgameplay\b', r"\blet's\s+play\b",
    r'\bwalkthrough\b', r'\btutorial\b', r'\bhow[\s-]to\b',
    r'\bunboxing\b', r'\breview\b', r'\bproduct\s+(demo|review)\b',
    r'\bprank\b', r'\bchallenge\b', r'\bshorts?\b', r'#shorts',
    r'\basmr\b', r'\bmeditation\b', r'\bworkout\b',
    r'\bcooking\b', r'\brecipe\b', r'\broutine\b',
    r'\bcompilation\b', r'\btrailer\b', r'\bteaser\b',
    r'\bvlog\b', r'\b24/7\b',
]

DIALOGUE_PATTERNS = [
    r'\bpodcast\b', r'\bepisode\b', r'\bep[\.\s]?\d+',
    r'\binterview\b', r'\binterviews?\b', r'\bwith\s', r'\bft\.?\b', r'\bfeaturing\b',
    r'\bvs\.?\b', r'\bversus\b', r'\bconversation\b',
    r'\bdialogue\b', r'\bdiscussion\b', r'\bdebate\b',
    r'\btalk\s+show\b', r'\bpanel\b', r'\broundtable\b',
    r'\bguest\b', r'\bhost\b', r'\bchat\b', r'\bfireside\b',
    r'\bspeaks?\s+(with|to|about)\b', r'\btalks?\s+(with|to|about)\b',
    r'\bqa\b', r'\bq\s*&\s*a\b',
    '\u5c08\u8a2a', '\u8a2a\u8ac7', '\u5c0d\u8ac7', '\u5c0d\u8a71',
    '\u8bbf\u8c08', '\u5bf9\u8bdd', '\u9762\u5bf9\u9762',
    '\ub300\ub2f4', '\uc778\ud130\ubdf0', '\ud1a0\ud06c',
]

def title_classify(title):
    t = (title or "").lower()
    for p in ANIME_PATTERNS:
        if re.search(p, t):
            return "fail", "anime/cartoon"
    for p in MUSIC_PATTERNS:
        if re.search(p, t):
            return "fail", "music/mtv"
    for p in SOLO_PATTERNS:
        if re.search(p, t):
            return "uncertain", "solo/vlog_pattern"
    score = 0
    for p in DIALOGUE_PATTERNS:
        if re.search(p, t, re.I):
            score += 1
    if score >= 2:
        return "pass", f"dialogue:{score}"
    if score == 1:
        return "maybe", "weak_dialogue"
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
    "You are a classifier. Look at this YouTube video thumbnail. "
    "Answer exactly ONE WORD: YES or NO.\n\n"
    "YES = 2 REAL HUMAN BEINGS (not cartoon/anime/3D) appear together "
    "in this frame. They look like they are in a conversation \u2014 podcast, "
    "interview, talk show, debate, panel discussion, face-to-face chat.\n\n"
    "NO = ANY of these:\n"
    "- 0 or 1 person\n"
    "- 3 or more people (only exactly 2 is YES)\n"
    "- Cartoon, anime, 3D animation, manga\n"
    "- Big text on solid background (text cover)\n"
    "- Music stage, concert, music video\n"
    "- Product, animal, landscape, abstract\n"
    "- Movie poster with rendered characters\n\n"
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
    try:
        time.sleep(random.uniform(0, CONFIG["api_jitter_max"]))
        resp = requests.post(
            f"{CONFIG['api_base']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=CONFIG["api_timeout"],
        )
        if resp.status_code == 429:
            return {"result": "error", "raw": "rate_limited_429"}
        if resp.status_code != 200:
            return {"result": "error", "raw": f"HTTP {resp.status_code}: {resp.text[:100]}"}
        data = resp.json()
        answer = data["choices"][0]["message"]["content"].strip().upper()
        if answer.startswith("YES"):
            return {"result": "pass", "raw": answer}
        elif answer.startswith("NO"):
            return {"result": "fail", "raw": answer}
        elif "YES" in answer:
            return {"result": "pass", "raw": answer}
        else:
            return {"result": "fail", "raw": answer}
    except requests.Timeout:
        return {"result": "error", "raw": "timeout"}
    except Exception as e:
        return {"result": "error", "raw": str(e)[:100]}


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

def check_one(video_id, title="", duration_str=""):
    try:
        dur = float(duration_str) if duration_str else 0
    except:
        dur = 0
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
    for _ in range(CONFIG["max_retries"]):
        try:
            result = check_one(vid, title, dur_str)
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

def batch_qc(input_csv, output_csv=None, sample=None, resume=False, threads=4):
    if output_csv is None:
        output_csv = input_csv.replace(".csv", "_qc.csv")

    # --- 加载输入 ---
    print("[加载] 读取输入...")
    with open(input_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    N = len(all_rows)
    print(f"[输入] {N:,} 条")

    if sample and sample < N:
        import random as rnd
        rnd.seed(42)
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
    p.add_argument("--delay", type=float, default=0.05)
    p.add_argument("--model", default="qwen-vl-plus",
                   choices=["qwen-vl-max", "qwen-vl-plus", "qwen3-vl-plus", "qwen-vl-flash"])
    p.add_argument("--local", action="store_true", help="使用本地 Ollama qwen3-vl:8b")
    p.add_argument("--local-model", default="qwen3-vl:8b")
    args = p.parse_args()

    CONFIG["delay"] = args.delay
    CONFIG["model"] = args.model
    CONFIG["threads"] = args.threads
    CONFIG["use_local"] = args.local
    if args.local:
        CONFIG["local_model"] = args.local_model

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
             args.resume, threads=args.threads)

if __name__ == "__main__":
    main()

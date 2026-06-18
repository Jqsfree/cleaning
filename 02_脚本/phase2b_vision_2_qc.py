#!/usr/bin/env python3
"""
phase2b_vision_qc.py — 美妆视频 Storyboard 视觉质检

流程：
  1. yt-dlp 提取视频 info（不下载视频），拿到 storyboard mhtml URL（含签名）
  2. 下载最高分辨率 storyboard 拼图（一张大图，含若干小帧拼在一起）
  3. 切割出均匀分布的帧（默认取 6 帧）
  4. 调通义千问视觉模型质检三个维度

质检维度：
  Q1: 单人给自己化妆
  Q2: 讲解化妆步骤
  Q3: 完整人物/头部始终可见

依赖：
  pip install yt-dlp pandas Pillow tqdm openai

用法:
  python3 phase2b_vision_qc.py input.parquet -o ./output/ -w 4
  python3 phase2b_vision_qc.py input.parquet -w 4 --frames 8 --model qwen2.5-vl-7b-instruct
"""

import sys, os, json, time, argparse, shutil, random, re
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request

import pandas as pd
from PIL import Image

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────

DEFAULT_MODEL    = "qwen3-vl-flash"
DEFAULT_WORKERS  = 2
DEFAULT_FRAMES   = 6          # 从 storyboard 均匀取几帧送给模型
MAX_RETRIES      = 3
CHECKPOINT_EVERY = 50
FRAME_MAX_SIZE   = (320, 180) # 单帧压缩尺寸（storyboard 原图分辨率已经较低）
IMAGE_QUALITY    = 82

# yt-dlp 优先选哪个 storyboard 级别（sb0=最高质量，sb3=最低）
# 美妆质检建议 sb1（160x90）或 sb0（320x180），足够看清脸部
SB_PREFER_ORDER  = ["sb0", "sb1", "sb2", "sb3"]

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"

# ─────────────────────────────────────────────
# 提示词
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
你是严谨的美妆教程视觉审核员。根据视频抽帧画面，判断该视频是否属于单人在给自己化妆美妆的教程内容。仅输出 T 或 F，禁止任何解释。"""

USER_PROMPT_TMPL = """\
以下是从视频中均匀抽取的 {n} 帧画面（覆盖视频开头到结尾），请综合判断：

符合通过（T）:
    - 单人在给自己化妆美妆（不是给别人化）
    - 人物在讲解化妆美妆步骤（口述或字幕均可）
    - 始终有完整人脸/头部可见（不遮挡、不缺失）
非美妆教程（F）:
    - 无人脸/无真人/多人/给别人化/产品测评/无声化妆/美甲/发型
    
    严格按照要求输出，仅输出 T 或 F，禁止任何解释。"""



# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def fetch_url(url: str, timeout: int = 15) -> bytes:
    """带 UA 的 HTTP GET，返回 bytes"""
    req = Request(url, headers={"User-Agent": UA, "Referer": "https://www.youtube.com/"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ─────────────────────────────────────────────
# Storyboard 核心逻辑
# ─────────────────────────────────────────────

def get_storyboard_info(video_id: str, cookies_file: str | None = None) -> dict | None:
    """
    用 yt-dlp 提取 storyboard 信息，返回：
    {
        "sheet_urls": [...],   # 每张拼图的完整 URL（含签名）
        "cols": int,           # 每张拼图的列数
        "rows": int,           # 每张拼图的行数
        "frame_w": int,        # 单帧宽度 px
        "frame_h": int,        # 单帧高度 px
        "total_frames": int,   # 全部帧总数
    }
    失败返回 None
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("请先安装 yt-dlp：pip install yt-dlp")

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"skip": ["hls", "dash"]}},
    }
    if cookies_file and os.path.exists(cookies_file):
        ydl_opts["cookiefile"] = cookies_file

    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        log(f"  yt-dlp 失败 [{video_id}]: {e}")
        return None

    formats = info.get("formats", [])
    sb_formats = {f["format_id"]: f for f in formats if f.get("format_note") == "storyboard"}

    if not sb_formats:
        log(f"  [{video_id}] 无 storyboard 格式")
        return None

    # 按优先级选最佳质量
    chosen = None
    for pref in SB_PREFER_ORDER:
        if pref in sb_formats:
            chosen = sb_formats[pref]
            break
    if chosen is None:
        chosen = list(sb_formats.values())[0]

    rows      = chosen.get("rows", 10)
    cols      = chosen.get("columns", 10)
    frame_w   = chosen.get("width", 160)
    frame_h   = chosen.get("height", 90)
    fragments = chosen.get("fragments", [])

    if not fragments:
        # 有些格式只有顶层 url 没有 fragments
        top_url = chosen.get("url", "")
        if top_url:
            fragments = [{"url": top_url}]
        else:
            log(f"  [{video_id}] storyboard fragments 为空")
            return None

    sheet_urls   = [frag["url"] for frag in fragments]
    total_frames = rows * cols * len(sheet_urls)

    log(f"  [{video_id}] storyboard={chosen['format_id']} "
        f"{frame_w}x{frame_h}/帧 {cols}列x{rows}行 x{len(sheet_urls)}张 = {total_frames}帧")

    return {
        "sheet_urls":   sheet_urls,
        "cols":         cols,
        "rows":         rows,
        "frame_w":      frame_w,
        "frame_h":      frame_h,
        "total_frames": total_frames,
    }


def crop_frames_from_sheet(sheet_bytes: bytes, cols: int, rows: int,
                            frame_w: int, frame_h: int) -> list[Image.Image]:
    """把一张 storyboard 拼图切割成单帧列表（按行列顺序）"""
    sheet = Image.open(BytesIO(sheet_bytes)).convert("RGB")
    frames = []
    for r in range(rows):
        for c in range(cols):
            x = c * frame_w
            y = r * frame_h
            # 防止拼图尺寸不足（最后一行可能不完整）
            if x + frame_w > sheet.width or y + frame_h > sheet.height:
                break
            frame = sheet.crop((x, y, x + frame_w, y + frame_h))
            frames.append(frame)
    return frames


def fetch_storyboard_frames(sb_info: dict, n_frames: int = DEFAULT_FRAMES) -> list[Image.Image]:
    """
    下载 storyboard 拼图，均匀采样 n_frames 帧返回。
    优先只下载覆盖采样点所需的那几张拼图（节省带宽）。
    """
    sheet_urls   = sb_info["sheet_urls"]
    cols         = sb_info["cols"]
    rows         = sb_info["rows"]
    frame_w      = sb_info["frame_w"]
    frame_h      = sb_info["frame_h"]
    total_frames = sb_info["total_frames"]
    frames_per_sheet = cols * rows

    # 均匀采样的全局帧索引
    if total_frames <= n_frames:
        sample_indices = list(range(total_frames))
    else:
        step = total_frames / n_frames
        sample_indices = [int(i * step + step / 2) for i in range(n_frames)]

    # 算出需要哪些拼图 sheet
    needed_sheets = sorted(set(idx // frames_per_sheet for idx in sample_indices))

    # 下载需要的拼图并切帧
    all_frames: dict[int, Image.Image] = {}
    for sheet_idx in needed_sheets:
        if sheet_idx >= len(sheet_urls):
            break
        try:
            data = fetch_url(sheet_urls[sheet_idx])
            sheet_frames = crop_frames_from_sheet(data, cols, rows, frame_w, frame_h)
            base = sheet_idx * frames_per_sheet
            for local_i, img in enumerate(sheet_frames):
                all_frames[base + local_i] = img
        except Exception as e:
            log(f"    下载 sheet[{sheet_idx}] 失败: {e}")

    # 按采样顺序收集，缩略
    result = []
    for idx in sample_indices:
        if idx in all_frames:
            img = all_frames[idx].copy()
            img.thumbnail(FRAME_MAX_SIZE, Image.LANCZOS)
            result.append(img)

    return result


# ─────────────────────────────────────────────
# 模型调用
# ─────────────────────────────────────────────

def image_to_b64(img: Image.Image) -> str:
    import base64
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=IMAGE_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


def call_vision(client, video_id: str, model: str,
                n_frames: int = DEFAULT_FRAMES,
                cookies_file: str | None = None) -> tuple[dict | None, str]:
    """
    完整流程：yt-dlp 拿 storyboard → 下载切帧 → 调模型
    返回 (result_dict, error_str)
    """
    # Step1: 拿 storyboard 信息
    sb_info = get_storyboard_info(video_id, cookies_file=cookies_file)
    if sb_info is None:
        return None, "storyboard_not_found"

    # Step2: 下载并切帧
    frames = fetch_storyboard_frames(sb_info, n_frames=n_frames)
    if len(frames) == 0:
        return None, "storyboard_download_failed"

    log(f"  [{video_id}] 实际取得 {len(frames)} 帧，调模型...")

    # Step3: 构建多图消息
    content = []
    for img in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_to_b64(img)}"},
        })
    content.append({"type": "text", "text": USER_PROMPT_TMPL.format(n=len(frames))})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": content},
    ]

    # Step4: 调模型，带重试
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip().upper()
            if "T" in raw and "F" not in raw:
                return {"overall": True, "reason": raw[:20]}, ""
            elif "F" in raw and "T" not in raw:
                return {"overall": False, "reason": raw[:20]}, ""
            else:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1 + random.uniform(0, 0.5))
                    continue
                return None, f"invalid_response:{raw[:50]}"

        except Exception as e:
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str:
                wait = 2 ** attempt + random.uniform(0, 1)
                log(f"  [{video_id}] rate_limit，等待 {wait:.1f}s...")
                time.sleep(wait)
                if attempt < MAX_RETRIES - 1:
                    continue
            return None, f"api_error:{type(e).__name__}:{str(e)[:80]}"

    return None, "max_retries_exceeded"


# ─────────────────────────────────────────────
# IO 工具
# ─────────────────────────────────────────────

def atomic_write(df: pd.DataFrame, target_path: str):
    tmp = target_path + ".tmp"
    ext = os.path.splitext(target_path)[1].lower()
    if ext == ".parquet":
        df.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, target_path)


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="美妆视频 Storyboard 视觉质检")
    parser.add_argument("input",          help="输入 parquet/csv（需含 video_id 列）")
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("-m", "--model",  default=DEFAULT_MODEL,
                        help=f"模型名（默认 {DEFAULT_MODEL}）")
    parser.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS,
                        help="并发数（默认 2，受 yt-dlp 和 API 双重限制）")
    parser.add_argument("--frames",  type=int, default=DEFAULT_FRAMES,
                        help=f"每视频取几帧（默认 {DEFAULT_FRAMES}）")
    parser.add_argument("--cookies", default=None,
                        help="cookies 文件路径（Netscape 格式 .txt），防止 YouTube 风控")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="最多处理 N 行（0=全部，建议先试 5）")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force",   action="store_true", help="清除已有结果重新跑")
    args = parser.parse_args()

    # cookies 文件校验
    if args.cookies:
        if not os.path.exists(args.cookies):
            print(f"[ERROR] cookies 文件不存在: {args.cookies}")
            sys.exit(1)
        log(f"使用 cookies: {args.cookies}")

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(output_dir, exist_ok=True)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key and not args.dry_run:
        print("[ERROR] 未设置 DASHSCOPE_API_KEY")
        sys.exit(1)

    from openai import OpenAI
    client = None if args.dry_run else OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    t0     = time.perf_counter()
    run_id = make_run_id()

    # 读数据
    log(f"读取: {args.input}")
    ext = os.path.splitext(args.input)[1].lower()
    df  = (pd.read_parquet(args.input) if ext == ".parquet"
           else pd.read_csv(args.input, dtype=str, low_memory=False)).fillna("").astype(str)
    log(f"  总行数: {len(df):,}")

    # 备份
    bak = args.input + f".bak_{run_id}"
    if not os.path.exists(bak):
        shutil.copy2(args.input, bak)
        log(f"已备份: {bak}")

    # 初始化 QC 列
    qc_cols = ["qc_vision_result", "qc_vision_model", "qc_vision_run_id",
               "qc_vision_error_reason"]
    for col in qc_cols:
        if col not in df.columns:
            df[col] = ""
    if args.force:
        for col in qc_cols:
            df[col] = ""

    # 待处理
    pending_mask = df["qc_vision_result"].isin(["", "ERROR"]) | df["qc_vision_result"].isna()
    pending_idx  = df[pending_mask].index.tolist()
    if args.max_rows > 0 and len(pending_idx) > args.max_rows:
        pending_idx = pending_idx[:args.max_rows]
    log(f"待处理: {len(pending_idx):,} / {len(df):,}")

    if len(pending_idx) == 0:
        log("全部已质检，无需重跑。")
        return

    if args.dry_run:
        log(f"[dry-run] 将处理 {len(pending_idx)} 条，模型={args.model}，帧数={args.frames}")
        return

    # 并发执行
    completed = 0
    last_ckpt = time.time()
    n_ok = n_err = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(call_vision, client,
                            str(df.at[idx, "video_id"]),
                            args.model, args.frames,
                            args.cookies): idx
            for idx in pending_idx
        }

        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(pending_idx), desc="Storyboard QC")
        except ImportError:
            pbar = None

        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                result, error = future.result()
            except Exception as ex:
                result, error = None, f"future_exception:{type(ex).__name__}"

            if error:
                df.at[idx, "qc_vision_result"]       = "ERROR"
                df.at[idx, "qc_vision_error_reason"] = error
                n_err += 1
            elif result:
                overall = "T" if result.get("overall") else "F"
                df.at[idx, "qc_vision_result"] = overall
                df.at[idx, "qc_vision_error_reason"] = ""
                n_ok += 1
            else:
                df.at[idx, "qc_vision_result"]       = "ERROR"
                df.at[idx, "qc_vision_error_reason"] = "unknown_error"
                n_err += 1

            df.at[idx, "qc_vision_model"]  = args.model
            df.at[idx, "qc_vision_run_id"] = run_id
            completed += 1

            if pbar:
                pbar.update(1)
                pbar.set_postfix(ok=n_ok, err=n_err)

            now = time.time()
            if completed % CHECKPOINT_EVERY == 0 or (now - last_ckpt) >= 60:
                atomic_write(df, args.input)
                log(f"  checkpoint ✓ {completed:,} 条 → {args.input}")
                last_ckpt = now

        if pbar:
            pbar.close()

    # 最终写入
    atomic_write(df, args.input)

    elapsed = time.perf_counter() - t0
    print()
    print("=" * 60)
    print("  Storyboard 视觉 QC 完成")
    print("=" * 60)
    print(f"  run_id:    {run_id}")
    print(f"  模型:      {args.model}")
    print(f"  每视频帧数: {args.frames}")
    print(f"  处理:      {completed:,}  (OK={n_ok:,}  ERR={n_err:,})")
    print(f"  耗时:      {elapsed:.0f}s  ({elapsed/max(completed,1):.1f}s/条)")
    print(f"  回写至:    {args.input}")
    print("=" * 60)


if __name__ == "__main__":
    main()

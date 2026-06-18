#!/usr/bin/env python3
"""
phase2b_vision_qc.py — 美妆视频封面视觉质检

用 YouTube 封面图 + 通义千问视觉模型，判断：
  Q1: 单人给自己化妆
  Q2: 讲解化妆步骤
  Q3: 完整人物/头部始终可见

用法:
  python3 phase2b_vision_qc.py data/runs/beauty/002_audit/xxx_qc.parquet -o data/runs/beauty/002_audit/ -w 10
"""

import sys, os, json, time, argparse, shutil, random
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODEL   = "qwen-vl-plus"
DEFAULT_WORKERS = 2
MAX_RETRIES     = 3
CHECKPOINT_EVERY = 50

# YouTube 自动缩略图（开头/中间/结尾）
THUMBNAIL_URLS = [
    "https://i.ytimg.com/vi/{video_id}/1.jpg",
    "https://i.ytimg.com/vi/{video_id}/2.jpg",
    "https://i.ytimg.com/vi/{video_id}/3.jpg",
]
FRAME_LABELS = ["开头", "中间", "结尾"]
IMAGE_MAX_SIZE = (640, 360)
IMAGE_QUALITY = 85


SYSTEM_PROMPT = """\
你是一名专业的美妆教程视频审核员，根据视频封面图判断内容质量。
请只返回 JSON，不要添加任何额外说明。"""

USER_PROMPT = """\
以下是从视频开头、中间、结尾抽取的 3 帧画面，请综合判断：

Q1（单人自化妆）：是否只有一人，且该人在给自己化妆？
Q2（讲解步骤）：是否可能为讲解化妆步骤的教程视频？
Q3（完整人物存在）：三帧中是否始终有完整人脸/头部可见？

返回严格 JSON（无其他内容）：
{"Q1": true或false, "Q2": true或false, "Q3": true或false, "overall": true或false, "reason": "10字内说明"}"""


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def fetch_thumbnails(video_id: str) -> list[Image.Image]:
    """下载 YouTube 开头/中间/结尾三帧缩略图，失败返回空列表"""
    frames = []
    for url_tpl in THUMBNAIL_URLS:
        try:
            url = url_tpl.format(video_id=video_id)
            with urlopen(url, timeout=10) as resp:
                img = Image.open(BytesIO(resp.read()))
                img.thumbnail(IMAGE_MAX_SIZE, Image.LANCZOS)
                frames.append(img)
        except Exception:
            frames.append(None)
    return frames


def image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=IMAGE_QUALITY)
    import base64
    return base64.b64encode(buf.getvalue()).decode()


def call_vision(client, video_id: str, model: str) -> tuple[dict | None, str]:
    """下载 3 帧 → 调 vision 模型 → 返回 (result_dict, error_reason)"""
    frames = fetch_thumbnails(video_id)
    valid_frames = [f for f in frames if f is not None]
    if len(valid_frames) == 0:
        return None, "all_thumbnails_download_failed"

    # 构建多图消息
    content = []
    for i, img in enumerate(frames):
        label = FRAME_LABELS[i] if i < len(FRAME_LABELS) else f"帧{i+1}"
        if img is not None:
            content.append({"image": f"data:image/jpeg;base64,{image_to_b64(img)}"})
        else:
            content.append({"text": f"[{label}帧下载失败]"})

    content.append({"text": USER_PROMPT})

    messages = [
        {"role": "system", "content": [{"text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
            result = json.loads(raw)
            return result, ""

        except json.JSONDecodeError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1 + random.uniform(0, 0.5))
                continue
            return None, f"json_parse_error:{raw[:50]}"

        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep(2 ** attempt + random.uniform(0, 1))
                if attempt < MAX_RETRIES - 1:
                    continue
            return None, f"api_error:{type(e).__name__}:{str(e)[:80]}"

    return None, "max_retries_exceeded"


def atomic_write(df: pd.DataFrame, target_path: str):
    tmp = target_path + ".tmp"
    ext = os.path.splitext(target_path)[1].lower()
    if ext == ".parquet":
        df.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, target_path)


def main():
    parser = argparse.ArgumentParser(description="美妆视频封面视觉质检")
    parser.add_argument("input", help="输入 parquet/csv 文件（需含 video_id 列）")
    parser.add_argument("-o", "--output-dir", default=None, help="输出目录")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="清除已有结果重新跑")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(output_dir, exist_ok=True)

    # API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key and not args.dry_run:
        print("[ERROR] 未设置 DASHSCOPE_API_KEY")
        sys.exit(1)

    from openai import OpenAI
    client = None if args.dry_run else OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    t0 = time.perf_counter()
    run_id = make_run_id()

    # 读数据
    log(f"读取: {args.input}")
    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(args.input).fillna("").astype(str)
    else:
        df = pd.read_csv(args.input, dtype=str, low_memory=False).fillna("")
    log(f"  样本行数: {len(df):,}")

    # 备份
    bak = args.input + f".bak_{run_id}"
    if not os.path.exists(bak):
        shutil.copy2(args.input, bak)
        log(f"已备份: {bak}")

    # 初始化 QC 列
    for col in ["qv_q1_self", "qv_q2_explain", "qv_q3_fullface",
                "qv_overall", "qv_reason", "qv_model", "qv_run_id", "qv_error"]:
        if col not in df.columns:
            df[col] = ""

    if args.force:
        for col in ["qv_q1_self", "qv_q2_explain", "qv_q3_fullface",
                     "qv_overall", "qv_reason", "qv_model", "qv_run_id", "qv_error"]:
            if col in df.columns:
                df[col] = ""

    # 待处理
    pending_mask = df["qv_overall"].isin(["", "ERROR"]) | df["qv_overall"].isna()
    pending_idx = df[pending_mask].index.tolist()
    log(f"待处理: {len(pending_idx):,} / {len(df):,}")

    if len(pending_idx) == 0:
        log("全部已质检，无需重跑。")
        return

    if args.dry_run:
        log("dry-run 结束。")
        return

    # 并发
    completed = 0
    last_checkpoint = time.time()
    n_ok = 0
    n_err = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {}
        for idx in pending_idx:
            vid = str(df.at[idx, "video_id"])
            future_map[executor.submit(call_vision, client, vid, args.model)] = idx

        from tqdm import tqdm
        with tqdm(total=len(pending_idx), desc="视觉 QC") as pbar:
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    result, error = future.result()
                except Exception as ex:
                    result, error = None, f"future_exception:{type(ex).__name__}"

                if error:
                    df.at[idx, "qv_overall"] = "ERROR"
                    df.at[idx, "qv_error"] = error
                    n_err += 1
                elif result:
                    df.at[idx, "qv_q1_self"] = str(result.get("Q1", ""))
                    df.at[idx, "qv_q2_explain"] = str(result.get("Q2", ""))
                    df.at[idx, "qv_q3_fullface"] = str(result.get("Q3", ""))
                    df.at[idx, "qv_overall"] = str(result.get("overall", ""))
                    df.at[idx, "qv_reason"] = str(result.get("reason", ""))
                    n_ok += 1
                else:
                    df.at[idx, "qv_overall"] = "ERROR"
                    df.at[idx, "qv_error"] = "unknown_error"
                    n_err += 1

                df.at[idx, "qv_model"] = args.model
                df.at[idx, "qv_run_id"] = run_id

                completed += 1
                pbar.update(1)

                now = time.time()
                if completed % CHECKPOINT_EVERY == 0 or (now - last_checkpoint) >= 60:
                    atomic_write(df, args.input)
                    log(f"  checkpoint ✓  {completed:,}  → {args.input}")
                    last_checkpoint = now

    # 最终写入
    atomic_write(df, args.input)

    elapsed = time.perf_counter() - t0
    print()
    print("=" * 60)
    print("  视觉 QC 完成")
    print("=" * 60)
    print(f"  run_id:    {run_id}")
    print(f"  模型:      {args.model}")
    print(f"  处理:      {completed:,} (OK={n_ok:,} ERR={n_err:,})")
    print(f"  耗时:      {elapsed:.0f}s")
    print(f"  回写至:    {args.input}")
    print("=" * 60)


if __name__ == "__main__":
    main()

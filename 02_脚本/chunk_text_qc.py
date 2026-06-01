#!/usr/bin/env python3
"""
chunk_text_qc.py — Step 2: 全量文本 LLM 质检（单 chunk）

对已分类的 chunk CSV，逐行调用 Qwen 文生模型判断是否体育赛事解说。
输入: Step 1 产出的 classified CSV（含 _p2_bucket 列）
输出: 同 CSV 追加 qc_text_result / qc_text_model 列 + summary JSON

冲突检测已禁用 — _detect_conflict() 统一返回兜底值。
用法:
  conda activate data_cleaning
  python3 chunk_text_qc.py /path/to/classified.csv
  python3 chunk_text_qc.py /path/to/classified.csv -w 20        # 20 并发
  python3 chunk_text_qc.py /path/to/classified.csv -m qwen-plus # 换模型
  python3 chunk_text_qc.py /path/to/classified.csv --dry-run    # 只统计不调用
"""

import sys, os, time, json, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# 前置检查
try:
    import pandas as pd
    from tqdm import tqdm
    from openai import OpenAI, APIConnectionError, RateLimitError
except ImportError as e:
    print(f"[ERROR] 缺少依赖: {e}")
    print("  conda activate data_cleaning")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
# 配置默认值
# ══════════════════════════════════════════════════════════════

DEFAULT_MODEL     = "qwen3.5-flash"
DEFAULT_WORKERS   = 20
CHECKPOINT_EVERY  = 200       # 每 N 条写一次中间文件
MAX_RETRIES       = 3
API_BASE_URL      = "https://dashscope.aliyuncs.com/compatible-mode/v1"

SYSTEM_PROMPT = """\
你是严谨的体育内容审核员。你的任务是判断一个 YouTube 视频是否属于体育赛事解说。

体育赛事解说包括:
- 体育比赛直播/录播/回放
- 有解说/解说的比赛视频
- 完整的赛事转播

非体育赛事解说包括:
- 游戏实况/电竞（即使标题有体育词汇）
- 音乐视频/MV
- 综艺/娱乐/电影/电视剧
- 教学/教程/vlog
- 新闻/采访/发布会
- 集锦/highlights（非完整比赛）
- 播客/谈话节目

严格按照要求输出，禁止任何解释。"""


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════
# LLM 调用
# ══════════════════════════════════════════════════════════════

def _build_user_prompt(row) -> str:
    """从一行数据构建 user prompt。"""
    title   = str(row.get("title", "") or "")
    channel = str(row.get("channel", "") or "")
    keyword = str(row.get("keyword", "") or "")
    desc    = str(row.get("description", "") or "")[:200]  # 截断长描述

    parts = []
    if keyword:
        parts.append(f"搜索关键词: {keyword}")
    if title:
        parts.append(f"视频标题: {title}")
    if channel:
        parts.append(f"频道名称: {channel}")
    if desc:
        parts.append(f"视频简介: {desc}")

    context = "\n".join(parts)
    return f"{context}\n\n该视频是否属于体育赛事解说？仅输出 T 或 F。"


def check_one(client, row, model: str) -> tuple:
    """
    对单行调用 LLM 判断。

    返回 (result_str, model_used)
      result_str: "T" | "F" | "ERROR"
    """
    user_prompt = _build_user_prompt(row)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=5,
            )
            raw = resp.choices[0].message.content.strip().upper()
            if "T" in raw:
                return ("T", model)
            elif "F" in raw:
                return ("F", model)
            else:
                # 模型返回了非预期格式，再试
                if attempt < MAX_RETRIES - 1:
                    time.sleep(0.5)
                    continue
                return ("ERROR", model)

        except RateLimitError:
            wait = 2 ** attempt
            time.sleep(wait)
        except APIConnectionError:
            time.sleep(1)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return ("ERROR", model)
            time.sleep(1)

    return ("ERROR", model)


# ══════════════════════════════════════════════════════════════
# 冲突检测
# ══════════════════════════════════════════════════════════════

def _detect_conflict(row) -> str:
    """
    比较规则分桶 (_p2_bucket) 与文本 QC (qc_text_result)。
    返回冲突标签或空字符串。
    冲突检测已禁用，统一返回兜底值。
    """
    return "other"


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def run_text_qc(input_csv: str, output_dir: str, model: str = DEFAULT_MODEL,
                workers: int = DEFAULT_WORKERS, dry_run: bool = False) -> dict:
    """
    对已分类 chunk 做全量文本 QC。

    返回 summary dict。
    """
    t0 = time.perf_counter()

    # ── API Key ────────────────────────────────────────────
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key and not dry_run:
        print("[ERROR] 未设置 DASHSCOPE_API_KEY 环境变量")
        print("  export DASHSCOPE_API_KEY='你的key'")
        sys.exit(1)

    client = None if dry_run else OpenAI(api_key=api_key, base_url=API_BASE_URL)

    # ── 读取 ──────────────────────────────────────────────
    log(f"读取: {input_csv}")
    df = pd.read_csv(input_csv, dtype=str, low_memory=False).fillna("")
    n_total = len(df)
    log(f"  行数: {n_total:,}")

    # ── 初始化结果列 ──────────────────────────────────────
    if "qc_text_result" not in df.columns:
        df["qc_text_result"] = None
    if "qc_text_model" not in df.columns:
        df["qc_text_model"] = None
    # ── 找出待处理行 ──────────────────────────────────────
    pending_mask = df["qc_text_result"].isna() | (df["qc_text_result"] == "ERROR")
    pending_idx = df[pending_mask].index.tolist()
    n_pending = len(pending_idx)

    if n_pending == 0:
        log("全部已质检，无需重复。")
    else:
        log(f"待处理: {n_pending:,} / {n_total:,} ({n_pending/n_total*100:.1f}%)")

    # ── 输出路径 ──────────────────────────────────────────
    # 用输入文件 stem 做后缀，避免多次跑互相覆盖
    input_stem = os.path.splitext(os.path.basename(input_csv))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_csv    = os.path.join(output_dir, f"02_text_qc_{input_stem}.csv")
    summary_path  = os.path.join(output_dir, f"02_text_qc_{input_stem}_summary.json")

    # ── 并发 QC ───────────────────────────────────────────
    if n_pending > 0 and not dry_run:
        assert client is not None
        completed = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(check_one, client, df.loc[idx].to_dict(), model): idx
                for idx in pending_idx
            }

            with tqdm(total=n_pending, desc="文本 LLM 质检") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result_str, model_used = future.result()
                    except Exception:
                        result_str, model_used = "ERROR", model

                    df.at[idx, "qc_text_result"] = result_str
                    df.at[idx, "qc_text_model"]  = model_used
                    completed += 1
                    pbar.update(1)

                    # checkpoint
                    if completed % CHECKPOINT_EVERY == 0:
                        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
                        log(f"  checkpoint: {completed:,} / {n_pending:,}")

        # 最终写出
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        log(f"写出: {output_csv}")

    elif dry_run:
        log("dry-run 模式，跳过 LLM 调用。")

    # ── 统计 ──────────────────────────────────────────────
    elapsed = time.perf_counter() - t0

    t_count = int((df["qc_text_result"] == "T").sum())
    f_count = int((df["qc_text_result"] == "F").sum())
    err_count = int((df["qc_text_result"] == "ERROR").sum())
    pending_remain = int(df["qc_text_result"].isna().sum())

    summary = {
        "step": "text_qc",
        "model": model,
        "input": os.path.abspath(input_csv),
        "total_rows": n_total,
        "pending_before": n_pending,
        "elapsed_sec": round(elapsed, 1),
        "results": {
            "T": t_count, "T_pct": round(t_count / n_total * 100, 1),
            "F": f_count, "F_pct": round(f_count / n_total * 100, 1),
            "ERROR": err_count,
            "pending_after": pending_remain,
        },
        "outputs": {
            "qc_csv": output_csv,
            "summary_json": summary_path,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"写出摘要: {summary_path}")

    # ── 写出结果 CSV ────────────────────────────────────────
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    log(f"写出结果 CSV: {output_csv}")

    # ── 终端摘要 ──────────────────────────────────────────
    print()
    print("=" * 55)
    print(f"  Step 2 文本 QC — {os.path.basename(input_csv)}")
    print("=" * 55)
    print(f"  模型:             {model}")
    print(f"  并发:             {workers}")
    print(f"  耗时:             {elapsed:>10.1f}s")
    print(f"  {'─'*45}")
    print(f"  QC 结果:")
    print(f"    T (是体育):      {t_count:>10,}  ({t_count/n_total*100:5.1f}%)")
    print(f"    F (非体育):      {f_count:>10,}  ({f_count/n_total*100:5.1f}%)")
    print(f"    ERROR:           {err_count:>10,}")
    print("=" * 55)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: 全量文本 LLM 质检（单 chunk）"
    )
    parser.add_argument("input", help="Step 1 产出的 classified CSV")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="输出目录（默认与输入同目录）")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help=f"Qwen 模型名（默认 {DEFAULT_MODEL}）")
    parser.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"并发线程数（默认 {DEFAULT_WORKERS}）")
    parser.add_argument("--dry-run", action="store_true",
                        help="不调 API，仅统计")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(args.input)
    run_text_qc(args.input, output_dir, model=args.model,
                workers=args.workers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

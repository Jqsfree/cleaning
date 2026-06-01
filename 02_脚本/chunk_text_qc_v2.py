#!/usr/bin/env python3
"""
chunk_text_qc_v2.py — 文本 LLM 质检（重构版 v2）

核心变更（相对 v1）:
  1. 结果回写原始 CSV（原地修改，原子写保护）
  2. 运行前自动备份原文件
  3. 新增 qc_run_id / qc_error_reason 列（可追溯）
  4. 输出文件统一命名规范: {stem}_textqc_{YYYYMMDD}_{HHmmss}_{suffix}.ext
  5. checkpoint 改为原子写（先写 .tmp 再 os.replace）
  6. retry 加 jitter，ERROR 行记录原因

用法:
  python3 chunk_text_qc_v2.py input.csv
  python3 chunk_text_qc_v2.py input.csv -w 20
  python3 chunk_text_qc_v2.py input.csv -m qwen-plus
  python3 chunk_text_qc_v2.py input.csv --dry-run
"""

import sys, os, time, json, argparse, random, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    import pandas as pd
    from tqdm import tqdm
    from openai import OpenAI, APIConnectionError, RateLimitError
except ImportError as e:
    print(f"[ERROR] 缺少依赖: {e}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════

DEFAULT_MODEL    = "qwen3.5-flash"
DEFAULT_WORKERS  = 20
CHECKPOINT_EVERY = 500          # 改为每 500 条（或按时间间隔）
MAX_RETRIES      = 3
API_BASE_URL     = "https://dashscope.aliyuncs.com/compatible-mode/v1"

SYSTEM_PROMPT = """\
你是严谨的体育内容审核员。你的任务是判断一个视频是否属于体育赛事相关内容。

符合通过（T）:
- 体育比赛全场/节选/集锦/highlights（含官方频道、赛事组织发布的赛事高光）
- match, game, race, final, semi-final, quarter-final, championship, tournament
- vs, versus, round, set, period, heat, bout, shootout, overtime
- live stream, live broadcast, commentary, play-by-play, full replay, full coverage
- league, cup, open, olympic, world cup, diamond league, grand prix
- 各项体育运动名称: basketball, football, tennis, swimming, athletics, boxing 等
- 赛事组织/缩写: NBA, NFL, FIFA, ATP, WTA, WSL, NCAA, UFC 等
- 官方体育频道的赛事集锦/分析（如 MLB Highlights、NHL Highlights、Premier League Highlights 等）
- 体育比赛的慢动作回放、技术分析片段（只要主体是赛事画面）

非体育内容（应输出 F）:
- 游戏实况/电竞 (即使标题含体育词汇。注意: 体育模拟游戏如 FIFA/PGA 2K/NBA 2K 也属于电竞)
- 音乐/MV、综艺/娱乐、电影/剧集
- 教学/教程/vlog、新闻/采访/发布会
- 播客/谈话节目（即使话题是体育，但不是赛事画面）
- 纪录片/幕后花絮、器材评测、装备开箱
- 动漫/卡通、儿童内容
- 体育下注/博彩分析、DFS/Fantasy 讨论
- 非赛事画面：颁奖/开幕式、抗议/罢工、场馆建设、运动员场外生活

严格按照要求输出，仅输出 T 或 F，禁止任何解释。"""


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_run_id() -> str:
    """生成本次运行 ID，格式: YYYYMMDD_HHmmss"""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_output_stem(input_stem: str, run_id: str) -> str:
    """
    统一命名规范: {原文件名}_textqc_{run_id}
    例: sports_dataset_textqc_20260601_143022
    """
    return f"{input_stem}_textqc_{run_id}"


def backup_input(input_csv: str, run_id: str) -> str:
    """
    备份原文件: input.csv → input.csv.bak_20260601_143022
    只备份一次，文件存在则跳过。
    """
    bak_path = f"{input_csv}.bak_{run_id}"
    if not os.path.exists(bak_path):
        shutil.copy2(input_csv, bak_path)
        log(f"已备份原文件: {bak_path}")
    else:
        log(f"备份已存在，跳过: {bak_path}")
    return bak_path


def atomic_write_csv(df: pd.DataFrame, target_path: str):
    """
    原子写 CSV：先写 .tmp，再 os.replace → 避免写到一半崩溃损坏文件。
    """
    tmp_path = target_path + ".tmp"
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, target_path)


# ══════════════════════════════════════════════════════════════
# LLM 调用
# ══════════════════════════════════════════════════════════════

def _build_user_prompt(row: dict) -> str:
    title   = str(row.get("title",       ""))
    channel = str(row.get("channel",     ""))
    keyword = str(row.get("keyword",     ""))
    desc    = str(row.get("description", ""))[:200]

    parts = []
    if keyword: parts.append(f"搜索关键词: {keyword}")
    if title:   parts.append(f"视频标题: {title}")
    if channel: parts.append(f"频道名称: {channel}")
    if desc:    parts.append(f"视频简介: {desc}")

    return "\n".join(parts) + "\n\n分析该视频的内容是否属于体育赛事解说，仅输出结果 T or F  符合通过的关键词"


def check_one(client, row: dict, model: str) -> tuple[str, str, str]:
    """
    返回 (result, model_used, error_reason)
    result: "T" | "F" | "ERROR"
    error_reason: 正常时为 ""，出错时记录具体原因
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
                return ("T", model, "")
            elif "F" in raw:
                return ("F", model, "")
            else:
                # 模型返回了非 T/F 内容
                if attempt < MAX_RETRIES - 1:
                    time.sleep(0.5 + random.uniform(0, 0.5))   # jitter
                    continue
                return ("ERROR", model, f"invalid_response:{raw[:20]}")

        except RateLimitError as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            time.sleep(wait)
            if attempt == MAX_RETRIES - 1:
                return ("ERROR", model, f"rate_limit_error:{str(e)[:80]}")

        except APIConnectionError:
            time.sleep(1 + random.uniform(0, 0.5))
            if attempt == MAX_RETRIES - 1:
                return ("ERROR", model, "api_connection_error")

        except Exception as ex:
            time.sleep(1)
            if attempt == MAX_RETRIES - 1:
                return ("ERROR", model, f"exception:{type(ex).__name__}:{str(ex)[:80]}")

    return ("ERROR", model, "max_retries_exceeded")


# ══════════════════════════════════════════════════════════════
# 名单输出
# ══════════════════════════════════════════════════════════════

def _write_name_lists(df: pd.DataFrame, pass_path: str, fail_path: str, error_path: str):
    pass_ids  = df[df["qc_text_result"] == "T"]["video_id"].tolist()
    fail_ids  = df[df["qc_text_result"] == "F"]["video_id"].tolist()
    error_ids = df[df["qc_text_result"] == "ERROR"]["video_id"].tolist()

    for path, ids in [(pass_path, pass_ids), (fail_path, fail_ids), (error_path, error_ids)]:
        with open(path, "w") as f:
            for vid in ids:
                f.write(f"{vid}\n")

    log(f"通过名单:   {pass_path}  ({len(pass_ids):,} 条)")
    log(f"未通过名单: {fail_path}  ({len(fail_ids):,} 条)")
    if error_ids:
        log(f"错误名单:   {error_path}  ({len(error_ids):,} 条)")


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def run_text_qc(
    input_csv: str,
    output_dir: str,
    model: str = DEFAULT_MODEL,
    workers: int = DEFAULT_WORKERS,
    dry_run: bool = False,
) -> dict:

    t0     = time.perf_counter()
    run_id = make_run_id()

    # ── API Key ──────────────────────────────────────────────
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key and not dry_run:
        print("[ERROR] 未设置 DASHSCOPE_API_KEY 环境变量")
        sys.exit(1)
    client = None if dry_run else OpenAI(api_key=api_key, base_url=API_BASE_URL)

    # ── 读取 ─────────────────────────────────────────────────
    log(f"读取: {input_csv}")
    df = pd.read_csv(input_csv, dtype=str, low_memory=False).fillna("")
    n_total = len(df)
    safe_total = max(n_total, 1)  # 除零保护
    log(f"  行数: {n_total:,}")

    # ── 备份原文件 ───────────────────────────────────────────
    bak_path = backup_input(input_csv, run_id)

    # ── 初始化新增列 ─────────────────────────────────────────
    for col in ["qc_text_result", "qc_text_model", "qc_run_id", "qc_error_reason"]:
        if col not in df.columns:
            df[col] = ""

    # ── 找出待处理行 ─────────────────────────────────────────
    pending_mask = df["qc_text_result"].isin(["", "ERROR"]) | df["qc_text_result"].isna()
    pending_idx  = df[pending_mask].index.tolist()
    n_pending    = len(pending_idx)

    if n_pending == 0:
        log("全部已质检，无需重跑。")
    else:
        log(f"待处理: {n_pending:,} / {n_total:,}  ({n_pending/safe_total*100:.1f}%)")

    # ── 输出路径 ─────────────────────────────────────────────
    input_stem  = Path(input_csv).stem
    output_stem = make_output_stem(input_stem, run_id)
    os.makedirs(output_dir, exist_ok=True)

    # 结果回写到原始文件（同时在输出目录保留一份快照）
    snapshot_csv   = os.path.join(output_dir, f"{output_stem}.csv")
    summary_path   = os.path.join(output_dir, f"{output_stem}_summary.json")
    pass_list_path = os.path.join(output_dir, f"{output_stem}_pass.txt")
    fail_list_path = os.path.join(output_dir, f"{output_stem}_fail.txt")
    error_list_path= os.path.join(output_dir, f"{output_stem}_error.txt")

    # ── 并发 QC ──────────────────────────────────────────────
    if n_pending > 0 and not dry_run:
        completed = 0
        last_checkpoint = time.time()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(check_one, client, df.loc[idx].to_dict(), model): idx
                for idx in pending_idx
            }

            with tqdm(total=n_pending, desc="文本 LLM 质检") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result_str, model_used, error_reason = future.result()
                    except Exception as ex:
                        result_str, model_used, error_reason = "ERROR", model, f"future_exception:{type(ex).__name__}"

                    # 回写结果到 df
                    df.at[idx, "qc_text_result"]  = result_str
                    df.at[idx, "qc_text_model"]   = model_used
                    df.at[idx, "qc_run_id"]        = run_id
                    df.at[idx, "qc_error_reason"]  = error_reason

                    completed += 1
                    pbar.update(1)

                    # checkpoint：按条数 或 每 60 秒，原子写回原文件
                    now = time.time()
                    if completed % CHECKPOINT_EVERY == 0 or (now - last_checkpoint) >= 60:
                        atomic_write_csv(df, input_csv)
                        log(f"  checkpoint ✓  {completed:,} / {n_pending:,}  → {input_csv}")
                        last_checkpoint = now

    elif dry_run:
        log("dry-run 模式，跳过 LLM 调用。")

    # ── 原子写回原始文件 + 快照 ──────────────────────────────
    atomic_write_csv(df, input_csv)
    log(f"结果已回写原文件: {input_csv}")

    atomic_write_csv(df, snapshot_csv)
    log(f"快照已写出: {snapshot_csv}")

    # ── 名单 ─────────────────────────────────────────────────
    _write_name_lists(df, pass_list_path, fail_list_path, error_list_path)

    # ── 统计 ─────────────────────────────────────────────────
    elapsed   = time.perf_counter() - t0
    t_count   = int((df["qc_text_result"] == "T").sum())
    f_count   = int((df["qc_text_result"] == "F").sum())
    err_count = int((df["qc_text_result"] == "ERROR").sum())

    summary = {
        "step":           "text_qc_v2",
        "run_id":         run_id,
        "model":          model,
        "workers":        workers,
        "input":          os.path.abspath(input_csv),
        "backup":         bak_path,
        "total_rows":     n_total,
        "pending_before": n_pending,
        "elapsed_sec":    round(elapsed, 1),
        "results": {
            "T":     t_count,  "T_pct":   round(t_count / safe_total * 100, 1),
            "F":     f_count,  "F_pct":   round(f_count / safe_total * 100, 1),
            "ERROR": err_count,
        },
        "outputs": {
            "written_back_to": input_csv,
            "snapshot_csv":    snapshot_csv,
            "summary_json":    summary_path,
            "pass_list":       pass_list_path,
            "fail_list":       fail_list_path,
            "error_list":      error_list_path,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"写出摘要: {summary_path}")

    # ── 终端摘要 ─────────────────────────────────────────────
    print()
    print("=" * 62)
    print(f"  文本 QC v2 — {os.path.basename(input_csv)}")
    print("=" * 62)
    print(f"  run_id:   {run_id}")
    print(f"  模型:     {model}    并发: {workers}")
    print(f"  耗时:     {elapsed:>10.1f}s")
    print(f"  {'─'*54}")
    print(f"  QC 结果:")
    print(f"    T (是体育):  {t_count:>10,}  ({t_count/safe_total*100:5.1f}%)")
    print(f"    F (非体育):  {f_count:>10,}  ({f_count/safe_total*100:5.1f}%)")
    print(f"    ERROR:       {err_count:>10,}")
    print(f"  {'─'*54}")
    print(f"  备份:     {bak_path}")
    print(f"  回写至:   {input_csv}")
    print(f"  快照:     {snapshot_csv}")
    print("=" * 62)

    return summary


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="文本 LLM 质检 v2")
    parser.add_argument("input",        help="输入 CSV")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="输出目录（摘要/名单，默认与输入同目录）")
    parser.add_argument("-m", "--model",   default=DEFAULT_MODEL)
    parser.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dry-run",       action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    run_text_qc(args.input, output_dir, model=args.model,
                workers=args.workers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

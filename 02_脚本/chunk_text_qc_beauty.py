#!/usr/bin/env python3
"""
chunk_text_qc_beauty.py — 文本 LLM 质检（美妆版）

核心变更（相对 v2）:
  1. 结果回写原始 CSV（原地修改，原子写保护）
  2. 运行前自动备份原文件
  3. 新增 qc_run_id / qc_error_reason 列（可追溯）
  4. 输出文件统一命名规范: {stem}_textqc_{YYYYMMDD}_{HHmmss}_{suffix}.ext
  5. checkpoint 改为原子写（先写 .tmp 再 os.replace）
  6. retry 加 jitter，ERROR 行记录原因

用法:
  python3 chunk_text_qc_beauty.py input.csv
  python3 chunk_text_qc_beauty.py input.csv -w 20
  python3 chunk_text_qc_beauty.py input.csv -m qwen-plus
  python3 chunk_text_qc_beauty.py input.csv --dry-run
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
你是严谨的美妆/化妆教学内容审核员。你的任务是判断一个视频是否属于美妆教程/化妆相关内容。

符合通过（T）:
- 化妆教程: makeup tutorial, beauty tutorial, 化妆教程, 美妆教学
- 妆前妆后对比: before/after makeup, transformation
- 产品评测/开箱: makeup review, beauty haul, unboxing, swatch
- 护肤流程: skincare routine, 护肤步骤
- 发型/美发教程: hair tutorial, hairstyle
- 美甲教程: nail art tutorial, 美甲
- 彩妆技法: eyeshadow/eyeliner/lipstick/contour/foundation 等专项教学
- 个人化妆日常: GRWM (Get Ready With Me), makeup routine, 化妆日常
- 美妆博主发布的化妆相关视频
- 新娘妆/宴会妆/万圣节妆等特殊场合化妆
- 美妆产品试色/测评
- 一人面对镜头完成的化妆内容（真人出镜）

非美妆内容（应输出 F）:
- 游戏/电竞/动画/卡通
- 体育赛事/运动集锦
- 烹饪/美食、生活vlog（非美妆主题）
- 音乐/MV、综艺/娱乐、电影/剧集
- 播客/谈话节目/采访
- 科技/编程/教学（非美妆类）
- 新闻/政治/发布会
- 宗教/灵修
- 儿童玩具/卡通内容
- 军事/装备展示
- 赛事解说/分析

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


def atomic_write(df: pd.DataFrame, target_path: str):
    """
    原子写：先写 .tmp，再 os.replace → 避免写到一半崩溃损坏文件。
    自动根据文件扩展名选择 parquet 或 csv 格式。
    """
    tmp_path = target_path + ".tmp"
    ext = os.path.splitext(target_path)[1].lower()
    if ext == ".parquet":
        df.to_parquet(tmp_path, index=False)
    else:
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
    # 不提供 keyword 给 LLM，避免限定特定运动
    # if keyword: parts.append(f"搜索关键词: {keyword}")
    if title:   parts.append(f"视频标题: {title}")
    if channel: parts.append(f"频道名称: {channel}")
    if desc:    parts.append(f"视频简介: {desc}")

    return "\n".join(parts) + "\n\n该视频是否属于美妆/化妆教程内容？不限风格（日常妆、宴会妆、创意妆、护肤流程、产品测评等均可）。仅输出 T 或 F。"


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
    ext = os.path.splitext(input_csv)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(input_csv).fillna("").astype(str)
    else:
        df = pd.read_csv(input_csv, dtype=str, low_memory=False).fillna("")
    n_total = len(df)
    safe_total = max(n_total, 1)  # 除零保护
    log(f"  样本行数: {n_total:,}")

    # ── 自动检测全量数据池大小 ─────────────────────────────
    input_abs = os.path.abspath(input_csv)
    input_dir = os.path.dirname(input_abs)
    file_name = os.path.basename(input_csv).lower()
    pool_size = None
    pool_label = "全量"

    # 搜索范围：QC目录自身 → 父(数据集根) → 005_clean下的run目录
    search_dirs = [input_dir]
    p = input_dir
    for _ in range(4):
        p = os.path.dirname(p)
        if p and os.path.isdir(p):
            search_dirs.append(p)

    # 额外：找 005_clean 下的 run 目录
    for sdir in list(search_dirs):
        clean_dir = os.path.join(sdir, "005_clean")
        if os.path.isdir(clean_dir):
            for rd in sorted(os.listdir(clean_dir)):
                rp = os.path.join(clean_dir, rd)
                if os.path.isdir(rp):
                    search_dirs.append(rp)

    for sdir in search_dirs:
        sf = os.path.join(sdir, "clean_summary.json")
        if os.path.exists(sf):
            import json as _json
            with open(sf) as _f:
                _s = _json.load(_f)
            if "keep" in file_name:
                pool_size = _s.get("total_keep")
                pool_label = "全量 KEEP"
            elif "drop" in file_name:
                pool_size = _s.get("total_drop")
                pool_label = "全量 DROP"
            if pool_size is not None and pool_size > 0:
                break

    # fallback: QC目录自身的 progress.json 中的 total 字段
    if pool_size is None:
        pf = os.path.join(input_dir, "progress.json")
        if os.path.exists(pf):
            import json as _json
            with open(pf) as _f:
                _p = _json.load(_f)
            pool_size = _p.get("total") or _p.get("total_rows") or _p.get("keep")
            if pool_size and ("keep" in file_name):
                pool_label = "全量 KEEP"
            elif pool_size and ("drop" in file_name):
                pool_label = "全量 DROP"

    if pool_size is not None and pool_size > 0:
        log(f"  {pool_label}数据池: {pool_size:,} 条")
    else:
        log(f"  (未找到全量数据池信息)")

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
                        atomic_write(df, input_csv)
                        log(f"  checkpoint ✓  {completed:,} / {n_pending:,}  → {input_csv}")
                        last_checkpoint = now

    elif dry_run:
        log("dry-run 模式，跳过 LLM 调用。")

    # ── 原子写回原始文件 + 快照 ──────────────────────────────
    atomic_write(df, input_csv)
    log(f"结果已回写原文件: {input_csv}")

    atomic_write(df, snapshot_csv)
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
    print(f"  文本 QC 美妆版 — {os.path.basename(input_csv)}")
    print("=" * 62)
    print(f"  run_id:   {run_id}")
    print(f"  模型:     {model}    并发: {workers}")
    print(f"  耗时:     {elapsed:>10.1f}s")
    print(f"  {'─'*54}")
    print(f"  QC 结果:")
    print(f"    T (美妆内容):    {t_count:>10,}  ({t_count/safe_total*100:5.1f}%)")
    print(f"    F (非美妆内容):  {f_count:>10,}  ({f_count/safe_total*100:5.1f}%)")
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

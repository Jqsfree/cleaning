#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
"""
chunk_text_qc.py — 文本 LLM 质检（统一版）

从 categories/<name>/rules/qc.toml 加载类别专属的 prompt 配置。
支持通过 --category 参数或自动从输入路径推断类别。

用法:
  python3 chunk_text_qc.py input.csv --category language_teaching
  python3 chunk_text_qc.py input.csv --category beauty -w 20
  python3 chunk_text_qc.py input.csv --category language_teaching -m qwen-plus --dry-run
"""

import sys, os, time, json, argparse, random, shutil, tomllib, signal, threading
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

try:
    import httpx
except ImportError:
    httpx = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.adaptive_api import AdaptiveConcurrencyGate
from core.io import resolve_output_dir
from core.progress import ThrottledProgress
from core.sop import write_run_log


# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════

DEFAULT_MODEL    = "qwen3.5-flash"
DEFAULT_WORKERS  = 20
CHECKPOINT_EVERY = 200           # 检查点行数间隔
CHECKPOINT_SECS  = 60            # 检查点时间间隔（秒）
MAX_RETRIES      = 3
API_BASE_URL     = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_CATEGORIES_DIR = Path(__file__).resolve().parent.parent / "categories"


def load_qc_config(category: str) -> dict:
    """从 categories/<category>/rules/qc.toml 加载 QC 配置。"""
    config_path = _CATEGORIES_DIR / category / "rules" / "qc.toml"
    if not config_path.exists():
        print(f"[ERROR] 类别 '{category}' 的 qc.toml 不存在: {config_path}")
        sys.exit(1)

    try:
        cfg = tomllib.loads(config_path.read_text("utf-8"))
    except tomllib.TOMLDecodeError as e:
        print(f"[ERROR] qc.toml 格式错误: {config_path}")
        print(f"  {e}")
        sys.exit(1)

    try:
        return {
            "category": cfg["meta"]["category"],
            "category_label": cfg["meta"]["category_label"],
            "system_prompt": cfg["prompts"]["system_prompt"],
            "user_prompt_question": cfg["prompts"]["user_prompt_question"],
            "pass_label": cfg["labels"]["pass_label"],
            "fail_label": cfg["labels"]["fail_label"],
        }
    except KeyError as e:
        print(f"[ERROR] qc.toml 缺少必需字段 {e}: {config_path}")
        print(f"  需要 [meta].category, [meta].category_label")
        print(f"       [prompts].system_prompt, [prompts].user_prompt_question")
        print(f"       [labels].pass_label, [labels].fail_label")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_run_id() -> str:
    """生成本次运行 ID，格式: YYYYMMDD_HHmmss"""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_output_stem(input_stem: str, run_id: str) -> str:
    """统一命名规范: {原文件名}_textqc_{run_id}"""
    return f"{input_stem}_textqc_{run_id}"


def backup_input(input_csv: str, run_id: str) -> str:
    """备份原文件，只备份一次。"""
    bak_path = f"{input_csv}.bak_{run_id}"
    if not os.path.exists(bak_path):
        shutil.copy2(input_csv, bak_path)
        log(f"已备份原文件: {bak_path}")
    else:
        log(f"备份已存在，跳过: {bak_path}")
    return bak_path


def atomic_write(df: pd.DataFrame, target_path: str):
    """原子写：先写 .tmp，再 os.replace。"""
    tmp_path = target_path + ".tmp"
    ext = os.path.splitext(target_path)[1].lower()
    if ext == ".parquet":
        df.to_parquet(tmp_path, index=False)
    else:
        df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, target_path)


def _create_client(api_key: str, workers: int = DEFAULT_WORKERS) -> OpenAI:
    """创建 OpenAI 兼容 client，增大连接池并设置显式超时。

    trust_env=False：忽略 shell 里的 HTTP(S)_PROXY / ALL_PROXY。
    常见本地代理为 socks://…，httpx 默认不支持，会直接在 Client() 初始化时报错。
    DashScope 兼容端点通常直连即可。
    """
    if httpx is not None:
        max_conn = max(30, workers + 10)
        keep = max(20, min(max_conn, workers + 5))
        http_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=max_conn,
                max_keepalive_connections=keep,
            ),
            timeout=httpx.Timeout(30.0, connect=10.0),
            trust_env=False,
        )
        return OpenAI(api_key=api_key, base_url=API_BASE_URL, http_client=http_client)
    else:
        return OpenAI(api_key=api_key, base_url=API_BASE_URL, timeout=30.0)


# ══════════════════════════════════════════════════════════════
# LLM 调用
# ══════════════════════════════════════════════════════════════

def _build_user_prompt(row: dict, question: str) -> str:
    title   = str(row.get("title",       ""))
    channel = str(row.get("channel",     ""))
    desc    = str(row.get("description", ""))[:200]

    parts = []
    if title:   parts.append(f"视频标题: {title}")
    if channel: parts.append(f"频道名称: {channel}")
    if desc:    parts.append(f"视频简介: {desc}")

    return "\n".join(parts) + "\n\n" + question


def check_one(client, row: dict, model: str, system_prompt: str,
              user_question: str, gate: AdaptiveConcurrencyGate) -> tuple[str, str, str]:
    """返回 (result, model_used, error_reason)"""
    user_prompt = _build_user_prompt(row, user_question)

    for attempt in range(MAX_RETRIES):
        gate.acquire()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=5,
            )
            raw = resp.choices[0].message.content.strip().upper()
            # 取首个 T/F/U（无法确认 → U，勿当 ERROR）
            label = next((ch for ch in raw if ch in ("T", "F", "U")), None)
            if label:
                gate.record_outcome(ok=True)
                return (label, model, "")
            if attempt < MAX_RETRIES - 1:
                time.sleep(random.uniform(0.05, 0.25))
                continue
            return ("ERROR", model, f"invalid_response:{raw[:20]}")

        except RateLimitError as e:
            gate.on_rate_limit(log_fn=log)
            gate.record_outcome(transient_error=True, log_fn=log)
            # 靠 gate 降并发；仅极短 jitter
            time.sleep(random.uniform(0.05, 0.25))
            if attempt == MAX_RETRIES - 1:
                return ("ERROR", model, f"rate_limit_error:{str(e)[:80]}")

        except APIConnectionError:
            gate.record_outcome(transient_error=True, log_fn=log)
            time.sleep(random.uniform(0.05, 0.25))
            if attempt == MAX_RETRIES - 1:
                return ("ERROR", model, "api_connection_error")

        except Exception as ex:
            time.sleep(random.uniform(0.05, 0.25))
            if attempt == MAX_RETRIES - 1:
                return ("ERROR", model, f"exception:{type(ex).__name__}:{str(ex)[:80]}")

        finally:
            gate.release()

    return ("ERROR", model, "max_retries_exceeded")


# ══════════════════════════════════════════════════════════════
# 名单输出
# ══════════════════════════════════════════════════════════════

def _write_name_lists(
    df: pd.DataFrame,
    pass_path: str,
    fail_path: str,
    error_path: str,
    uncertain_path: str | None = None,
):
    pass_ids = df[df["qc_text_result"] == "T"]["video_id"].tolist()
    fail_ids = df[df["qc_text_result"] == "F"]["video_id"].tolist()
    error_ids = df[df["qc_text_result"] == "ERROR"]["video_id"].tolist()
    u_ids = df[df["qc_text_result"] == "U"]["video_id"].tolist()

    pairs = [(pass_path, pass_ids), (fail_path, fail_ids), (error_path, error_ids)]
    if uncertain_path:
        pairs.append((uncertain_path, u_ids))

    for path, ids in pairs:
        with open(path, "w") as f:
            for vid in ids:
                f.write(f"{vid}\n")

    log(f"通过名单:   {pass_path}  ({len(pass_ids):,} 条)")
    log(f"未通过名单: {fail_path}  ({len(fail_ids):,} 条)")
    if uncertain_path:
        log(f"不确定名单: {uncertain_path}  ({len(u_ids):,} 条)")
    if error_ids:
        log(f"错误名单:   {error_path}  ({len(error_ids):,} 条)")


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def run_text_qc(
    input_csv: str,
    output_dir: str,
    category: str = "language_teaching",
    model: str = DEFAULT_MODEL,
    workers: int = DEFAULT_WORKERS,
    dry_run: bool = False,
    sample: int = 0,
    force: bool = False,
    resume: bool = False,
) -> dict:
    """运行文本 LLM 质检。

    Args:
        category: 类别名（language_teaching, beauty 等），用于加载 qc.toml
        sample: 从待处理行中随机抽样 N 条（0=全量）
    """
    qc_cfg = load_qc_config(category)
    system_prompt = qc_cfg["system_prompt"]
    user_question = qc_cfg["user_prompt_question"]
    pass_label    = qc_cfg["pass_label"]
    fail_label    = qc_cfg["fail_label"]

    t0     = time.perf_counter()
    run_id = make_run_id()

    # ── API Key ──────────────────────────────────────────────
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key and not dry_run:
        print("[ERROR] 未设置 DASHSCOPE_API_KEY 环境变量")
        sys.exit(1)
    if dry_run:
        client = None
    else:
        assert api_key is not None  # 在 dry_run=False 时保证非空
        client = _create_client(api_key, workers=workers)
    gate   = AdaptiveConcurrencyGate(initial=workers)

    # ── 读取 ─────────────────────────────────────────────────
    log(f"读取: {input_csv}")
    ext = os.path.splitext(input_csv)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(input_csv).fillna("").astype(str)
    else:
        df = pd.read_csv(input_csv, dtype=str, low_memory=False).fillna("")
    n_total = len(df)
    safe_total = max(n_total, 1)
    log(f"  样本行数: {n_total:,}")

    # ── 自动检测全量数据池大小 ─────────────────────────────
    input_abs = os.path.abspath(input_csv)
    input_dir = os.path.dirname(input_abs)
    file_name = os.path.basename(input_csv).lower()
    pool_size = None
    pool_label = "全量"

    search_dirs = [input_dir]
    p = input_dir
    for _ in range(4):
        p = os.path.dirname(p)
        if p and os.path.isdir(p):
            search_dirs.append(p)

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
    if resume:
        bak_path = ""  # 续跑不复建备份
        log("续跑模式：复用已有备份，不创建新备份")
    else:
        bak_path = backup_input(input_csv, run_id)

    # ── 续跑: 尝试继承已有 run_id ─────────────────────────────
    if resume:
        existing_run = df.loc[(df["qc_run_id"] != "") & (df["qc_run_id"].notna()), "qc_run_id"]
        if len(existing_run) > 0:
            prev_run_id = existing_run.mode().iloc[0] if len(existing_run.mode()) > 0 else existing_run.iloc[0]
            run_id = prev_run_id
            log(f"续跑模式：继承已有 run_id: {run_id}")
        else:
            log(f"续跑模式：无已有 run_id，使用当前: {run_id}")

    # ── 续跑状态摘要 ───────────────────────────────────────────
    if resume:
        done_t = int((df["qc_text_result"] == "T").sum())  # pyright: ignore[reportArgumentType]
        done_f = int((df["qc_text_result"] == "F").sum())  # pyright: ignore[reportArgumentType]
        done_u = int((df["qc_text_result"] == "U").sum())  # pyright: ignore[reportArgumentType]
        done_err = int((df["qc_text_result"] == "ERROR").sum())  # pyright: ignore[reportArgumentType]
        empty = int(df["qc_text_result"].isin(["", "ERROR"]).sum())  # pyright: ignore[reportArgumentType]
        print(
            f"  [续跑] 已完成: T={done_t}  F={done_f}  U={done_u}  "
            f"ERROR={done_err}  |  待处理: {empty}"
        )

    # ── 初始化新增列 ─────────────────────────────────────────
    for col in ["qc_text_result", "qc_text_model", "qc_run_id", "qc_error_reason"]:
        if col not in df.columns:
            df[col] = ""

    # ── force: 清除已有 QC 结果 ───────────────────────────────
    if force:
        for col in ["qc_text_result", "qc_text_model", "qc_run_id", "qc_error_reason"]:
            df[col] = ""
        log("force 模式：已清除已有 QC 结果，全部重新质检")

    # ── 找出待处理行 ─────────────────────────────────────────
    pending_mask = df["qc_text_result"].isin(["", "ERROR"]) | df["qc_text_result"].isna()
    pending_idx  = df[pending_mask].index.tolist()
    n_pending    = len(pending_idx)

    # 随机抽样
    if sample > 0 and n_pending > sample:
        import random
        pending_idx = random.sample(pending_idx, sample)
        log(f"随机抽样: {len(pending_idx):,} / {n_pending:,} 条")
        n_pending = len(pending_idx)

    if n_pending == 0:
        log("全部已质检，无需重跑。")
    else:
        log(f"待处理: {n_pending:,} / {n_total:,}  ({n_pending/safe_total*100:.1f}%)")

    # ── 输出路径 ─────────────────────────────────────────────
    input_stem  = Path(input_csv).stem
    output_stem = make_output_stem(input_stem, run_id)
    os.makedirs(output_dir, exist_ok=True)

    snapshot_csv   = os.path.join(output_dir, f"{output_stem}.csv")
    summary_path   = os.path.join(output_dir, f"{output_stem}_summary.json")
    pass_list_path = os.path.join(output_dir, f"{output_stem}_pass.txt")
    fail_list_path = os.path.join(output_dir, f"{output_stem}_fail.txt")
    uncertain_list_path = os.path.join(output_dir, f"{output_stem}_uncertain.txt")
    error_list_path = os.path.join(output_dir, f"{output_stem}_error.txt")

    # ── 并发 QC ──────────────────────────────────────────────
    if n_pending > 0 and not dry_run:
        completed = 0
        last_checkpoint = time.time()
        prog = ThrottledProgress(
            output_dir, "qc_text",
            interval_sec=5.0, every_n=50,
            input=input_csv, category=category, total=n_pending,
        )
        prog.tick(force=True, done=0, T=0, F=0, U=0, ERROR=0)

        flush_lock = threading.Lock()
        interrupted = {"flag": False}

        def _flush_now(reason: str = "checkpoint") -> None:
            with flush_lock:
                atomic_write(df, input_csv)
                log(f"  {reason} ✓  {completed:,} / {n_pending:,}  → {input_csv}")

        def _on_signal(signum, _frame):
            interrupted["flag"] = True
            try:
                _flush_now(f"signal_{signum}_flush")
            except Exception as e:
                log(f"  [WARN] 中断落盘失败: {e}")
            prog.tick(force=True, done=completed, status="interrupted")
            raise SystemExit(128 + (signum if isinstance(signum, int) else 2))

        prev_int = signal.signal(signal.SIGINT, _on_signal)
        prev_term = signal.signal(signal.SIGTERM, _on_signal)
        try:
            # 有界 in-flight：避免十万级 future 一次性堆积
            max_inflight = max(workers * 2, 32)
            pending_iter = iter(pending_idx)
            future_to_idx: dict = {}

            def _submit_one(ex, idx):
                return ex.submit(
                    check_one, client, df.loc[idx].to_dict(),
                    model, system_prompt, user_question, gate,
                )

            with ThreadPoolExecutor(max_workers=workers) as executor:
                for _ in range(min(max_inflight, n_pending)):
                    try:
                        idx = next(pending_iter)
                    except StopIteration:
                        break
                    future_to_idx[_submit_one(executor, idx)] = idx

                with tqdm(total=n_pending, desc="文本 LLM 质检") as pbar:
                    while future_to_idx:
                        if interrupted["flag"]:
                            break
                        # 逐个完成再补位，保持 in-flight 有界
                        done_futs = []
                        for future in as_completed(future_to_idx):
                            done_futs.append(future)
                            break  # 只取一个已完成的

                        for future in done_futs:
                            idx = future_to_idx.pop(future)
                            try:
                                result_str, model_used, error_reason = future.result()
                            except Exception as ex:
                                result_str, model_used, error_reason = (
                                    "ERROR", model, f"future_exception:{type(ex).__name__}"
                                )

                            df.at[idx, "qc_text_result"]  = result_str
                            df.at[idx, "qc_text_model"]   = model_used
                            df.at[idx, "qc_run_id"]        = run_id
                            df.at[idx, "qc_error_reason"]  = error_reason

                            completed += 1
                            pbar.update(1)

                            now = time.time()
                            if completed % CHECKPOINT_EVERY == 0 or (now - last_checkpoint) >= CHECKPOINT_SECS:
                                _flush_now("checkpoint")
                                last_checkpoint = now

                            t_now = int((df["qc_text_result"] == "T").sum())
                            f_now = int((df["qc_text_result"] == "F").sum())
                            u_now = int((df["qc_text_result"] == "U").sum())
                            e_now = int((df["qc_text_result"] == "ERROR").sum())
                            prog.tick(done=completed, T=t_now, F=f_now, U=u_now, ERROR=e_now)

                            if not interrupted["flag"]:
                                try:
                                    nidx = next(pending_iter)
                                    future_to_idx[_submit_one(executor, nidx)] = nidx
                                except StopIteration:
                                    pass
        finally:
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)

    elif dry_run:
        log("dry-run 模式，跳过 LLM 调用。")
        from core.progress import mark_done
        mark_done(
            output_dir, "qc_text",
            dry_run=True, input=input_csv, category=category,
            total=n_total, pending_before=n_pending,
        )
        write_run_log(
            "qc_text", input_csv, output_dir,
            stats={
                "category": category,
                "dry_run": True,
                "total_rows": n_total,
                "pending_before": n_pending,
                "input": os.path.abspath(input_csv),
            },
            command=f"text.py {input_csv} --category {category} -o {output_dir} --dry-run",
            category=category,
        )
        return {
            "step": "text_qc_v2",
            "category": category,
            "dry_run": True,
            "run_id": run_id,
            "total_rows": n_total,
            "pending_before": n_pending,
        }

    # ── 原子写回原始文件 + 快照 ──────────────────────────────
    atomic_write(df, input_csv)
    log(f"结果已回写原文件: {input_csv}")

    atomic_write(df, snapshot_csv)
    log(f"快照已写出: {snapshot_csv}")

    # ── 名单 ─────────────────────────────────────────────────
    _write_name_lists(
        df, pass_list_path, fail_list_path, error_list_path,
        uncertain_path=uncertain_list_path,
    )

    # ── 统计 ─────────────────────────────────────────────────
    elapsed   = time.perf_counter() - t0
    t_count   = int((df["qc_text_result"] == "T").sum())
    f_count   = int((df["qc_text_result"] == "F").sum())
    u_count   = int((df["qc_text_result"] == "U").sum())
    err_count = int((df["qc_text_result"] == "ERROR").sum())

    summary = {
        "step":           "text_qc_v2",
        "run_id":         run_id,
        "category":       category,
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
            "U":     u_count,  "U_pct":   round(u_count / safe_total * 100, 1),
            "ERROR": err_count,
        },
        "outputs": {
            "written_back_to": input_csv,
            "snapshot_csv":    snapshot_csv,
            "summary_json":    summary_path,
            "pass_list":       pass_list_path,
            "fail_list":       fail_list_path,
            "uncertain_list":  uncertain_list_path,
            "error_list":      error_list_path,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log(f"写出摘要: {summary_path}")

    # ── 终端摘要 ─────────────────────────────────────────────
    print()
    print("=" * 62)
    print(f"  文本 QC ({qc_cfg['category_label']}) — {os.path.basename(input_csv)}")
    print("=" * 62)
    print(f"  run_id:   {run_id}")
    print(f"  模型:     {model}    并发: {workers}")
    print(f"  耗时:     {elapsed:>10.1f}s")
    print(f"  {'─'*54}")
    print(f"  QC 结果:")
    print(f"    T ({pass_label}):  {t_count:>10,}  ({t_count/safe_total*100:5.1f}%)")
    print(f"    F ({fail_label}):  {f_count:>10,}  ({f_count/safe_total*100:5.1f}%)")
    print(f"    U (无法确认): {u_count:>10,}  ({u_count/safe_total*100:5.1f}%)")
    print(f"    ERROR:       {err_count:>10,}")
    print("  注: LLM-QC% / keep% 不是交付 KPI；U 应交人工或小模型")
    print(f"  {'─'*54}")
    print(f"  备份:     {bak_path}")
    print(f"  回写至:   {input_csv}")
    print(f"  快照:     {snapshot_csv}")
    print("=" * 62)

    from core.progress import mark_done
    mark_done(
        output_dir, "qc_text",
        input=input_csv, category=category,
        done=t_count + f_count + u_count + err_count, total=n_total,
        T=t_count, F=f_count, U=u_count, ERROR=err_count,
        snapshot=snapshot_csv, elapsed_sec=round(elapsed, 1),
    )
    write_run_log(
        "qc_text", input_csv, output_dir,
        stats={
            "category": category,
            "total_rows": n_total,
            "T": t_count,
            "F": f_count,
            "U": u_count,
            "ERROR": err_count,
            "elapsed_sec": round(elapsed, 1),
            "snapshot_csv": snapshot_csv,
            "summary_json": summary_path,
            "written_back_to": input_csv,
        },
        command=f"text.py {input_csv} --category {category} -o {output_dir}",
        category=category,
    )

    return summary


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="文本 LLM 质检（统一版）")
    parser.add_argument("input",        help="输入 CSV 或 Parquet")
    parser.add_argument("-c", "--category", default="language_teaching",
                        help="类别名（language_teaching, beauty 等）")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="输出目录（默认与输入同目录）")
    parser.add_argument("-m", "--model",   default=DEFAULT_MODEL)
    parser.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--sample", type=int, default=0,
                        help="从待处理行中随机抽样 N 条（0=全量）")
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--force",         action="store_true")
    parser.add_argument("--resume",        action="store_true",
                        help="续跑模式：继承已有 run_id，跳过已完成行，不复建备份")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    output_dir = resolve_output_dir(args.output_dir, args.input)
    run_text_qc(args.input, output_dir, category=args.category,
                model=args.model, workers=args.workers, dry_run=args.dry_run,
                sample=args.sample, force=args.force, resume=args.resume)


if __name__ == "__main__":
    main()

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================
# DashScope / Qwen 配置
# =====================

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

MODEL_NAME = "qwen3.6-flash"

# =====================
# 配置项
# =====================

INPUT_CSV   = "/home/jqs/tiyu/dispose_data/sample_videos.csv"
OUTPUT_CSV  = "/home/jqs/tiyu/dispose_data/final_baseball_videos_2cat .csv"
CKPT_CSV    = "/home/jqs/tiyu/dispose_data/_checkpoint_qc.csv"   # 断点续跑缓存

MAX_WORKERS     = 8    # 并发线程数；按 API 限速调整，建议先跑 8 观察报错率
SAVE_EVERY      = 200   # 每处理多少条写一次磁盘
MAX_RETRIES     = 3     # 单条最大重试次数
RETRY_BACKOFF   = 2     # 重试等待基数（秒），实际等待 = RETRY_BACKOFF * attempt

# =====================
# 全局写锁（多线程安全写盘）
# =====================

_write_lock   = threading.Lock()
_counter_lock = threading.Lock()
_pending       = []          # 待落盘的 (video_id, result) 列表
_pending_count = 0

# =====================
# 读取数据
# =====================

df_raw = pd.read_csv(INPUT_CSV)

# =====================
# 1. 去重
# =====================

df_raw = df_raw.drop_duplicates(subset=["video_id"])

# =====================
# 2. 基础清洗
# =====================

df_raw = df_raw.dropna(subset=["title", "channel", "duration_seconds", "view_count"])

df_raw["duration_seconds"] = pd.to_numeric(df_raw["duration_seconds"], errors="coerce")
df_raw["view_count"]       = pd.to_numeric(df_raw["view_count"],       errors="coerce")

df_raw = df_raw[df_raw["duration_seconds"] > 30]
df_raw = df_raw[df_raw["view_count"] > 100]

# =====================
# 3. 文本标准化
# =====================

df_raw["title"] = df_raw["title"].astype(str).str.strip().str.lower()

# =====================
# 4. 过滤会员/私密/下架视频
# =====================

BLOCK_KEYWORDS = [
    "private video",
    "deleted video",
    "members only",
    "subscriber only",
    "video unavailable",
]

pattern = "|".join(BLOCK_KEYWORDS)
df_raw = df_raw[~df_raw["title"].str.contains(pattern, case=False, na=False)]

# =====================
# 5. 断点续跑：读取已有结果
# =====================

if os.path.exists(CKPT_CSV):
    df_done  = pd.read_csv(CKPT_CSV)
    done_ids = set(df_done["video_id"].astype(str))
    print(f"[续跑] 已完成 {len(done_ids)} 条，跳过重新请求")
else:
    df_done  = pd.DataFrame(columns=["video_id", "qc_ai"])
    done_ids = set()

# 只处理未完成的行
df_todo = df_raw[~df_raw["video_id"].astype(str).isin(done_ids)].copy()
print(f"[待处理] {len(df_todo)} 条 / 总计清洗后 {len(df_raw)} 条")

# =====================
# 6. AI 质检函数（带重试）
# =====================

SYSTEM_PROMPT = """
分析该视频的内容是否属于体育赛事解说。

仅输出：
T
或
F
"""

def ai_qc(video_id: str, text: str) -> tuple[str, str]:
    """返回 (video_id, 'T'|'F')"""

    if not text or not str(text).strip():
        return video_id, "F"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text},
                ],
                temperature=0,
                max_tokens=1,
            )

            result = (
                response.choices[0]
                .message.content
                .strip()
                .upper()
            )

            if result not in ("T", "F"):
                result = "F"

            return video_id, result

        except Exception as e:
            wait = RETRY_BACKOFF * attempt
            print(f"[WARN] video_id={video_id} attempt={attempt} error={e}; retry in {wait}s")
            time.sleep(wait)

    return video_id, "F"   # 全部重试失败，保守返回 F

# =====================
# 7. 落盘函数（线程安全）
# =====================

def _flush_pending(force: bool = False):
    """将 _pending 缓冲写入 CKPT_CSV；force=True 时忽略 SAVE_EVERY 阈值"""
    global _pending, _pending_count

    with _write_lock:
        if not _pending:
            return
        if not force and len(_pending) < SAVE_EVERY:
            return

        chunk = pd.DataFrame(_pending, columns=["video_id", "qc_ai"])
        _pending = []
        _pending_count = 0

    # 追加写入（写锁已释放，用文件操作锁保护即可）
    write_header = not os.path.exists(CKPT_CSV)
    with _write_lock:
        chunk.to_csv(CKPT_CSV, mode="a", header=write_header, index=False)

    print(f"\n[✓] 已落盘 {len(chunk)} 条 → {CKPT_CSV}")

# =====================
# 8. 多线程批量执行
# =====================

rows = list(df_todo.itertuples(index=False))

with tqdm(total=len(rows), desc="AI质检", unit="条") as pbar:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {
            executor.submit(ai_qc, str(row.video_id), row.title): row.video_id
            for row in rows
        }

        for future in as_completed(futures):
            vid, result = future.result()

            with _counter_lock:
                _pending.append((vid, result))
                _pending_count += 1
                should_flush = (_pending_count % SAVE_EVERY == 0)

            if should_flush:
                _flush_pending()

            pbar.update(1)

# 处理剩余未落盘的条目
_flush_pending(force=True)

# =====================
# 9. 合并全量结果并输出
# =====================

df_all_qc = pd.read_csv(CKPT_CSV)

# 去重（断点重跑可能存在少量重复）
df_all_qc = df_all_qc.drop_duplicates(subset=["video_id"])

df_final = df_raw.merge(
    df_all_qc[["video_id", "qc_ai"]],
    on="video_id",
    how="inner",
)

df_final = df_final[df_final["qc_ai"] == "T"]

df_final.to_csv(OUTPUT_CSV, index=False)

print("\n清洗 + AI质检完成")
print("最终数量:", len(df_final))

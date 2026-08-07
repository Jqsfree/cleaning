#!/usr/bin/env python3
"""
tools/fetch_live_broadcast.py — 查询 YouTube 视频是否直播（liveBroadcastContent）

用 videos.list part=snippet 的 liveBroadcastContent 字段：
    none      = 非直播 / 直播已结束的回放（API 无单独 completed）
    upcoming  = 已排期未开播
    live      = 正在直播

与 fetch_yt_definition.py 同构：断点续传 + checkpoint。

用法:
  export YOUTUBE_API_KEY='你的密钥'
  02_脚本/tools/fetch_live_broadcast.py input.csv -o out.csv
  02_脚本/tools/fetch_live_broadcast.py input.parquet -o out.csv --overwrite
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

API_URL = "https://www.googleapis.com/youtube/v3/videos"
BATCH_SIZE = 50
TIMEOUT = 60.0
DEFAULT_SLEEP = 0.05
# live_status 终态（断点续跑用）
TERMINAL = {"ok", "not_found"}
# YouTube snippet.liveBroadcastContent 官方枚举（结束后回放变为 none，无 completed）
VALID_LIVE_BROADCAST = {"none", "upcoming", "live"}
CHECKPOINT_COLS = ["video_id", "live_broadcast_content", "live_status"]


def resolve_api_key(cli_key: str | None) -> str:
    key = (cli_key or os.getenv("YOUTUBE_API_KEY", "") or "").strip()
    if not key:
        print("[ERROR] 未设置 API Key。请先: export YOUTUBE_API_KEY='...'")
        sys.exit(1)
    return key


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, low_memory=False)


def fetch_live_broadcast_batch(
    api_key: str, video_ids: list[str], timeout: float = TIMEOUT
) -> dict[str, dict[str, str]]:
    """查一批 video_id 的 liveBroadcastContent。"""
    params = urllib.parse.urlencode(
        {
            "part": "snippet",
            "id": ",".join(video_ids),
            "key": api_key,
            "maxResults": BATCH_SIZE,
        }
    )
    url = f"{API_URL}?{params}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"网络错误: {e}") from e

    found: dict[str, dict[str, str]] = {}
    for item in payload.get("items") or []:
        vid = str(item.get("id") or "").strip()
        if not vid:
            continue
        lb = str((item.get("snippet") or {}).get("liveBroadcastContent") or "").strip().lower()
        if lb in VALID_LIVE_BROADCAST:
            found[vid] = {"live_broadcast_content": lb, "live_status": "ok"}
        else:
            # 未知枚举值保留原文便于排查
            found[vid] = {
                "live_broadcast_content": lb,
                "live_status": "error",
            }

    out: dict[str, dict[str, str]] = {}
    for vid in video_ids:
        if vid in found:
            out[vid] = found[vid]
        else:
            out[vid] = {"live_broadcast_content": "", "live_status": "not_found"}
    return out


def load_input_df(input_path: str, ckpt_path: str, final_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        print(f"[ERROR] 文件不存在: {input_path}")
        sys.exit(1)

    base = _read_table(input_path)
    for col in ("live_broadcast_content", "live_status"):
        if col not in base.columns:
            base[col] = ""
    candidates: list[tuple[str, str, float]] = []
    for path, label in (
        (ckpt_path, f"断点 {ckpt_path}"),
        (final_path, f"已有输出 {final_path}"),
    ):
        if os.path.exists(path) and os.path.abspath(path) != os.path.abspath(input_path):
            candidates.append((path, label, os.path.getmtime(path)))
    if not candidates:
        return base

    path, label, _ = max(candidates, key=lambda x: x[2])
    progress = _read_table(path)
    for col in ("live_broadcast_content", "live_status"):
        if col not in progress.columns:
            progress[col] = ""

    if "video_id" not in base.columns or "video_id" not in progress.columns:
        print(f"[续跑] {label}（无 video_id，直接用进度文件）")
        return progress

    print(f"[续跑] 合并进度自 {label}")
    prog = (
        progress[["video_id", "live_broadcast_content", "live_status"]]
        .astype(str)
        .fillna("")
        .drop_duplicates(subset=["video_id"], keep="last")
    )
    base = base.drop(
        columns=[c for c in ("live_broadcast_content", "live_status") if c in base.columns]
    )
    merged = base.merge(prog, on="video_id", how="left")
    for col in ("live_broadcast_content", "live_status"):
        merged[col] = merged[col].fillna("")
    return merged


def get_pending_mask(df: pd.DataFrame, *, overwrite: bool, retry_errors: bool) -> pd.Index:
    if overwrite:
        df["live_broadcast_content"] = ""
        df["live_status"] = ""
        return df.index
    status = df["live_status"].fillna("").astype(str).str.strip()
    mask = ~status.isin(TERMINAL)
    if not retry_errors:
        mask = mask & (status != "error")
    return df[mask].index


def flush_checkpoint(df: pd.DataFrame, ckpt_path: str) -> None:
    cols = [c for c in CHECKPOINT_COLS if c in df.columns]
    mask = df["live_status"].fillna("").astype(str).str.strip() != ""
    slim = df.loc[mask, cols]
    parent = os.path.dirname(os.path.abspath(ckpt_path)) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = ckpt_path + ".tmp"
    slim.to_csv(tmp, index=False)
    os.replace(tmp, ckpt_path)


def write_final_csv(df: pd.DataFrame, final_path: str, ckpt_path: str | None = None) -> None:
    parent = os.path.dirname(os.path.abspath(final_path)) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = final_path + ".flush_tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, final_path)
    for side in (ckpt_path, final_path + ".ckpt.csv"):
        if side and os.path.exists(side) and os.path.abspath(side) != os.path.abspath(final_path):
            try:
                os.unlink(side)
            except OSError:
                pass


def smoke_api_key(api_key: str) -> None:
    print("冒烟: videos.list jNQXAC9IVRw …")
    try:
        r = fetch_live_broadcast_batch(api_key, ["jNQXAC9IVRw"])
    except RuntimeError as e:
        print(f"[ERROR] 冒烟失败: {e}")
        sys.exit(1)
    print(f"  冒烟 ok: {r.get('jNQXAC9IVRw')}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="查 YouTube liveBroadcastContent（none/upcoming/live/completed）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="输入 CSV/Parquet（含 video_id 列）")
    p.add_argument("-o", "--output", required=True, help="输出 CSV 路径")
    p.add_argument("--api-key", default=None, help="YOUTUBE_API_KEY（缺省读环境变量）")
    p.add_argument("--overwrite", action="store_true", help="忽略已有结果重查")
    p.add_argument("--retry-errors", action="store_true", help="重试 error 状态")
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                   help=f"每批请求间隔秒数（防速率限制；默认 {DEFAULT_SLEEP}）")
    p.add_argument("-n", "--limit", type=int, default=None,
                   help="仅查前 N 个 video_id（调试用）")
    p.add_argument("--no-smoke", action="store_true", help="跳过 API 冒烟")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    api_key = resolve_api_key(args.api_key)
    if not args.no_smoke:
        smoke_api_key(api_key)

    input_path = args.input
    output_path = args.output
    ckpt_path = output_path + ".ckpt.csv"

    # 全量输入只在最后 merge；查询阶段只维护 video_id 瘦表（百万级必需）
    base = _read_table(input_path)
    if "video_id" not in base.columns:
        print("[ERROR] 输入需含 video_id 列")
        sys.exit(1)

    work = (
        base[["video_id"]]
        .astype(str)
        .assign(video_id=lambda d: d["video_id"].str.strip())
        .drop_duplicates(subset=["video_id"], keep="first")
        .reset_index(drop=True)
    )
    work["live_broadcast_content"] = ""
    work["live_status"] = ""

    candidates: list[tuple[str, str, float]] = []
    for path, label in (
        (ckpt_path, f"断点 {ckpt_path}"),
        (output_path, f"已有输出 {output_path}"),
    ):
        if os.path.exists(path) and os.path.abspath(path) != os.path.abspath(input_path):
            candidates.append((path, label, os.path.getmtime(path)))
    if candidates and not args.overwrite:
        path, label, _ = max(candidates, key=lambda x: x[2])
        print(f"[续跑] 合并进度自 {label}")
        prog = _read_table(path)
        if "video_id" in prog.columns:
            keep = [c for c in CHECKPOINT_COLS if c in prog.columns]
            prog = (
                prog[keep]
                .astype(str)
                .fillna("")
                .assign(video_id=lambda d: d["video_id"].str.strip())
                .drop_duplicates(subset=["video_id"], keep="last")
            )
            work = work.drop(columns=["live_broadcast_content", "live_status"], errors="ignore")
            work = work.merge(prog, on="video_id", how="left")
            for col in ("live_broadcast_content", "live_status"):
                if col not in work.columns:
                    work[col] = ""
                work[col] = work[col].fillna("")

    pending = get_pending_mask(work, overwrite=args.overwrite, retry_errors=args.retry_errors)
    ids = work.loc[pending, "video_id"].astype(str).str.strip().tolist()
    # 去重保序（pending 已在 unique work 上）
    if args.limit is not None and args.limit > 0:
        ids = ids[: args.limit]
    print(f"输入 {len(base)} 行 / unique {len(work)}，待查 {len(ids)} 个 video_id")

    if ids:
        MAX_RETRIES = 3
        RETRY_SLEEP = 2.0
        done = 0
        # index 映射：O(1) 更新瘦表
        idx_map = {vid: i for i, vid in enumerate(work["video_id"].astype(str))}
        for i in range(0, len(ids), BATCH_SIZE):
            chunk = ids[i : i + BATCH_SIZE]
            last_err = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    result = fetch_live_broadcast_batch(api_key, chunk)
                    break
                except RuntimeError as e:
                    last_err = e
                    if attempt < MAX_RETRIES:
                        print(f"  批次重试 {attempt}/{MAX_RETRIES}: {e}", flush=True)
                        time.sleep(RETRY_SLEEP * attempt)
            else:
                print(f"[ERROR] 批次失败（{MAX_RETRIES}次重试后）: {last_err}（进度已保存，可断点续跑）")
                flush_checkpoint(work, ckpt_path)
                sys.exit(1)
            for vid, meta in result.items():
                j = idx_map.get(vid)
                if j is None:
                    continue
                work.at[j, "live_broadcast_content"] = meta["live_broadcast_content"]
                work.at[j, "live_status"] = meta["live_status"]
            done += len(chunk)
            if args.sleep > 0 and i + BATCH_SIZE < len(ids):
                time.sleep(args.sleep)
            if i % (BATCH_SIZE * 10) == 0 or done == len(ids):
                flush_checkpoint(work, ckpt_path)
                print(f"  进度 {done}/{len(ids)}", flush=True)

    # 合并回全量输入
    base = base.copy()
    base["video_id"] = base["video_id"].astype(str).str.strip()
    for col in ("live_broadcast_content", "live_status"):
        if col in base.columns:
            base = base.drop(columns=[col])
    out = base.merge(
        work[["video_id", "live_broadcast_content", "live_status"]],
        on="video_id",
        how="left",
    )
    for col in ("live_broadcast_content", "live_status"):
        out[col] = out[col].fillna("")

    write_final_csv(out, output_path, ckpt_path)
    print(f"完成: {output_path}")
    print(out["live_broadcast_content"].fillna("").value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

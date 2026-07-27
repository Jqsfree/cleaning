#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
"""
fetch_yt_definition.py — YouTube Data API contentDetails.definition (hd/sd) + ≥720 过滤

用途: 快速判断是否 ≥720p（hd）或仅标清（sd）。不能区分 720/1080/4K。

用法:
  export YOUTUBE_API_KEY='你的密钥'
  python3 02_脚本/tools/fetch_yt_definition.py input.csv -o out_yt_definition.csv -n 100
  python3 02_脚本/tools/fetch_yt_definition.py input.csv -o out_yt_definition.csv --filter-ge720
  python3 02_脚本/tools/fetch_yt_definition.py input.csv --filter-only \\
    --definition out_yt_definition.csv -o deliver_大于720.csv
  python3 02_脚本/tools/fetch_yt_definition.py --batch-list files.txt -o data/runs/.../06_tools/

输出列: yt_definition / is_hd / definition_status
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.progress import ThrottledProgress, mark_done  # noqa: E402
from core.sop import write_run_log  # noqa: E402

API_URL = "https://www.googleapis.com/youtube/v3/videos"
BATCH_SIZE = 50
DEFAULT_SLEEP = 0.05
CHECKPOINT_COLS = ("video_id", "yt_definition", "is_hd", "definition_status")
TERMINAL = {"ok", "not_found"}


def resolve_api_key(cli_key: str | None) -> str:
    key = (cli_key or os.getenv("YOUTUBE_API_KEY", "") or "").strip()
    if not key:
        print("[ERROR] 未设置 API Key。请先: export YOUTUBE_API_KEY='...'")
        sys.exit(1)
    return key


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path).astype(str).fillna("")
    return pd.read_csv(path, dtype=str).fillna("")


def resolve_output_path(input_path: str, output_arg: str | None) -> tuple[str, str]:
    if output_arg:
        final = output_arg
    else:
        stem, ext = os.path.splitext(input_path)
        if ext.lower() in (".parquet", ".pq"):
            final = f"{stem}_yt_definition.csv"
        else:
            final = f"{stem}_yt_definition{ext or '.csv'}"
    return final + ".ckpt.csv", final


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("yt_definition", "is_hd", "definition_status"):
        if col not in df.columns:
            df[col] = ""
    return df


def load_input_df(input_path: str, ckpt_path: str, final_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        print(f"[ERROR] 文件不存在: {input_path}")
        sys.exit(1)

    base = _read_table(input_path)
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
    for col in ("yt_definition", "is_hd", "definition_status"):
        if col not in progress.columns:
            progress[col] = ""

    if "video_id" not in base.columns or "video_id" not in progress.columns:
        print(f"[续跑] {label}（无 video_id，直接用进度文件）")
        return progress

    print(f"[续跑] 合并进度自 {label}")
    prog = (
        progress[["video_id", "yt_definition", "is_hd", "definition_status"]]
        .astype(str)
        .fillna("")
        .drop_duplicates(subset=["video_id"], keep="last")
    )
    base = base.drop(
        columns=[c for c in ("yt_definition", "is_hd", "definition_status") if c in base.columns]
    )
    merged = base.merge(prog, on="video_id", how="left")
    for col in ("yt_definition", "is_hd", "definition_status"):
        merged[col] = merged[col].fillna("")
    return merged


def get_pending_mask(df: pd.DataFrame, *, overwrite: bool, retry_errors: bool) -> pd.Index:
    if overwrite:
        df["yt_definition"] = ""
        df["is_hd"] = ""
        df["definition_status"] = ""
        return df.index

    status = df["definition_status"].fillna("").astype(str).str.strip()
    mask = ~status.isin(TERMINAL)
    if not retry_errors:
        mask = mask & (status != "error")
    return df[mask].index


def flush_checkpoint(df: pd.DataFrame, ckpt_path: str) -> None:
    cols = [c for c in CHECKPOINT_COLS if c in df.columns]
    mask = df["definition_status"].fillna("").astype(str).str.strip() != ""
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


def filter_ge720(
    orig_path: str,
    definition_path: str,
    keep_path: str,
    drop_path: str | None = None,
    *,
    strip_aux: bool = True,
) -> tuple[int, int]:
    """按 is_hd=1 过滤原表。返回 (n_keep, n_drop)。"""
    df_o = _read_table(orig_path)
    df_d = _read_table(definition_path)
    if "video_id" not in df_o.columns or "video_id" not in df_d.columns:
        raise ValueError("原表与 definition 均需 video_id 列")
    if "is_hd" not in df_d.columns:
        raise ValueError("definition 表缺少 is_hd 列")

    hd_ids = set(df_d.loc[df_d["is_hd"].astype(str).str.strip() == "1", "video_id"].astype(str))
    if strip_aux:
        drop_cols = [c for c in ("yt_definition", "is_hd", "definition_status") if c in df_o.columns]
        if drop_cols:
            df_o = df_o.drop(columns=drop_cols)

    mask = df_o["video_id"].astype(str).isin(hd_ids)
    keep = df_o.loc[mask]
    drop = df_o.loc[~mask]

    os.makedirs(os.path.dirname(os.path.abspath(keep_path)) or ".", exist_ok=True)
    keep.to_csv(keep_path, index=False)
    if drop_path:
        os.makedirs(os.path.dirname(os.path.abspath(drop_path)) or ".", exist_ok=True)
        drop.to_csv(drop_path, index=False)
    return int(mask.sum()), int((~mask).sum())


def _default_ge720_paths(definition_path: str) -> tuple[str, str]:
    stem, _ = os.path.splitext(definition_path)
    if stem.endswith("_yt_definition"):
        base = stem[: -len("_yt_definition")]
    else:
        base = stem
    return f"{base}_大于720.csv", f"{base}_低于720或缺失.csv"


def fetch_definitions_batch(api_key: str, video_ids: list[str], timeout: float = 60.0) -> dict[str, dict[str, str]]:
    params = urllib.parse.urlencode(
        {
            "part": "contentDetails",
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
        definition = str((item.get("contentDetails") or {}).get("definition") or "").strip().lower()
        if definition in ("hd", "sd"):
            found[vid] = {
                "yt_definition": definition,
                "is_hd": "1" if definition == "hd" else "0",
                "definition_status": "ok",
            }
        else:
            found[vid] = {
                "yt_definition": "",
                "is_hd": "",
                "definition_status": "error",
            }

    out: dict[str, dict[str, str]] = {}
    for vid in video_ids:
        if vid in found:
            out[vid] = found[vid]
        else:
            out[vid] = {
                "yt_definition": "",
                "is_hd": "",
                "definition_status": "not_found",
            }
    return out


def smoke_api_key(api_key: str) -> None:
    print("冒烟: videos.list jNQXAC9IVRw …")
    try:
        result = fetch_definitions_batch(api_key, ["jNQXAC9IVRw"])
    except Exception as e:
        print(f"[ERROR] API 冒烟失败: {e}")
        sys.exit(1)
    row = result.get("jNQXAC9IVRw") or {}
    print(f"  → definition={row.get('yt_definition')!r} status={row.get('definition_status')!r}")
    if row.get("definition_status") != "ok":
        print("[ERROR] 冒烟未拿到 ok")
        sys.exit(1)
    print("  冒烟通过")


def _run_filter(
    orig: str,
    definition: str,
    keep_path: str,
    drop_path: str | None,
    *,
    strip_aux: bool,
) -> None:
    if not drop_path:
        _, drop_path = _default_ge720_paths(definition)
        if "大于720" in keep_path:
            drop_path = keep_path.replace("_大于720", "_低于720或缺失")
        elif "ge720" in keep_path:
            drop_path = keep_path.replace("ge720", "lt720")
    n_keep, n_drop = filter_ge720(
        orig, definition, keep_path, drop_path, strip_aux=strip_aux,
    )
    print(f"过滤 ≥720: 保留 {n_keep} → {keep_path}")
    print(f"过滤剔除: {n_drop} → {drop_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YouTube Data API definition (hd/sd) + 可选 ≥720 过滤",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "见 AGENTS.md：清晰度为横切工具，交付前按需调用。\n"
            "  --filter-ge720  拉取后写出 *_大于720.csv\n"
            "  --filter-only   仅过滤（需 --definition）\n"
            "  --batch-list    多文件顺序拉取（-o 为目录）\n"
        ),
    )
    p.add_argument("input", nargs="?", default=None, help="输入 CSV/Parquet")
    p.add_argument("-o", "--output", default=None, help="definition 输出；filter-only 时为 keep 路径")
    p.add_argument("-n", "--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--api-key", default=None)
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    p.add_argument("--no-smoke", action="store_true")
    p.add_argument("--filter-ge720", action="store_true", help="拉取后按 is_hd=1 过滤")
    p.add_argument("--filter-only", action="store_true", help="不调 API，只过滤")
    p.add_argument("--definition", default=None, help="已有 definition CSV")
    p.add_argument("--drop-output", default=None, help="剔除表路径")
    p.add_argument("--batch-list", default=None, help="每行一个输入路径；-o 为目录")
    p.add_argument("--keep-aux-cols", action="store_true", help="keep 表保留 definition 列")
    return p


def run_one_fetch(args: argparse.Namespace, input_path: str, final_path: str) -> int:
    api_key = resolve_api_key(args.api_key)
    if not getattr(args, "_smoked", False) and not args.no_smoke:
        smoke_api_key(api_key)
        args._smoked = True

    ckpt_path = final_path + ".ckpt.csv"
    print(f"输入: {input_path}")
    print(f"输出: {final_path}")

    df = load_input_df(input_path, ckpt_path, final_path)
    if "video_id" not in df.columns:
        print(f"[ERROR] 未找到 video_id 列: {', '.join(df.columns)}")
        sys.exit(1)
    df = ensure_columns(df)

    pending_idx = get_pending_mask(df, overwrite=args.overwrite, retry_errors=args.retry_errors)
    if args.limit is not None and args.limit > 0:
        pending_idx = pending_idx[: args.limit]

    total_pending = len(pending_idx)
    out_dir = os.path.dirname(os.path.abspath(final_path)) or "."

    if total_pending == 0:
        print("没有待处理的行。")
        write_final_csv(df, final_path, ckpt_path)
        if args.filter_ge720:
            keep_p, drop_p = _default_ge720_paths(final_path)
            if args.drop_output:
                drop_p = args.drop_output
            _run_filter(input_path, final_path, keep_p, drop_p, strip_aux=not args.keep_aux_cols)
        return 0

    n_batches = (total_pending + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"待处理: {total_pending} 条 ≈ {n_batches} 批")

    prog = ThrottledProgress(
        out_dir, "yt_definition",
        interval_sec=5.0, every_n=200,
        input=input_path, output=final_path, total=total_pending,
    )
    stats = {"ok": 0, "not_found": 0, "error": 0, "hd": 0, "sd": 0}
    t0 = time.monotonic()
    done = 0
    interrupted = False

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_pending, desc="yt_definition", unit="条")
    except ImportError:
        pbar = None

    try:
        for start in range(0, total_pending, BATCH_SIZE):
            batch_indices = list(pending_idx[start : start + BATCH_SIZE])
            ids: list[str] = []
            idx_for_id: dict[str, list[int]] = {}
            for idx in batch_indices:
                vid = str(df.at[idx, "video_id"]).strip()
                if not vid or vid.lower() == "nan":
                    df.at[idx, "definition_status"] = "error"
                    stats["error"] += 1
                    done += 1
                    if pbar:
                        pbar.update(1)
                    continue
                ids.append(vid)
                idx_for_id.setdefault(vid, []).append(idx)

            unique_ids = list(dict.fromkeys(ids))
            if unique_ids:
                try:
                    results = fetch_definitions_batch(api_key, unique_ids)
                except Exception as e:
                    print(f"\n[WARN] 批次失败: {e}")
                    results = {
                        vid: {"yt_definition": "", "is_hd": "", "definition_status": "error"}
                        for vid in unique_ids
                    }

                for vid, rows in idx_for_id.items():
                    r = results.get(vid) or {
                        "yt_definition": "", "is_hd": "", "definition_status": "error",
                    }
                    for idx in rows:
                        df.at[idx, "yt_definition"] = r["yt_definition"]
                        df.at[idx, "is_hd"] = r["is_hd"]
                        df.at[idx, "definition_status"] = r["definition_status"]
                        stats[r["definition_status"]] = stats.get(r["definition_status"], 0) + 1
                        if r["yt_definition"] == "hd":
                            stats["hd"] += 1
                        elif r["yt_definition"] == "sd":
                            stats["sd"] += 1
                        done += 1
                        if pbar:
                            pbar.update(1)

            if done % 500 < BATCH_SIZE or start + BATCH_SIZE >= total_pending:
                flush_checkpoint(df, ckpt_path)
            if pbar:
                pbar.set_postfix(
                    ok=stats["ok"], nf=stats["not_found"], err=stats["error"],
                    hd=stats["hd"], sd=stats["sd"],
                )
            prog.tick(done=done, **stats)
            if args.sleep > 0 and start + BATCH_SIZE < total_pending:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[中断] 进度: {ckpt_path}")
    finally:
        if pbar:
            pbar.close()
        flush_checkpoint(df, ckpt_path)

    if interrupted:
        return 130

    elapsed = time.monotonic() - t0
    write_final_csv(df, final_path, ckpt_path)
    print(f"\n输出: {final_path}")
    print(
        f"耗时: {elapsed:.0f}s | ok={stats['ok']} hd={stats['hd']} sd={stats['sd']} "
        f"not_found={stats['not_found']} error={stats['error']}"
    )
    mark_done(
        out_dir, "yt_definition",
        input=input_path, output=final_path,
        done=done, total=total_pending,
        elapsed_sec=round(elapsed, 1), **stats,
    )
    write_run_log(
        "yt_definition", input_path, out_dir,
        stats={**stats, "pending": total_pending, "elapsed_sec": round(elapsed, 1)},
        command=f"fetch_yt_definition.py {input_path} -o {final_path}",
    )
    if args.filter_ge720:
        keep_p, drop_p = _default_ge720_paths(final_path)
        if args.drop_output:
            drop_p = args.drop_output
        _run_filter(input_path, final_path, keep_p, drop_p, strip_aux=not args.keep_aux_cols)
    return 0


def main() -> None:
    args = build_arg_parser().parse_args()
    args._smoked = False

    if args.filter_only:
        if not args.input or not args.definition:
            print("[ERROR] --filter-only 需要 input 与 --definition")
            sys.exit(2)
        keep_path = args.output
        if not keep_path:
            keep_path, _ = _default_ge720_paths(args.definition)
        _run_filter(
            args.input, args.definition, keep_path, args.drop_output,
            strip_aux=not args.keep_aux_cols,
        )
        out_dir = os.path.dirname(os.path.abspath(keep_path)) or "."
        write_run_log(
            "yt_definition_filter", args.input, out_dir,
            stats={"keep": keep_path, "definition": args.definition},
            command="fetch_yt_definition.py --filter-only",
        )
        return

    if args.batch_list:
        if not os.path.exists(args.batch_list):
            print(f"[ERROR] 不存在: {args.batch_list}")
            sys.exit(1)
        lines = [
            ln.strip() for ln in open(args.batch_list, encoding="utf-8")
            if ln.strip() and not ln.strip().startswith("#")
        ]
        out_dir = args.output or "."
        if str(out_dir).lower().endswith(".csv"):
            print("[ERROR] --batch-list 时 -o 应为目录")
            sys.exit(2)
        os.makedirs(out_dir, exist_ok=True)
        for inp in lines:
            stem = os.path.splitext(os.path.basename(inp))[0]
            final = os.path.join(out_dir, f"{stem}_yt_definition.csv")
            rc = run_one_fetch(args, inp, final)
            if rc != 0:
                sys.exit(rc)
        return

    if not args.input:
        print("[ERROR] 需要 input，或 --batch-list / --filter-only")
        sys.exit(2)

    _, final_path = resolve_output_path(args.input, args.output)
    sys.exit(run_one_fetch(args, args.input, final_path))


if __name__ == "__main__":
    main()

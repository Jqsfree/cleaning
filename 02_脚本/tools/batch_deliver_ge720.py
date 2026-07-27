#!/usr/bin/env python3
"""
batch_deliver_ge720.py — 对 keep 表：拉 definition（可跳过）→ 过滤 ≥720 → 写入 07_deliver + manifest

用法:
  export YOUTUBE_API_KEY=...
  02_脚本/tools/batch_deliver_ge720.py path/to/keep.csv \\
    --batch-root data/runs/film_tv/human_0727 --batch-id 0727

  # 已有 definition，只过滤落地:
  02_脚本/tools/batch_deliver_ge720.py keep.csv \\
    --batch-root … --batch-id 0727 \\
    --definition …/06_tools/0727_yt_definition.csv --skip-fetch
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.run_manifest import load_manifest, update_stage  # noqa: E402
from core.sop import write_run_log  # noqa: E402

FETCH = Path(__file__).resolve().parent / "fetch_yt_definition.py"


def main() -> None:
    p = argparse.ArgumentParser(description="一键 definition + ≥720 交付到 batch 07_deliver/")
    p.add_argument("input", help="quality keep 或 clean keep CSV")
    p.add_argument("--batch-root", required=True, help="批次根，如 data/runs/film_tv/human_0727")
    p.add_argument("--batch-id", required=True, help="批号，用于文件名，如 0727")
    p.add_argument("--definition", default=None, help="已有 yt_definition CSV（可与 --skip-fetch）")
    p.add_argument("--skip-fetch", action="store_true", help="不调 API，仅用 --definition 过滤")
    p.add_argument("--no-smoke", action="store_true")
    p.add_argument("-n", "--limit", type=int, default=None, help="拉取条数上限（测试）")
    args = p.parse_args()

    batch = Path(args.batch_root)
    tools = batch / "06_tools"
    deliver = batch / "07_deliver"
    tools.mkdir(parents=True, exist_ok=True)
    deliver.mkdir(parents=True, exist_ok=True)

    bid = args.batch_id.strip()
    defn_path = Path(args.definition) if args.definition else tools / f"{bid}_yt_definition.csv"
    keep_path = deliver / f"{bid}_deliver_ge720.csv"
    drop_path = tools / f"{bid}_低于720或缺失.csv"

    inp = Path(args.input)
    if not inp.exists():
        print(f"[ERROR] 输入不存在: {inp}")
        sys.exit(1)

    if args.skip_fetch:
        if not defn_path.exists():
            print(f"[ERROR] --skip-fetch 需要已有 definition: {defn_path}")
            sys.exit(1)
        cmd = [
            sys.executable, str(FETCH), str(inp),
            "--filter-only",
            "--definition", str(defn_path),
            "-o", str(keep_path),
            "--drop-output", str(drop_path),
        ]
        print(" ", " ".join(cmd))
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            sys.exit(rc)
    else:
        cmd = [
            sys.executable, str(FETCH), str(inp),
            "-o", str(defn_path),
            "--filter-ge720",
            "--drop-output", str(drop_path),
        ]
        if args.no_smoke:
            cmd.append("--no-smoke")
        if args.limit:
            cmd.extend(["-n", str(args.limit)])
        print(" ", " ".join(cmd))
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            sys.exit(rc)
        # filter-ge720 默认写在 definition 旁 *_大于720.csv → 拷到 deliver 标准名
        auto_keep = defn_path.with_name(
            defn_path.name.replace("_yt_definition.csv", "_大于720.csv")
            if defn_path.name.endswith("_yt_definition.csv")
            else defn_path.stem + "_大于720.csv"
        )
        if not auto_keep.exists():
            # fetch 也可能写出 stem_大于720
            cand = list(tools.glob(f"{bid}*大于720*.csv")) + list(defn_path.parent.glob("*大于720*.csv"))
            auto_keep = cand[0] if cand else auto_keep
        if auto_keep.exists() and auto_keep.resolve() != keep_path.resolve():
            shutil.copy2(auto_keep, keep_path)
            print(f"交付拷贝: {auto_keep} → {keep_path}")
        elif not keep_path.exists() and auto_keep.exists():
            shutil.copy2(auto_keep, keep_path)

    if not keep_path.exists():
        print(f"[ERROR] 未找到交付文件: {keep_path}")
        sys.exit(1)

    if load_manifest(batch):
        try:
            update_stage(
                batch,
                "tools_ge720",
                paths={
                    "definition": str(defn_path),
                    "deliver": str(keep_path),
                    "drop": str(drop_path),
                },
                deliver_path=str(keep_path),
            )
        except Exception as e:
            print(f"[WARN] manifest 更新失败: {e}")
    else:
        print(f"[WARN] 无 manifest.json，跳过索引更新（可先 run_manifest.py init）")

    write_run_log(
        "yt_definition_filter",
        str(inp),
        str(batch),
        stats={"deliver": str(keep_path), "definition": str(defn_path)},
        command=f"batch_deliver_ge720.py {inp} --batch-root {batch}",
    )
    print(f"完成 deliver → {keep_path}")


if __name__ == "__main__":
    main()

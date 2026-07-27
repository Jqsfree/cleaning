#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
"""
pipeline/run.py — 按采集来源薄编排（默认仅 quality，不自动 clean）

人工采 / 机采主链见 AGENTS.md。本脚本只串联 quality（可选 clean，有门禁）。

用法:
  02_脚本/pipeline/run.py raw/xxx.csv --category film_tv --source human \\
    -o data/runs/film_tv/human_0724/
  02_脚本/pipeline/run.py raw/xxx.csv --category film_tv --source machine \\
    -o data/runs/film_tv/machine_0724/ --stages quality
  # 机采在规则就绪后才可加 clean：
  02_脚本/pipeline/run.py … --source machine --stages quality,clean --rules-ready
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_PIPE = Path(__file__).resolve().parent
_ROOT = _PIPE.parent
sys.path.insert(0, str(_ROOT))

from core.log import log  # noqa: E402
from core.run_manifest import init_manifest, update_stage  # noqa: E402
from core.sop import print_banner  # noqa: E402

QUALITY = _PIPE / "01_quality.py"
CLEAN = _PIPE / "02_clean.py"

_CLEAN_CATEGORIES = ("language_teaching", "beauty", "welding", "film_tv")


def _run(cmd: list[str]) -> None:
    log(" ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def _latest_quality_keep(out_dir: Path) -> Path | None:
    cands = sorted(out_dir.glob("*_quality_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    cands = [p for p in cands if "_quality_drop_" not in p.name]
    if not cands:
        # 宽松：任意非 drop 的 csv
        cands = sorted(
            [p for p in out_dir.glob("*.csv") if "drop" not in p.name.lower()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    return cands[0] if cands else None


def _infer_batch(out: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    name = out.name  # e.g. human_0724
    if "_" in name:
        return name.split("_", 1)[1]
    return name


def main() -> None:
    p = argparse.ArgumentParser(
        description="薄编排：按 --source 跑 quality（默认）；clean 需显式门禁",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "人工采: 默认仅 quality → 再 sample → 人工质检；合格勿 clean。\n"
            "机采: 默认仅 quality → sample → 文本 QC → 改规则后再 --stages clean --rules-ready。\n"
            "详见 AGENTS.md 双路径。"
        ),
    )
    p.add_argument("input", help="原始 CSV")
    p.add_argument("-c", "--category", required=True, choices=_CLEAN_CATEGORIES)
    p.add_argument(
        "--source", required=True, choices=("human", "machine"),
        help="采集来源：human=人工采，machine=机采",
    )
    p.add_argument(
        "-o", "--output-dir", required=True,
        help="批次根目录，如 data/runs/film_tv/human_0724/",
    )
    p.add_argument(
        "--stages", default="quality",
        help="逗号分隔：quality[,clean]（默认仅 quality）",
    )
    p.add_argument("--batch", default=None, help="批号（默认从目录名 {source}_{batch} 推断）")
    p.add_argument("--min", type=int, default=60, help="quality 最小时长")
    p.add_argument("--max", type=int, default=14400, help="quality 最大时长")
    p.add_argument("-r", "--run", default="run01", help="clean run 名")
    p.add_argument(
        "--rules-ready", action="store_true",
        help="机采：确认已抽样文本质检并更新/确认 TOML 规则，允许 clean",
    )
    p.add_argument(
        "--allow-clean", action="store_true",
        help="人工采：确认输入为不合格集，允许 clean",
    )
    args = p.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in ("quality", "clean")]
    if unknown:
        log(f"不支持的 stages: {unknown}（仅 quality,clean）", level="ERROR")
        sys.exit(2)

    if "clean" in stages:
        if args.source == "machine" and not args.rules_ready:
            log(
                "机采 clean 须先 sample→文本 QC→规则；请加 --rules-ready，或先去掉 clean",
                level="ERROR",
            )
            sys.exit(2)
        if args.source == "human" and not args.allow_clean:
            log(
                "人工采默认合格直交付；仅不合格集可 clean，请加 --allow-clean",
                level="ERROR",
            )
            sys.exit(2)

    out = Path(args.output_dir)
    batch = _infer_batch(out, args.batch)
    q_dir = out / "01_quality"
    c_dir = out / "05_clean" / args.run

    print_banner("pipeline", category=args.category)
    log(f"source={args.source}  stages={stages}  batch={batch}")
    log("默认不自动 clean；完整双路径见 AGENTS.md")

    try:
        init_manifest(
            out,
            category=args.category,
            source=args.source,
            batch=batch,
            input_path=args.input,
        )
    except Exception as e:
        log(f"manifest init 跳过: {e}", level="WARN")

    keep_path: Path | None = None
    if "quality" in stages:
        q_dir.mkdir(parents=True, exist_ok=True)
        _run([
            sys.executable, str(QUALITY), args.input,
            "-o", str(q_dir),
            "--min", str(args.min),
            "--max", str(args.max),
        ])
        keep_path = _latest_quality_keep(q_dir)
        if keep_path is None:
            log(f"初筛后未找到 keep CSV: {q_dir}", level="ERROR")
            sys.exit(1)
        log(f"quality keep: {keep_path}")
        try:
            update_stage(out, "quality", paths={"keep": str(keep_path)})
        except Exception as e:
            log(f"manifest update 跳过: {e}", level="WARN")

    if "clean" in stages:
        if keep_path is None:
            keep_path = Path(args.input)
            if not keep_path.exists():
                log(f"clean 输入不存在: {keep_path}", level="ERROR")
                sys.exit(1)
        c_dir.mkdir(parents=True, exist_ok=True)
        clean_cmd = [
            sys.executable, str(CLEAN), str(keep_path),
            "--category", args.category,
            "--source", args.source,
            "-o", str(c_dir),
            "-r", args.run,
        ]
        if args.source == "machine":
            clean_cmd.append("--rules-ready")
        if args.source == "human":
            clean_cmd.append("--allow-clean")
        _run(clean_cmd)
        try:
            update_stage(out, "clean", paths={"dir": str(c_dir)})
        except Exception as e:
            log(f"manifest update 跳过: {e}", level="WARN")

    log(f"完成 → {out}")


if __name__ == "__main__":
    main()

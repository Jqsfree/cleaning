#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
"""
pipeline/run.py — 按采集来源薄编排（默认仅 quality，不自动 clean）

人工采 / 机采主链见 AGENTS.md。本脚本只串联 quality（可选 clean，有门禁）。
sample / QC / deliver 用 tools/run_manifest.py checklist 自查，不自动跑。

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
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_PIPE = Path(__file__).resolve().parent
_ROOT = _PIPE.parent
sys.path.insert(0, str(_ROOT))

from core.category_registry import list_cleaner_categories  # noqa: E402
from core.clean_gates import assert_clean_gates, assert_clean_not_raw  # noqa: E402
from core.log import log  # noqa: E402
from core.run_manifest import init_manifest  # noqa: E402
from core.sop import print_banner  # noqa: E402

QUALITY_PY = _PIPE / "01_quality.py"
CLEAN_PY = _PIPE / "02_clean.py"


def _load_pipeline_mod(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载: {path}")
    mod = importlib.util.module_from_spec(spec)
    # 保证同进程内相对 import / 单例
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _latest_quality_keep(out_dir: Path) -> Path | None:
    cands = sorted(out_dir.glob("*_quality_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    cands = [p for p in cands if "_quality_drop_" not in p.name]
    if not cands:
        cands = sorted(
            [p for p in out_dir.glob("*.csv") if "drop" not in p.name.lower()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    return cands[0] if cands else None


def _infer_batch(out: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    name = out.name
    if "_" in name:
        return name.split("_", 1)[1]
    return name


def main() -> None:
    cats = list_cleaner_categories()
    p = argparse.ArgumentParser(
        description="薄编排：按 --source 跑 quality（默认）；clean 需显式门禁",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "人工采: 默认仅 quality → 再 sample → 人工质检；合格勿 clean。\n"
            "机采: 默认仅 quality → sample → 文本 QC → 改规则后再 --stages clean --rules-ready。\n"
            "自查: 02_脚本/tools/run_manifest.py checklist -o <batch>/\n"
            "详见 AGENTS.md 双路径。"
        ),
    )
    p.add_argument("input", help="原始 CSV")
    p.add_argument("-c", "--category", required=True, choices=cats)
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
    p.add_argument(
        "--skip-evidence", action="store_true",
        help="跳过 clean 证据门禁（临时）",
    )
    p.add_argument(
        "--reinit-manifest", action="store_true",
        help="重建 manifest（清空已有 stages）；默认 merge 保留",
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
            reinit=args.reinit_manifest,
        )
    except Exception as e:
        log(f"manifest init 跳过: {e}", level="WARN")

    keep_path: Path | None = None
    ran_quality = False
    if "quality" in stages:
        q_dir.mkdir(parents=True, exist_ok=True)
        quality_mod = _load_pipeline_mod("pipeline_01_quality", QUALITY_PY)
        quality_mod.quality_check(
            args.input,
            str(q_dir),
            min_duration=args.min,
            max_duration=args.max,
        )
        ran_quality = True
        keep_path = _latest_quality_keep(q_dir)
        if keep_path is None:
            log(f"初筛后未找到 keep CSV: {q_dir}", level="ERROR")
            sys.exit(1)
        log(f"quality keep: {keep_path}")

    if "clean" in stages:
        if keep_path is None:
            keep_path = Path(args.input)
            if not keep_path.exists():
                log(f"clean 输入不存在: {keep_path}", level="ERROR")
                sys.exit(1)
            assert_clean_not_raw(
                keep_path=keep_path,
                ran_quality=ran_quality,
                source=args.source,
                skip_evidence=args.skip_evidence,
            )
        if args.source == "human" and ran_quality and not args.skip_evidence:
            log(
                "人工经 run.py 对 quality keep 做 clean 不合 SOP；"
                "请对 03_qc/fail.csv 单独 02_clean，或加 --skip-evidence",
                level="ERROR",
            )
            sys.exit(2)

        c_dir.mkdir(parents=True, exist_ok=True)
        assert_clean_gates(
            source=args.source,
            input_path=str(keep_path),
            output_dir=str(c_dir),
            rules_ready=args.rules_ready,
            allow_clean=args.allow_clean,
            skip_evidence=args.skip_evidence,
        )

        clean_mod = _load_pipeline_mod("pipeline_02_clean", CLEAN_PY)
        clean_mod.run_clean(
            str(keep_path),
            args.category,
            str(c_dir),
            source=args.source,
            rules_ready=args.rules_ready,
            allow_clean=args.allow_clean,
            skip_evidence=args.skip_evidence,
            run=args.run,
            # run.py 已做过门禁；子调用再做一次无害，保持与 standalone 一致
            enforce_gates=True,
        )

    log(f"完成 → {out}")
    log(f"自查: 02_脚本/tools/run_manifest.py checklist -o {out}/")


if __name__ == "__main__":
    main()

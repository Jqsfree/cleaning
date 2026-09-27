#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
"""
02_clean.py — 规则清洗：对基表应用黑/白名单规则

门禁（见 AGENTS.md 双路径）:
  机采 --source machine：须先抽样→文本质检→规则，并加 --rules-ready
  人工 --source human：仅不合格集，并加 --allow-clean
  必须传 --source；遗留脚本临时加 --legacy（跳过 source 校验）
  证据门禁默认开启；临时跳过加 --skip-evidence

用法:
  02_脚本/pipeline/02_clean.py quality.csv --category film_tv --source machine --rules-ready \\
    -o data/runs/film_tv/machine_0724/05_clean/run01/
  02_脚本/pipeline/02_clean.py human_fail.csv --category film_tv --source human --allow-clean \\
    -o data/runs/film_tv/human_0724/05_clean/run01/
"""

from __future__ import annotations

import sys
import os
import argparse
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.log import log, banner
from core.sop import write_run_log
from core.progress import mark_done
from core.batch_layout import require_output_dir, warn_outside_batch
from core.clean_gates import assert_clean_gates
from core.category_registry import list_cleaner_categories, load_cleaner
from core.run_manifest import maybe_update_stage


def run_clean(
    input_path: str,
    category: str,
    output_dir: str,
    *,
    source: str | None = None,
    legacy: bool = False,
    skip_evidence: bool = False,
    rules_ready: bool = False,
    allow_clean: bool = False,
    run: str = "run01",
    keep_score: int | None = None,
    gray_low: int | None = None,
    med_min: int | None = None,
    no_medium: bool = False,
    with_r2: bool = False,
    enforce_gates: bool = True,
) -> dict[str, Any]:
    """执行品类 clean；供 CLI 与 pipeline/run.py 直接调用。"""
    if not os.path.exists(input_path):
        log(f"文件不存在: {input_path}", level="ERROR")
        sys.exit(1)

    output_dir = require_output_dir(output_dir)
    warn_outside_batch(output_dir, log_fn=log)

    if enforce_gates:
        if source is None:
            if legacy:
                log(
                    "未指定 --source（--legacy）；请尽快改为 human|machine",
                    level="WARN",
                )
            else:
                log(
                    "必须指定 --source human|machine（见 AGENTS.md）；"
                    "遗留脚本临时加 --legacy",
                    level="ERROR",
                )
                sys.exit(2)
        elif source == "machine" and not rules_ready:
            log(
                "机采 clean 须 --rules-ready（确认已抽样文本质检并更新规则）",
                level="ERROR",
            )
            sys.exit(2)
        elif source == "human" and not allow_clean:
            log(
                "人工采 clean 须 --allow-clean（确认输入为不合格集，非合格交付集）",
                level="ERROR",
            )
            sys.exit(2)

        assert_clean_gates(
            source=source,
            input_path=input_path,
            output_dir=output_dir,
            rules_ready=rules_ready,
            allow_clean=allow_clean,
            skip_evidence=skip_evidence,
            legacy=legacy,
        )

    from core.io import strip_stem

    raw_stem = os.path.splitext(os.path.basename(input_path))[0]
    stem = strip_stem(raw_stem)
    clean_func = load_cleaner(category)

    summary = clean_func(
        input_path=input_path,
        stem=stem,
        output_dir=output_dir,
        raw_name=stem,
        run=run,
        keep_score=keep_score,
        gray_low=gray_low,
        med_min=med_min,
        no_medium=no_medium,
        with_r2=with_r2,
    )

    print()
    banner(f"规则清洗 完成 ({category})")
    log(f"总行数:  {summary['total_rows']:>12,}")
    log(f"保留:    {summary['total_keep']:>12,}  ({summary['retention_pct']}%)")
    if category == "language_teaching":
        log(f"  high:  {summary.get('total_keep_high', 0):>12,}")
        log(f"  medium:{summary.get('total_keep_medium', 0):>12,}")
    log(f"移除:    {summary['total_drop']:>12,}")
    log(f"耗时:    {summary['elapsed_sec']:>11.1f}s")
    log(f"产物:    {output_dir}/")
    banner("")

    stats: dict[str, Any] = {
        "total_rows": summary["total_rows"],
        "total_keep": summary["total_keep"],
        "total_drop": summary["total_drop"],
        "retention_pct": summary["retention_pct"],
        "elapsed_sec": summary["elapsed_sec"],
    }
    paths: dict[str, str] = {"dir": output_dir}
    if summary.get("keep_path"):
        stats["keep_path"] = summary["keep_path"]
        paths["keep"] = summary["keep_path"]
    if summary.get("drop_path"):
        stats["drop_path"] = summary["drop_path"]
        paths["drop"] = summary["drop_path"]
    if summary.get("keep_high_path"):
        stats["keep_high_path"] = summary["keep_high_path"]
        paths["keep_high"] = summary["keep_high_path"]
    if summary.get("keep_medium_path"):
        stats["keep_medium_path"] = summary["keep_medium_path"]
        paths["keep_medium"] = summary["keep_medium_path"]

    progress_extra = {
        "final": summary["total_keep"],
        "total": summary["total_rows"],
        "retain_pct": summary["retention_pct"],
        "elapsed_sec": summary["elapsed_sec"],
    }
    if category == "language_teaching":
        stats["keep_high"] = summary.get("total_keep_high", 0)
        stats["keep_medium"] = summary.get("total_keep_medium", 0)
        progress_extra["high"] = summary.get("total_keep_high", 0)
        progress_extra["medium"] = summary.get("total_keep_medium", 0)

    mark_done(output_dir, "clean", **progress_extra)
    write_run_log(
        "clean", input_path, output_dir, stats=stats,
        command=f"02_clean.py {input_path} --category {category} -o {output_dir}",
        category=category,
    )
    if maybe_update_stage(
        output_dir,
        "clean",
        paths=paths,
        stats={
            "total_rows": summary["total_rows"],
            "total_keep": summary["total_keep"],
            "total_drop": summary["total_drop"],
            "retention_pct": summary["retention_pct"],
        },
        category=category,
        provenance=__import__("core.provenance", fromlist=["build_provenance"]).build_provenance(
            input_path=input_path,
            rules_dir=str(
                Path(__file__).resolve().parent.parent
                / "categories" / category / "rules"
            ),
            extra={"category": category},
        ),
    ):
        log("manifest 已更新 stage=clean")
    keep_for_contract = summary.get("keep_path") or paths.get("keep")
    if keep_for_contract:
        try:
            from core.contracts import assert_contracts
            assert_contracts(
                keep_for_contract,
                layer="clean",
                category=category,
                upstream_rows=summary["total_rows"],
                soft=True,  # clean keep 可为 0
            )
        except Exception as e:
            log(f"契约校验跳过: {e}", level="WARN")
    return summary


def main():
    cats = list_cleaner_categories()
    parser = argparse.ArgumentParser(
        description="规则清洗：应用黑/白名单规则过滤",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "门禁:\n"
            "  --source machine 须同时 --rules-ready（抽样文本质检后已确认规则）\n"
            "  --source human 须同时 --allow-clean（输入应为人工质检不合格集）\n"
            "  必须传 --source；遗留脚本加 --legacy\n"
            "  默认校验批次证据；--skip-evidence 仅临时逃生\n"
        ),
    )
    parser.add_argument("input", help="基表 CSV/Parquet（quality keep 或不合格集）")
    parser.add_argument(
        "-c", "--category", required=True, choices=cats,
        help="类别名（必填，禁止默认 language_teaching）",
    )
    parser.add_argument(
        "--source", default=None, choices=("human", "machine"),
        help="采集来源（必填，除非 --legacy）",
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="逃生阀：允许不传 --source 且跳过证据门禁（将移除）",
    )
    parser.add_argument(
        "--skip-evidence", action="store_true",
        help="跳过批次证据校验（rules-ready / fail 输入）",
    )
    parser.add_argument(
        "--rules-ready", action="store_true",
        help="机采：已完成 sample→文本 QC→规则确认",
    )
    parser.add_argument(
        "--allow-clean", action="store_true",
        help="人工采：确认本输入为不合格集",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="输出目录（须 …/{source}_{batch}/05_clean/runNN/）",
    )
    parser.add_argument("--keep-score", type=int, default=None,
                        help="high 阈值（仅 language_teaching，默认从规则读取）")
    parser.add_argument("--gray-low", type=int, default=None,
                        help="gray 低分阈值（仅 language_teaching）")
    parser.add_argument("--med-min", type=int, default=None,
                        help="medium 最低分（仅 language_teaching）")
    parser.add_argument("-r", "--run", default="run01",
                        help="run 名称")
    parser.add_argument("--no-medium", action="store_true",
                        help="只保留 high，丢弃 medium（仅 language_teaching）")
    parser.add_argument("--with-r2", action="store_true",
                        help="启用 r2 二次过滤（仅 welding）")
    args = parser.parse_args()

    run_clean(
        args.input,
        args.category,
        args.output_dir,
        source=args.source,
        legacy=args.legacy,
        skip_evidence=args.skip_evidence,
        rules_ready=args.rules_ready,
        allow_clean=args.allow_clean,
        run=args.run,
        keep_score=args.keep_score,
        gray_low=args.gray_low,
        med_min=args.med_min,
        no_medium=args.no_medium,
        with_r2=args.with_r2,
    )


if __name__ == "__main__":
    main()

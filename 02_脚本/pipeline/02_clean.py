#!/home/jqs/miniconda3/envs/data_cleaning/bin/python
"""
02_clean.py — 规则清洗：对基表应用黑/白名单规则

门禁（见 AGENTS.md 双路径）:
  机采 --source machine：须先抽样→文本质检→规则，并加 --rules-ready
  人工 --source human：仅不合格集，并加 --allow-clean
  必须传 --source；遗留脚本临时加 --legacy（跳过 source 校验）

用法:
  02_脚本/pipeline/02_clean.py quality.csv --category film_tv --source machine --rules-ready \\
    -o data/runs/film_tv/machine_0724/05_clean/run01/
  02_脚本/pipeline/02_clean.py human_fail.csv --category film_tv --source human --allow-clean \\
    -o data/runs/film_tv/human_0724/05_clean/run01/
"""

import sys, os, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.log import log, banner
from core.sop import write_run_log
from core.progress import mark_done

# 按类别名动态导入 cleaner
_CLEANERS = {
    "language_teaching": "categories.language_teaching.cleaner",
    "beauty": "categories.beauty.cleaner",
    "welding": "categories.welding.cleaner",
    "film_tv": "categories.film_tv.cleaner",
}


def _load_cleaner(category: str):
    """动态加载类别 cleaner 模块。"""
    module_path = _CLEANERS.get(category)
    if module_path is None:
        log(f"未知类别: {category}", level="ERROR")
        log(f"  可用: {', '.join(sorted(_CLEANERS.keys()))}")
        sys.exit(1)

    import importlib
    try:
        mod = importlib.import_module(module_path)
    except (ImportError, ModuleNotFoundError) as e:
        log(f"无法加载类别模块: {module_path}", level="ERROR")
        log(f"  {e}")
        log(f"  检查 categories/{category}/cleaner.py 是否存在")
        sys.exit(1)
    if not hasattr(mod, 'clean'):
        log(f"类别模块缺少 clean() 函数: {module_path}", level="ERROR")
        sys.exit(1)
    return mod.clean


def main():
    parser = argparse.ArgumentParser(
        description="规则清洗：应用黑/白名单规则过滤",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "门禁:\n"
            "  --source machine 须同时 --rules-ready（抽样文本质检后已确认规则）\n"
            "  --source human 须同时 --allow-clean（输入应为人工质检不合格集）\n"
            "  必须传 --source；遗留脚本加 --legacy\n"
        ),
    )
    parser.add_argument("input", help="基表 CSV/Parquet（quality keep 或不合格集）")
    parser.add_argument("-c", "--category", default="language_teaching",
                        help="类别名（language_teaching, beauty, welding, film_tv）")
    parser.add_argument(
        "--source", default=None, choices=("human", "machine"),
        help="采集来源（必填，除非 --legacy）",
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="逃生阀：允许不传 --source（将移除）",
    )
    parser.add_argument(
        "--rules-ready", action="store_true",
        help="机采：已完成 sample→文本 QC→规则确认",
    )
    parser.add_argument(
        "--allow-clean", action="store_true",
        help="人工采：确认本输入为不合格集",
    )
    parser.add_argument("-o", "--output-dir", default=None,
                        help="输出目录（建议 …/05_clean/runNN/）")
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

    if not os.path.exists(args.input):
        log(f"文件不存在: {args.input}", level="ERROR")
        sys.exit(1)

    if args.source is None:
        if args.legacy:
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
    elif args.source == "machine" and not args.rules_ready:
        log(
            "机采 clean 须 --rules-ready（确认已抽样文本质检并更新规则）",
            level="ERROR",
        )
        sys.exit(2)
    elif args.source == "human" and not args.allow_clean:
        log(
            "人工采 clean 须 --allow-clean（确认输入为不合格集，非合格交付集）",
            level="ERROR",
        )
        sys.exit(2)

    category = args.category
    if args.output_dir is None:
        if category == "beauty":
            args.output_dir = "data/runs/beauty/05_clean/run01"
        elif category == "welding":
            args.output_dir = "data/runs/welding/05_clean/run01"
        elif category == "film_tv":
            args.output_dir = "data/runs/film_tv/05_clean/run01"
        else:
            args.output_dir = "data/runs/language_teaching/05_clean/run01"

    raw_stem = os.path.splitext(os.path.basename(args.input))[0]
    from core.io import strip_stem
    stem = strip_stem(raw_stem)

    clean_func = _load_cleaner(category)

    # 构建传给 cleaner 的参数
    clean_kwargs = dict(
        input_path=args.input,
        stem=stem,
        output_dir=args.output_dir,
        raw_name=stem,
        run=args.run,
        keep_score=args.keep_score,
        gray_low=args.gray_low,
        med_min=args.med_min,
        no_medium=args.no_medium,
        with_r2=args.with_r2,
    )

    summary = clean_func(**clean_kwargs)

    print()
    banner(f"规则清洗 完成 ({category})")
    log(f"总行数:  {summary['total_rows']:>12,}")
    log(f"保留:    {summary['total_keep']:>12,}  ({summary['retention_pct']}%)")
    if category == "language_teaching":
        log(f"  high:  {summary.get('total_keep_high', 0):>12,}")
        log(f"  medium:{summary.get('total_keep_medium', 0):>12,}")
    log(f"移除:    {summary['total_drop']:>12,}")
    log(f"耗时:    {summary['elapsed_sec']:>11.1f}s")
    log(f"产物:    {args.output_dir}/")
    banner("")

    stats = {"total_rows": summary["total_rows"],
             "total_keep": summary["total_keep"],
             "total_drop": summary["total_drop"],
             "retention_pct": summary["retention_pct"],
             "elapsed_sec": summary["elapsed_sec"]}
    if summary.get("keep_path"):
        stats["keep_path"] = summary["keep_path"]
    if summary.get("drop_path"):
        stats["drop_path"] = summary["drop_path"]
    if summary.get("keep_high_path"):
        stats["keep_high_path"] = summary["keep_high_path"]
    if summary.get("keep_medium_path"):
        stats["keep_medium_path"] = summary["keep_medium_path"]
    progress_extra = {"final": summary["total_keep"],
                      "total": summary["total_rows"],
                      "retain_pct": summary["retention_pct"],
                      "elapsed_sec": summary["elapsed_sec"]}
    if category == "language_teaching":
        stats["keep_high"] = summary.get("total_keep_high", 0)
        stats["keep_medium"] = summary.get("total_keep_medium", 0)
        progress_extra["high"] = summary.get("total_keep_high", 0)
        progress_extra["medium"] = summary.get("total_keep_medium", 0)

    mark_done(args.output_dir, "clean", **progress_extra)
    write_run_log("clean", args.input, args.output_dir, stats=stats,
                  command=f"02_clean.py {args.input} --category {category} -o {args.output_dir}",
                  category=category)


if __name__ == "__main__":
    main()

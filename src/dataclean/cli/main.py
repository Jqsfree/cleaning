"""
cli/main.py — WIP 包化入口（已冻结，非生产）

生产请用 ``02_脚本/pipeline/`` 与 AGENTS.md 双路径。
本 CLI 仅在显式 ``--allow-wip`` 下可用。

用法（实验）:
  python -m dataclean.cli.main --allow-wip list phases
"""

from __future__ import annotations

import argparse
import sys


def cmd_phase0(args):
    """Phase 0: 数据规范化"""
    from dataclean.phases.normalize import NormalizePhase
    phase = NormalizePhase()
    result = phase.execute(
        input_path=args.input,
        output_dir=args.output_dir,
        min_duration=args.min_duration,
    )
    return result


def cmd_phase2_sample(args):
    """Phase 2: 统计学抽样"""
    from dataclean.phases.sample import SamplePhase
    phase = SamplePhase()
    result = phase.execute(
        input_path=args.input,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    return result


def cmd_phase5(args):
    """Phase 5: 规则清洗"""
    from dataclean.phases.clean import CleanPhase
    phase = CleanPhase()
    result = phase.execute(
        input_path=args.input,
        output_dir=args.output_dir,
        category=args.category,
        run=args.run,
        keep_score=args.keep_score,
        gray_low=args.gray_low,
        med_min=args.med_min,
        no_medium=args.no_medium,
    )
    return result


def cmd_list_categories(args):
    """列出所有已注册的数据集类别"""
    from dataclean.pipeline.registry import CATEGORY_REGISTRY, list_categories
    print("\n已注册的数据集类别:")
    print("=" * 50)
    for name in list_categories():
        print(f"  {name:25s} → {CATEGORY_REGISTRY[name]}")
    print()


def cmd_list_phases(args):
    """列出 WIP 冻结阶段（非生产）"""
    from dataclean.pipeline.registry import list_phases
    print("\n管道阶段（WIP-FROZEN，非生产 SOP）:")
    print("=" * 50)
    for pid, info in list_phases().items():
        print(f"  Phase {pid}: {info['name']:15s} {info['desc']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="dataclean WIP CLI（已冻结）— 生产请用 02_脚本/"
    )
    parser.add_argument(
        "--allow-wip", action="store_true",
        help="确认使用冻结中的 WIP Phase CLI（非生产）",
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    p0 = sub.add_parser("phase0", help="[WIP] Phase 0: 数据规范化")
    p0.add_argument("input", help="原始 CSV 文件")
    p0.add_argument("-o", "--output-dir", default="data/runs/001_baseline/", help="输出目录")
    p0.add_argument("--min-duration", type=int, default=10, help="最小时长(秒)")
    p0.set_defaults(func=cmd_phase0)

    p2 = sub.add_parser("phase2", help="[WIP] Phase 2: 抽样质检")
    p2_sub = p2.add_subparsers(dest="phase2_cmd")

    p2_sample = p2_sub.add_parser("sample", help="统计学抽样")
    p2_sample.add_argument("input", help="baseline parquet 文件")
    p2_sample.add_argument("-o", "--output-dir", default="data/runs/002_audit/", help="输出目录")
    p2_sample.add_argument("-n", "--sample-size", type=int, default=None, help="样本量")
    p2_sample.add_argument("--seed", type=int, default=42, help="随机种子")
    p2_sample.set_defaults(func=cmd_phase2_sample)

    p5 = sub.add_parser("phase5", help="[WIP] Phase 5: 规则清洗")
    p5.add_argument("input", help="baseline parquet 文件")
    p5.add_argument("-c", "--category", default="language_teaching",
                    help="类别名（language_teaching, beauty, welding, film_tv）")
    p5.add_argument("-o", "--output-dir", default=None, help="输出目录")
    p5.add_argument("--keep-score", type=int, default=None, help="high 阈值")
    p5.add_argument("--gray-low", type=int, default=None, help="gray 低分阈值")
    p5.add_argument("--med-min", type=int, default=None, help="medium 最低分")
    p5.add_argument("-r", "--run", default="run01", help="run 名称")
    p5.add_argument("--no-medium", action="store_true", help="只保留 high")
    p5.set_defaults(func=cmd_phase5)

    p_list = sub.add_parser("list", help="列出注册信息")
    p_list_sub = p_list.add_subparsers(dest="list_cmd")
    p_list_cat = p_list_sub.add_parser("categories", help="列出所有类别")
    p_list_cat.set_defaults(func=cmd_list_categories)
    p_list_ph = p_list_sub.add_parser("phases", help="列出所有阶段")
    p_list_ph.set_defaults(func=cmd_list_phases)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        print(
            "\n[提示] 本入口已冻结。生产: 02_脚本/pipeline/run.py（见 AGENTS.md）",
            file=sys.stderr,
        )
        sys.exit(0)

    from dataclean.pipeline.registry import assert_wip_allowed
    assert_wip_allowed(allow_wip=bool(getattr(args, "allow_wip", False)))

    args.func(args)


if __name__ == "__main__":
    main()

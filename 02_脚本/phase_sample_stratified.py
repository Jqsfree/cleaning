#!/usr/bin/env python3
"""
phase_sample_stratified.py — 分层随机抽样

按指定数值列分桶，每层按比例随机抽样，输出抽样 CSV 并打印各层
置信区间（默认 90%）及总时长统计。

用法:
  python3 phase_sample_stratified.py pass.csv -o sample.csv
  python3 phase_sample_stratified.py pass.csv -o sample.csv --col duration_seconds --rate 0.9 --confidence 0.90
  python3 phase_sample_stratified.py pass.csv -o sample.csv --buckets 60,120,300,600,1800
"""

import argparse, csv, math, random, sys
from pathlib import Path


DEFAULT_BUCKETS = [120, 300, 600, 1800]


def label(bound_seconds: int) -> str:
    """将秒阈值转为可读标签。"""
    if bound_seconds < 60:
        return f'{bound_seconds}s'
    if bound_seconds < 3600:
        return f'{bound_seconds // 60}min'
    return f'{bound_seconds // 3600}h'


def bucket_name(idx: int, limits: list[int]) -> str:
    """根据索引和分界点生成桶名。"""
    if idx == 0:
        return f'<={label(limits[0])}'
    if idx < len(limits):
        return f'{label(limits[idx - 1] + 1)}-{label(limits[idx])}'
    return f'{label(limits[-1] + 1)}+'


def main():
    parser = argparse.ArgumentParser(description='分层随机抽样')
    parser.add_argument('input', help='输入 CSV 文件')
    parser.add_argument('-o', '--output', required=True, help='输出 CSV 文件')
    parser.add_argument('-c', '--col', default='duration_seconds', help='分层列（默认 duration_seconds）')
    parser.add_argument('-r', '--rate', type=float, default=0.90, help='抽样比例（默认 0.90）')
    parser.add_argument('--confidence', type=float, default=0.90, help='置信水平（默认 0.90）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认 42）')
    parser.add_argument(
        '--buckets', type=str,
        default=','.join(str(b) for b in DEFAULT_BUCKETS),
        help=f'分桶上限（秒），逗号分隔，默认 {",".join(str(b) for b in DEFAULT_BUCKETS)}',
    )
    args = parser.parse_args()

    limits = [int(x.strip()) for x in args.buckets.split(',') if x.strip()]
    if not limits:
        print('[ERROR] 至少需要一个分桶上限', file=sys.stderr)
        sys.exit(1)

    # z-score lookup
    z_table = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_table.get(args.confidence)
    if z is None:
        z = 1.645
        print(f'[WARN] 不支持的置信度 {args.confidence}，回退到 0.90 (z={z})', file=sys.stderr)

    random.seed(args.seed)

    # --- read & bucket ---
    rows_by_bucket = {bucket_name(i, limits): [] for i in range(len(limits) + 1)}
    total_dur = 0.0
    with open(args.input, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if args.col not in fieldnames:
            print(f'[ERROR] 列 "{args.col}" 不存在，可用列: {", ".join(fieldnames)}', file=sys.stderr)
            sys.exit(1)
        for row in reader:
            val = float(row.get(args.col, 0) or 0)
            total_dur += val
            # bucket assignment
            b_idx = len(limits)
            for i, lim in enumerate(limits):
                if val <= lim:
                    b_idx = i
                    break
            rows_by_bucket[bucket_name(b_idx, limits)].append(row)

    # --- sample ---
    sampled = []
    bnames = [bucket_name(i, limits) for i in range(len(limits) + 1)]
    total_pop = 0
    total_sample = 0

    print(f'{"Stratum":<16} {"Pop":>6} {"Sample":>6} {"Rate":>6} {"±MoE":>7}  CI ({args.confidence:.0%})')
    print('-' * 72)

    for bn in bnames:
        rows = rows_by_bucket[bn]
        n_pop = len(rows)
        n_sample = max(1, int(n_pop * args.rate))
        sample_rows = random.sample(rows, n_sample)
        sampled.extend(sample_rows)

        p = n_sample / n_pop if n_pop > 0 else 0
        moe = z * math.sqrt(p * (1 - p) / n_pop) if n_pop > 1 else 0
        lo = max(0, p - moe)
        hi = min(1, p + moe)

        print(f'{bn:<16} {n_pop:>6} {n_sample:>6} {p:>5.1%}  ±{moe:>5.1%}  [{lo:.1%}, {hi:.1%}]')

        total_pop += n_pop
        total_sample += n_sample

    print('-' * 72)
    p_overall = total_sample / total_pop if total_pop else 0
    print(f'{"合计":<16} {total_pop:>6} {total_sample:>6} {p_overall:>5.1%}')

    # --- duration stats ---
    sample_dur = 0.0
    all_dur = {}
    with open(args.input, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dur = float(row.get(args.col, 0) or 0)
            all_dur[row.get('video_id', '')] = dur

    for row in sampled:
        vid = row.get('video_id', '')
        sample_dur += all_dur.get(vid, 0)

    def fmt_sec(sec):
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f'{h}h {m}m {s}s ({sec:.0f}s)'

    print(f'\n总时长        : {fmt_sec(total_dur)}')
    print(f'抽样后总时长   : {fmt_sec(sample_dur)}')
    print(f'平均时长(全量) : {total_dur / total_pop:.0f}s = {total_dur / total_pop / 60:.1f}min')
    print(f'平均时长(抽样) : {sample_dur / total_sample:.0f}s = {sample_dur / total_sample / 60:.1f}min')

    # --- write ---
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)

    print(f'\n输出: {args.output}')


if __name__ == '__main__':
    main()


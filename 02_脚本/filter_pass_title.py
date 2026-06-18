#!/usr/bin/env python3
"""
filter_pass_title.py — 对 vision QC pass 集做标题规则过滤

用法:
  # 测试模式（默认跑 100 条看效果）
  python3 filter_pass_title.py ../data/runs/tow-person/002_audit/all_four_pass.csv --test

  # 全量跑
  python3 filter_pass_title.py ../data/runs/tow-person/002_audit/all_four_pass.csv -o output.csv
"""

import sys, os, re, csv, argparse
from pathlib import Path
import duckdb

# ── 标题黑名单规则 ──
TITLE_RULES = {
    # 体育解说 (trade talk, fantasy, breaking sports news)
    "sports_talk": [
        r'\btrade\s+talk\b', r'\bAFL\s+Fantasy\b', r'\bfantasy\s+(football|draft)\b',
        r'\bPaul\s+George\b', r'\b(trade|draft)\s+(talks?|rumors?)\b',
        r'\bESPN\b.*(trade|draft|mock)', r'\bNBA\b.*(trade|deal)\b',
        r'\bNFL\b.*(draft|pick|mock)\b',
    ],
    # 音频录制/短剧/WhatsApp 通话
    "audio_drama": [
        r'\bcall\s+(record|recording)\b', r'\baudio\b',
        r'\bromantic\b.*(call|audio|recording)', r'\bwhatsapp\b',
        r'\bphone\s+call\b', r'\bcouple\s+(call|audio|recording)',
        r'\blove\s+(story|audio|recording|call)\b',
        r'\breal\s+(estate|investor)\b',  # 商务推销
    ],
    # 单人/非对话内容
    "solo_non_dialogue": [
        r'\bprank\b', r'\bshorts?\b', r'#shorts',
        r'\bvlog\b', r'\bhaul\b', r'\bunboxing\b',
        r'\bmukbang\b', r'\basmr\b',
        r'\bgameplay\b', r'\bwalkthrough\b', r'\blet.?s\s+play\b',
    ],
    # 法庭/仲裁/调解 (非对话，是程序)
    "court_legal": [
        r'\bjudge\s+judy\b', r'\bcourt\b.*(case|hearing|trial)\b',
        r'\bdivorce\s+court\b', r'\barbitration\b',
        r'\bdeposition\b',
    ],
    # 教学/课程 (非自然对话)
    "tutorial": [
        r'\bIELTS\s+speaking\s+test\b', r'\bIELTS\b.*(sample|mock|practice)\b',
        r'\bTOEFL\b.*speaking', r'\btutorial\b', r'\bhow[\s-]to\b',
    ],
    # 宗教/布道 (非双人对话)
    "religion_sermon": [
        r'\bsermon\b', r'\bworship\b', r'\bprayer\b', r'\bbible\s+study\b',
        r'\bpreaching\b', r'\bgospel\b', r'\bgod\b.*(said|says|speaks)\b',
    ],
}


def compile_rules():
    """编译所有规则为 {类别: [re.Pattern]}"""
    compiled = {}
    for cat, patterns in TITLE_RULES.items():
        compiled[cat] = [re.compile(p, re.IGNORECASE) for p in patterns]
    return compiled


def title_match(title, compiled_rules):
    """检查标题是否命中规则，返回 (是否命中, 命中类别)"""
    t = title or ""
    for cat, patterns in compiled_rules.items():
        for pat in patterns:
            if pat.search(t):
                return True, cat
    return False, None


def run_filter(input_csv, output_csv, test_mode=False):
    compiled = compile_rules()

    con = duckdb.connect()
    n_total = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{input_csv}', header=true, all_varchar=true)").fetchone()[0]
    print(f"输入: {n_total:,} 条")

    if test_mode:
        print("测试模式: 取前 100 条")
        df = con.execute(f"SELECT * FROM read_csv_auto('{input_csv}', header=true, all_varchar=true) LIMIT 100").fetchdf()
    else:
        print("全量模式...")
        df = con.execute(f"SELECT * FROM read_csv_auto('{input_csv}', header=true, all_varchar=true)").fetchdf()

    # 过滤
    hits = []
    for i, row in df.iterrows():
        hit, cat = title_match(row.get("title", ""), compiled)
        if hit:
            hits.append((i, cat, row.get("title", "")[:80]))

    n_drop = len(hits)
    n_keep = len(df) - n_drop
    print(f"\n保留: {n_keep:,}  |  移除: {n_drop:,}  ({(n_drop/max(len(df),1)*100):.1f}%)")

    # 命中分布
    print(f"\n命中规则分布:")
    cat_counts = {}
    for _, cat, _ in hits:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  [{cat}]: {cnt}")

    # 命中样本
    print(f"\n命中样本 (前15条):")
    for i, (idx, cat, title) in enumerate(hits[:15], 1):
        print(f"  [{i}] [{cat}] {title}")

    if not test_mode and output_csv:
        keep_df = df.drop([h[0] for h in hits])
        import csv as csv_mod
        with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv_mod.DictWriter(f, fieldnames=keep_df.columns.tolist())
            writer.writeheader()
            writer.writerows(keep_df.to_dict('records'))
        print(f"\n输出: {n_keep:,} 条 → {output_csv}")

    con.close()


def main():
    parser = argparse.ArgumentParser(description="pass 集标题规则过滤")
    parser.add_argument("input", help="输入 CSV (all_four_pass.csv)")
    parser.add_argument("-o", "--output", default=None, help="输出 CSV")
    parser.add_argument("--test", action="store_true", help="测试模式: 只跑前 100 条")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    run_filter(args.input, args.output, args.test)


if __name__ == "__main__":
    main()

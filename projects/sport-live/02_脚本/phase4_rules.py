#!/usr/bin/env python3
"""
phase4_rules.py -- SOP Phase 4: 规则生成 + 接入主 TOML

从标注后的 audit_sample 提取新规则，去重后追加到共享规则文件。

共享规则文件:
  02_脚本/rules/blacklist.toml   — [[pass2]] / [[r2]]
  02_脚本/rules/whitelist.toml   — [[positive]] / [[negative]]
  02_脚本/rules/entities.toml    — sports / synonyms

每条新规则记录来源 audit_sample。

用法:
  # 写入 sport 专属目录 (不修改主 TOML)
  python3 phase4_rules.py data/runs/pingpong/002_audit/audit_sample_v1.parquet \
    -o data/runs/pingpong/004_rules/

  # 同时追加到主 TOML
  python3 phase4_rules.py data/runs/pingpong/002_audit/audit_sample_v1.parquet \
    -o data/runs/pingpong/004_rules/ --merge
"""

import sys, os, time, argparse, textwrap, tomllib, re
from collections import Counter
from pathlib import Path
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.sop import load_sop, print_banner

MIN_FP_SAMPLES = 3
MIN_FP_RATIO   = 0.6


RULES_DIR = Path(__file__).resolve().parent / "rules"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_shared_tomls():
    """加载共享 TOML → (blacklist, whitelist, entities)。"""
    bl_path = RULES_DIR / "blacklist.toml"
    wl_path = RULES_DIR / "whitelist.toml"
    ent_path = RULES_DIR / "entities.toml"

    bl = tomllib.loads(bl_path.read_text()) if bl_path.exists() else {}
    wl = tomllib.loads(wl_path.read_text()) if wl_path.exists() else {}
    ent = tomllib.loads(ent_path.read_text()) if ent_path.exists() else {}
    return bl, wl, ent


def existing_patterns(bl, wl):
    """收集已有的所有正则 pattern，避免重复。"""
    pats = set()
    for section in ["pass2", "r2"]:
        for item in bl.get(section, []):
            pats.add(item.get("pattern", ""))
    for section in ["positive", "negative", "weak_titles", "weak_channels"]:
        for item in wl.get(section, []):
            pats.add(item.get("pattern", ""))
    return pats


def extract_keyword_tag_rules(con, table: str, label_col: str, existing: set) -> list[dict]:
    """从 FP keyword 标签提取黑名单正则。"""
    rows = con.execute(f"""
        SELECT keyword FROM {table}
        WHERE UPPER({label_col}) = 'F' AND keyword != ''
    """).fetchall()

    # Count tags in FP
    fp_tags = Counter()
    for (kw,) in rows:
        if ' -' in kw:
            for part in kw.split(' -')[1:]:
                tag = part.strip().split()[0] if part.strip() else ''
                if tag and len(tag) >= 2:
                    fp_tags[tag] += 1

    # Count same tags in TP
    tp_rows = con.execute(f"""
        SELECT keyword FROM {table}
        WHERE UPPER({label_col}) = 'T' AND keyword != ''
    """).fetchall()
    tp_tags = Counter()
    for (kw,) in tp_rows:
        if ' -' in kw:
            for part in kw.split(' -')[1:]:
                tag = part.strip().split()[0] if part.strip() else ''
                if tag:
                    tp_tags[tag] += 1

    n_fp = max(len(rows), 1)
    new_rules = []
    for tag, fp_cnt in fp_tags.most_common(60):
        if fp_cnt < MIN_FP_SAMPLES:
            continue
        fp_ratio = fp_cnt / max(fp_cnt + tp_tags.get(tag, 0), 1)
        if fp_ratio < MIN_FP_RATIO:
            continue
        pat = f"\\b{re.escape(tag)}\\b"
        if pat not in existing and pat not in {r["pattern"] for r in new_rules}:
            new_rules.append({
                "category": "audit_tag",
                "pattern": pat,
                "_source": f"FP={fp_cnt}, ratio={fp_ratio:.0%}",
            })
    return new_rules


def extract_channel_rules(con, table: str, label_col: str, existing: set) -> tuple[list[dict], list[str]]:
    """从标注数据提取 channel 正则规则 + 白名单 channel 名。"""
    fp = con.execute(f"""
        SELECT channel, COUNT(*) AS cnt
        FROM {table}
        WHERE UPPER({label_col}) = 'F' AND channel != ''
        GROUP BY channel HAVING COUNT(*) >= {MIN_FP_SAMPLES}
        ORDER BY cnt DESC
    """).fetchall()

    tp = con.execute(f"""
        SELECT channel, COUNT(*) AS total,
               SUM(CASE WHEN UPPER({label_col}) = 'T' THEN 1 ELSE 0 END) AS t_cnt
        FROM {table}
        WHERE channel != ''
        GROUP BY channel
        HAVING COUNT(*) >= 3
           AND CAST(SUM(CASE WHEN UPPER({label_col}) = 'T' THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) >= 0.8
        ORDER BY t_cnt DESC
    """).fetchall()

    new_bl = []
    for ch, cnt in fp:
        pat = f"\\b{re.escape(ch)}\\b"
        if pat not in existing and pat not in {r["pattern"] for r in new_bl}:
            new_bl.append({
                "category": "audit_channel",
                "pattern": pat,
                "_source": f"FP={cnt}",
            })

    new_wl = []
    for ch, total, t_cnt in tp:
        if ch.lower() not in existing:
            new_wl.append(ch)

    return new_bl, new_wl


STOP_WORDS = {
    "the", "and", "for", "with", "this", "that", "from", "have", "are", "was",
    "not", "but", "you", "all", "can", "had", "her", "his", "its", "our",
    "out", "has", "been", "were", "will", "what", "when", "where", "which",
    "who", "how", "new", "get", "one", "two", "now", "top", "see", "more",
    "just", "like", "also", "than", "then", "about", "some", "very", "into",
    "only", "other", "over", "after", "part", "full", "best", "way", "day",
}

def extract_title_rules(con, table: str, label_col: str, existing: set) -> list[dict]:
    """从 FP 标题中提取高频词生成黑名单正则。"""
    rows = con.execute(f"""
        SELECT title FROM {table}
        WHERE UPPER({label_col}) = 'F' AND title != ''
    """).fetchall()

    words = Counter()
    for (title,) in rows:
        for w in title.lower().split():
            w = w.strip('"\'.,!?()[]{}:;#')
            if len(w) >= 4 and not w.isdigit() and w not in STOP_WORDS:
                words[w] += 1

    n_fp = max(len(rows), 1)
    new_rules = []
    for word, cnt in words.most_common(60):
        if cnt < 5 or cnt / n_fp < 0.02:  # stricter: at least 5 samples, 2%
            continue
        tp_cnt = con.execute(f"""
            SELECT COUNT(*) FROM {table}
            WHERE UPPER({label_col}) = 'T'
              AND LOWER(title) LIKE '%{word}%'
        """).fetchone()[0]
        if tp_cnt > cnt * 0.2:  # skip if >20% TP overlap
            continue
        pat = f"\\b{re.escape(word)}\\b"
        if pat not in existing and pat not in {r["pattern"] for r in new_rules}:
            new_rules.append({
                "category": "audit_title",
                "pattern": pat,
                "_source": f"FP={cnt}/{n_fp}",
            })
    return new_rules[:15]


def _toml_escape(pattern: str) -> str:
    """将正则 pattern 转义为 TOML 兼容格式（反斜杠翻倍）。"""
    return pattern.replace("\\", "\\\\")


def append_to_toml(path: Path, section: str, new_rules: list[dict]):
    """将新规则追加到 TOML 文件的指定 [[section]] 末尾。"""
    text = path.read_text()
    for rule in new_rules:
        src = rule.pop("_source", "")
        text += f"""
[[{section}]]
category = "{rule['category']}"
pattern = "{_toml_escape(rule['pattern'])}"  # {src}
"""
    path.write_text(text)
    log(f"  追加 {len(new_rules)} 条到 {path.name} [[{section}]]")


def main():
    print_banner(4)

    parser = argparse.ArgumentParser(
        description="SOP Phase 4: 规则生成 + 接入主 TOML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="标注后的 audit sample (parquet/csv)")
    parser.add_argument("-o", "--output-dir", default="data/runs/004_rules")
    parser.add_argument("--label-col", default="audit_label")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    out_dir = args.output_dir.rstrip("/")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    con = duckdb.connect()

    ext = Path(args.input).suffix.lower()
    reader = "read_parquet" if ext == ".parquet" else "read_csv_auto"
    con.execute(f"CREATE TEMP TABLE data AS SELECT * FROM {reader}('{args.input}')")

    n_total = con.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    sports = con.execute(f"SELECT COUNT(*) FROM data WHERE UPPER({args.label_col}) = 'T'").fetchone()[0]
    non_sports = con.execute(f"SELECT COUNT(*) FROM data WHERE UPPER({args.label_col}) = 'F'").fetchone()[0]
    log(f"标注数: {n_total} (T={sports}, F={non_sports})")

    # Load shared TOMLs
    bl, wl, ent = load_shared_tomls()
    existing = existing_patterns(bl, wl)
    existing_ch = {c.lower() for c in ent.get("sports", [])}

    # Extract
    new_tag_rules = extract_keyword_tag_rules(con, "data", args.label_col, existing)
    log(f"Keyword 标签规则: {len(new_tag_rules)} 条")

    new_ch_bl, new_ch_wl = extract_channel_rules(con, "data", args.label_col, existing)
    log(f"Channel 黑名单: {len(new_ch_bl)} 条, 白名单: {len(new_ch_wl)} 条")

    new_title_rules = extract_title_rules(con, "data", args.label_col, existing)
    log(f"Title 模式规则: {len(new_title_rules)} 条")

    con.close()

    # Write to sport-specific directory
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    sport_dir = os.path.join(out_dir, f"rules_import_{time.strftime('%Y%m%d_%H%M%S')}.toml")
    with open(sport_dir, "w") as f:
        f.write(f"# 规则导入 — 生成于 {ts}\n")
        f.write(f"# 来源: {args.input}\n")
        f.write(f"# 样本: T={sports}, F={non_sports}\n\n")
        for r in new_tag_rules:
            f.write(f"# {r['_source']}\n")
            f.write(f'[[pass2]]\ncategory = "{r["category"]}"\npattern = "{r["pattern"]}"\n\n')
        for r in new_ch_bl:
            f.write(f"# {r['_source']}\n")
            f.write(f'[[pass2]]\ncategory = "{r["category"]}"\npattern = "{r["pattern"]}"\n\n')
        for r in new_title_rules:
            f.write(f"# {r['_source']}\n")
            f.write(f'[[pass2]]\ncategory = "{r["category"]}"\npattern = "{r["pattern"]}"\n\n')
    log(f"Sport 规则存档: {sport_dir}")

    elapsed = time.perf_counter() - t0
    print()
    print("=" * 62)
    print(f"  Phase 4 — 规则生成 完成")
    print("=" * 62)
    print(f"  Keyword 标签:    {len(new_tag_rules)} 条")
    print(f"  Channel 黑名单:  {len(new_ch_bl)} 条")
    print(f"  Channel 白名单:  {len(new_ch_wl)} 条")
    print(f"  Title 模式:      {len(new_title_rules)} 条")
    print(f"  耗时:            {elapsed:.1f}s")
    print(f"  候选规则档案:    {sport_dir}")
    print("=" * 62)
    print()
    print(f"  → Phase 5 验证规则效果: python3 clean_sports_v3.py baseline.parquet -o data/runs/005_clean/")
    print(f"  → Phase 6 效果验证后，人工审核通过再合并到 02_脚本/rules/")


if __name__ == "__main__":
    main()

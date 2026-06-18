#!/usr/bin/env python3
"""
phase3_analyze.py -- SOP Phase 3: 污染分析

从标注后的 audit_sample 统计污染来源。
输出: {output_dir}/{run}/pollution_analysis_v1.md

用法:
  python3 phase3_analyze.py audit_sample_v1.parquet -o data/runs/003_analysis/
  python3 phase3_analyze.py audit_sample_v1.parquet -o data/runs/003_analysis/ --run run02
"""

import sys, os, time, argparse, textwrap
from collections import Counter
from pathlib import Path
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.sop import load_sop, print_banner, write_run_log


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    sop_text = load_sop()
    if sop_text: print(sop_text[:600] + "...\n")
    print_banner(3)

    parser = argparse.ArgumentParser(description="SOP Phase 3: 污染分析")
    parser.add_argument("input", help="标注后的 audit sample (parquet/csv)")
    parser.add_argument("-o", "--output-dir", default="data/runs/003_analysis")
    parser.add_argument("--run", default="run01", help="迭代轮次")
    parser.add_argument("--label-col", default="qc_text_result")
    parser.add_argument("--category-col", default="audit_category")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] {args.input}"); sys.exit(1)

    out_dir = os.path.join(args.output_dir.rstrip("/"), args.run)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    con = duckdb.connect()

    ext = Path(args.input).suffix.lower()
    reader = "read_parquet" if ext == ".parquet" else "read_csv_auto"

    log(f"读取: {args.input}")
    n_total = con.execute(f"SELECT COUNT(*) FROM {reader}('{args.input}')").fetchone()[0]
    log(f"  标注数: {n_total}")

    label_col = args.label_col
    cat_col = args.category_col
    cols = [c[0].lower() for c in con.execute(f"DESCRIBE SELECT * FROM {reader}('{args.input}') LIMIT 0").fetchall()]
    has_label = label_col.lower() in cols
    has_cat = cat_col.lower() in cols

    if not has_label:
        log(f"  [WARN] 无 {label_col} 列")
        con.execute(f"CREATE TEMP TABLE data AS SELECT *, '' AS {label_col} FROM {reader}('{args.input}')")
    else:
        con.execute(f"CREATE TEMP TABLE data AS SELECT * FROM {reader}('{args.input}')")

    lang = con.execute(f"SELECT COUNT(*) FROM data WHERE UPPER({label_col}) = 'T'").fetchone()[0]
    non_lang = con.execute(f"SELECT COUNT(*) FROM data WHERE UPPER({label_col}) = 'F'").fetchone()[0]

    log(f"  语言教学: {lang} ({lang/max(n_total,1)*100:.1f}%)")
    log(f"  非语言教学: {non_lang} ({non_lang/max(n_total,1)*100:.1f}%)")

    kw_rows = con.execute(f"""
        SELECT keyword, COUNT(*) AS cnt FROM data
        WHERE UPPER({label_col}) = 'F' AND keyword != ''
        GROUP BY keyword ORDER BY cnt DESC LIMIT 20
    """).fetchall() if non_lang > 0 else []

    ch_rows = con.execute(f"""
        SELECT channel, COUNT(*) AS cnt FROM data
        WHERE UPPER({label_col}) = 'F' AND channel != ''
        GROUP BY channel ORDER BY cnt DESC LIMIT 20
    """).fetchall() if non_lang > 0 else []

    cat_rows = []
    if has_cat and non_lang > 0:
        cat_rows = con.execute(f"""
            SELECT {cat_col}, COUNT(*) AS cnt FROM data
            WHERE UPPER({label_col}) = 'F' AND {cat_col} != ''
            GROUP BY {cat_col} ORDER BY cnt DESC
        """).fetchall()

    con.close()

    # 报告
    lines = [
        f"# Pollution Analysis v1",
        f"",
        f"**Input:** `{args.input}`",
        f"**Run:** {args.run}",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total:** {n_total} · T={lang} F={non_lang}",
        f"",
        f"## 污染来源",
    ]
    if cat_rows:
        lines.append(f"**Top:** `{cat_rows[0][0]}` ({cat_rows[0][1]} 条)")
    lines.append("")
    lines.append("### 高频 keyword")
    lines.append("| Keyword | Count | Pct |")
    lines.append("|---------|-------|-----|")
    for kw, cnt in kw_rows:
        lines.append(f"| `{kw[:50]}` | {cnt} | {cnt/max(non_lang,1)*100:.1f}% |")
    lines.append("")
    lines.append("### 高频 channel")
    lines.append("| Channel | Count | Pct |")
    lines.append("|---------|-------|-----|")
    for ch, cnt in ch_rows:
        safe = (ch or "").replace("|", "/")
        lines.append(f"| `{safe[:50]}` | {cnt} | {cnt/max(non_lang,1)*100:.1f}% |")
    lines.append("")
    if cat_rows:
        lines.append("### 类别分布")
        lines.append("| Category | Count | Pct |")
        lines.append("|----------|-------|-----|")
        for cat, cnt in cat_rows:
            lines.append(f"| {cat} | {cnt} | {cnt/max(non_lang,1)*100:.1f}% |")
        lines.append("")

    lines.append("## 下一步")
    lines.append(f"根据污染分析结果更新 categories/language_teaching/rules/blacklist.toml 或 categories/beauty/rules/blacklist.toml")
    lines.append(f"然后重新运行 phase5_clean.py 迭代清洗")

    out_md = os.path.join(out_dir, "pollution_analysis_v1.md")
    with open(out_md, "w") as f: f.write("\n".join(lines))

    elapsed = time.perf_counter() - t0

    # Write progress
    from core.progress import mark_done
    mark_done(out_dir, 3, lang=lang, non_lang=non_lang,
              pollution_categories=len(cat_rows), pollution_keywords=len(kw_rows),
              elapsed_sec=round(elapsed, 1))

    write_run_log(3, args.input, out_dir,
                  stats={"samples": n_total, "lang": lang, "non_lang": non_lang,
                         "categories": len(cat_rows), "elapsed_sec": round(elapsed, 1)})

    print(f"\n{'='*62}\n  Phase 3 完成\n{'='*62}")
    print(f"  T={lang} F={non_lang} · {len(cat_rows)} 类污染")
    print(f"  产物: {out_dir}/\n{'='*62}")


if __name__ == "__main__":
    main()

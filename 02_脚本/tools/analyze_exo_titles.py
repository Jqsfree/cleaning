#!/usr/bin/env python3
"""
tools/analyze_exo_titles.py — exo 标题词频/语义桶分析 + 规则初筛

单工具流水线（方案 1）:
  1) 标题 unigram/bigram 词频
  2) 预定义语义桶命中（农采正信号 / 硬噪声）
  3) 调用 categories/exo cleaner 做 certain-noise 第一轮 keep/drop
  4) 可选：从 keep 抽样并复用 qc/text.py（T/F/U）

用法:
  02_脚本/tools/analyze_exo_titles.py \\
    data/runs/exo/machine_0807/01_quality/*quality*.csv \\
    -o data/runs/exo/machine_0807/ \\
    --apply-rules --sample-n 385
  # 有 DASHSCOPE_API_KEY 时再加: --run-text-qc -w 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.batch_layout import infer_batch_root, require_output_dir, warn_outside_batch
from core.io import duckdb_reader, strip_stem
from core.log import log as core_log
from core.run_manifest import maybe_update_stage
from categories.exo.cleaner import clean as exo_clean

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[\u4e00-\u9fff]+", re.I)

# 语义桶：只用于报告归因，不直接 drop（drop 走 blacklist.toml）
SEMANTIC_BUCKETS: dict[str, list[str]] = {
    "agri_positive": [
        "harvest", "harvesting", "picking", "farm", "farming", "agriculture",
        "vegetable", "vegetables", "fruit", "tomato", "potato", "onion",
        "cabbage", "carrot", "corn", "apple", "grape", "strawberry",
        "greenhouse", "orchard", "spinach", "cucumber", "eggplant",
        "workers", "manual", "by", "hand", "采摘", "收割", "蔬菜",
    ],
    "noise_game": ["gameplay", "minecraft", "roblox", "fortnite", "gaming", "simulator"],
    "noise_music": ["lyrics", "album", "official music", "official mv", "soundtrack", "music video"],
    "noise_film_tv": ["official trailer", "teaser trailer", "full movie", "movie clip", "episode ", "season "],
    "noise_cooking": ["recipe", "how to cook", "kitchen hack", "tasty recipe"],
    "noise_talk": ["tedx", "podcast", "breaking news", "talk show", "interview"],
    "noise_teach": ["online course", "webinar", "masterclass", "tutorial video", "beginner's guide"],
    "noise_beauty": ["makeup tutorial", "skincare routine", "beauty tips"],
    "noise_music_teach": ["piano lesson", "guitar lesson", "music lesson"],
    "noise_fight": ["street fight", "fight cam", "brawl"],
    "noise_sports": ["full match", "nba", "ufc", "match highlights"],
    "noise_anim_kids": ["anime", "cartoon", "nursery rhyme", "cocomelon", "peppa pig"],
    "noise_asmr": ["asmr"],
}

STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "from", "by", "at", "is", "are", "this", "that", "it", "as", "be",
    "how", "what", "your", "you", "my", "our", "their", "i", "we", "they",
    "video", "youtube", "shorts", "full", "new", "best", "day", "part",
})


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _tokenize(title: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(title or "") if t]


def _bigrams(tokens: list[str]) -> list[str]:
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def analyze_titles(df: pd.DataFrame, top_n: int = 80) -> dict:
    titles = df["title"].fillna("").astype(str)
    uni: Counter[str] = Counter()
    bi: Counter[str] = Counter()
    bucket_hits: dict[str, int] = {k: 0 for k in SEMANTIC_BUCKETS}
    bucket_examples: dict[str, list[str]] = {k: [] for k in SEMANTIC_BUCKETS}

    for title in titles:
        toks = _tokenize(title)
        uni.update(t for t in toks if t not in STOPWORDS and len(t) > 1)
        bi.update(_bigrams(toks))
        low = title.lower()
        for bucket, words in SEMANTIC_BUCKETS.items():
            if any(w in low for w in words):
                bucket_hits[bucket] += 1
                if len(bucket_examples[bucket]) < 8:
                    bucket_examples[bucket].append(title[:120])

    return {
        "n_rows": int(len(df)),
        "n_empty_title": int((titles.str.strip() == "").sum()),
        "top_unigrams": uni.most_common(top_n),
        "top_bigrams": bi.most_common(top_n),
        "bucket_hits": bucket_hits,
        "bucket_examples": bucket_examples,
    }


def write_analysis(out_dir: Path, analysis: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(analysis["top_unigrams"], columns=["token", "count"]).to_csv(
        out_dir / "title_unigram_freq.csv", index=False
    )
    pd.DataFrame(analysis["top_bigrams"], columns=["bigram", "count"]).to_csv(
        out_dir / "title_bigram_freq.csv", index=False
    )
    pd.DataFrame(
        [{"bucket": k, "hits": v} for k, v in analysis["bucket_hits"].items()]
    ).to_csv(out_dir / "title_semantic_buckets.csv", index=False)

    lines = [
        "# exo 标题分析",
        "",
        f"- rows: {analysis['n_rows']:,}",
        f"- empty_title: {analysis['n_empty_title']:,}",
        "",
        "## 语义桶命中（报告用，非直接 drop）",
        "",
    ]
    n = max(analysis["n_rows"], 1)
    for bucket, hits in sorted(analysis["bucket_hits"].items(), key=lambda x: -x[1]):
        lines.append(f"- **{bucket}**: {hits:,} ({hits / n:.1%})")
        for ex in analysis["bucket_examples"].get(bucket, [])[:5]:
            lines.append(f"  - {ex}")
    lines += ["", "## Top unigrams", ""]
    for tok, c in analysis["top_unigrams"][:40]:
        lines.append(f"- {tok}: {c}")
    lines += ["", "## Top bigrams", ""]
    for tok, c in analysis["top_bigrams"][:40]:
        lines.append(f"- {tok}: {c}")

    report = out_dir / "title_analysis.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "title_analysis.json").write_text(
        json.dumps(
            {
                "n_rows": analysis["n_rows"],
                "n_empty_title": analysis["n_empty_title"],
                "bucket_hits": analysis["bucket_hits"],
                "top_unigrams": analysis["top_unigrams"][:100],
                "top_bigrams": analysis["top_bigrams"][:100],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def sample_keep(keep_path: str, sample_dir: Path, n: int, seed: int) -> Path | None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(":memory:")
    reader = duckdb_reader(keep_path)
    total = db.execute(f"SELECT COUNT(*) FROM {reader}").fetchone()[0]
    if total == 0:
        _log("keep 为空，跳过抽样")
        return None
    n = min(n, total)
    out = sample_dir / f"exo_title_keep_sample_{n}.csv"
    db.execute(
        f"""
        COPY (
          SELECT * FROM {reader}
          ORDER BY hash(COALESCE(video_id, '') || '{seed}')
          LIMIT {n}
        ) TO '{out}' (HEADER, DELIMITER ',')
        """
    )
    _log(f"抽样 {n:,}/{total:,} → {out}")
    return out


def run_text_qc(sample_csv: Path, qc_dir: Path, workers: int) -> None:
    qc_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent.parent / "qc" / "text.py"
    cmd = [
        sys.executable,
        str(script),
        str(sample_csv),
        "--category",
        "exo",
        "-w",
        str(workers),
        "-o",
        str(qc_dir),
    ]
    _log("启动 text QC: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="exo 标题分析 + 规则初筛")
    ap.add_argument("input", help="quality keep csv/parquet")
    ap.add_argument(
        "-o",
        "--output",
        required=True,
        help="批次根目录 data/runs/exo/{source}_{batch}/",
    )
    ap.add_argument("--run", default="title01", help="clean run 名")
    ap.add_argument("--apply-rules", action="store_true", help="执行 cleaner 初筛")
    ap.add_argument("--sample-n", type=int, default=0, help="从 keep 抽样条数（0=不抽）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-text-qc", action="store_true", help="对抽样跑 qc/text.py")
    ap.add_argument("-w", "--workers", type=int, default=20)
    ap.add_argument("--top-n", type=int, default=80)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"[ERROR] 输入不存在: {args.input}")

    batch = Path(require_output_dir(args.output))
    warn_outside_batch(str(batch), log_fn=core_log)
    if infer_batch_root(batch) is None and not (
        batch.name.startswith("human_") or batch.name.startswith("machine_")
    ):
        _log(f"[WARN] -o 看起来不像批次根: {batch}")

    analysis_dir = batch / "03_qc" / "analysis" / args.run
    clean_dir = batch / "05_clean" / f"run_{args.run}"
    sample_dir = batch / "02_sample" / f"run_{args.run}"
    qc_dir = batch / "03_qc" / f"text_{args.run}"

    t0 = time.perf_counter()
    _log(f"读取: {args.input}")
    df = pd.read_csv(args.input, dtype=str, low_memory=False) if args.input.endswith(
        (".csv", ".tsv")
    ) else pd.read_parquet(args.input)
    if "title" not in df.columns:
        raise SystemExit("[ERROR] 缺少 title 列")

    analysis = analyze_titles(df, top_n=args.top_n)
    report = write_analysis(analysis_dir, analysis)
    _log(f"分析报告: {report}")
    for bucket, hits in sorted(analysis["bucket_hits"].items(), key=lambda x: -x[1])[:8]:
        _log(f"  bucket {bucket}: {hits:,}")

    summary: dict = {
        "input": args.input,
        "analysis_dir": str(analysis_dir),
        "bucket_hits": analysis["bucket_hits"],
        "n_rows": analysis["n_rows"],
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    if args.apply_rules:
        stem = strip_stem(Path(args.input).name) or "exo"
        clean_summary = exo_clean(
            input_path=args.input,
            stem=stem,
            output_dir=str(clean_dir),
            run=args.run,
            fmt="parquet",
        )
        summary["clean"] = clean_summary
        maybe_update_stage(
            str(clean_dir),
            "clean",
            paths={
                "keep": clean_summary.get("keep_path", ""),
                "drop": clean_summary.get("drop_path", ""),
            },
            stats={
                "n_total": clean_summary.get("n_total"),
                "n_keep": clean_summary.get("n_keep"),
                "n_drop": clean_summary.get("n_drop"),
                "run": args.run,
            },
            category="exo",
        )
        keep_path = clean_summary["keep_path"]
    else:
        keep_path = args.input
        _log("未加 --apply-rules，仅分析")

    sample_path = None
    if args.sample_n > 0:
        sample_path = sample_keep(keep_path, sample_dir, args.sample_n, args.seed)
        summary["sample_path"] = str(sample_path) if sample_path else None

    if args.run_text_qc:
        if not sample_path:
            raise SystemExit("[ERROR] --run-text-qc 需要 --sample-n > 0")
        if not os.environ.get("DASHSCOPE_API_KEY"):
            raise SystemExit("[ERROR] 缺少 DASHSCOPE_API_KEY")
        run_text_qc(sample_path, qc_dir, args.workers)
        summary["text_qc_dir"] = str(qc_dir)

    summary["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    summary_path = analysis_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"完成 {summary['elapsed_sec']}s → {summary_path}")


if __name__ == "__main__":
    main()

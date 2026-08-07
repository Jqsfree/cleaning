#!/usr/bin/env python3
"""
tools/write_rules_notes.py — 机采第一层：从 text QC 结果写 04_rules/NOTES.md

用法:
  02_脚本/tools/write_rules_notes.py -o $BATCH/
  02_脚本/tools/write_rules_notes.py -o $BATCH/ --qc-csv $BATCH/03_qc/xxx.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.run_manifest import load_manifest, update_stage  # noqa: E402


def _latest_text_qc(qc_dir: Path) -> Path | None:
    if not qc_dir.is_dir():
        return None
    cands: list[Path] = []
    for p in qc_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".csv", ".parquet"):
            continue
        name = p.name.lower()
        if "drop" in name and "qc" not in name:
            continue
        cands.append(p)
    if not cands:
        return None
    # 优先带 qc / text 字样
    ranked = sorted(
        cands,
        key=lambda p: (
            0 if ("qc" in p.name.lower() or "text" in p.name.lower()) else 1,
            -p.stat().st_mtime,
        ),
    )
    return ranked[0]


def _count_labels(path: Path) -> dict[str, int]:
    import duckdb

    con = duckdb.connect()
    try:
        if path.suffix.lower() == ".parquet":
            rel = f"read_parquet('{path}')"
        else:
            rel = f"read_csv_auto('{path}', header=true, ignore_errors=true)"
        cols = {c[0] for c in con.execute(f"SELECT * FROM {rel} LIMIT 0").description}
        if "qc_text_result" not in cols:
            n = con.execute(f"SELECT COUNT(*) FROM {rel}").fetchone()[0]
            return {"rows": int(n), "T": 0, "F": 0, "U": 0, "ERROR": 0, "missing_col": 1}
        rows = con.execute(
            f"""
            SELECT
              COUNT(*) AS rows,
              SUM(CASE WHEN qc_text_result = 'T' THEN 1 ELSE 0 END) AS t,
              SUM(CASE WHEN qc_text_result = 'F' THEN 1 ELSE 0 END) AS f,
              SUM(CASE WHEN qc_text_result = 'U' THEN 1 ELSE 0 END) AS u,
              SUM(CASE WHEN qc_text_result = 'ERROR' THEN 1 ELSE 0 END) AS e
            FROM {rel}
            """
        ).fetchone()
        return {
            "rows": int(rows[0] or 0),
            "T": int(rows[1] or 0),
            "F": int(rows[2] or 0),
            "U": int(rows[3] or 0),
            "ERROR": int(rows[4] or 0),
            "missing_col": 0,
        }
    finally:
        con.close()


def write_notes(
    batch_root: Path,
    *,
    qc_csv: Path | None = None,
    notes_extra: str = "",
) -> Path:
    root = Path(batch_root)
    qc_dir = root / "03_qc"
    rules_dir = root / "04_rules"
    rules_dir.mkdir(parents=True, exist_ok=True)

    qc_path = Path(qc_csv) if qc_csv else _latest_text_qc(qc_dir)
    sample_dir = root / "02_sample"
    samples = sorted(sample_dir.glob("*sample*"), key=lambda p: p.stat().st_mtime, reverse=True) if sample_dir.is_dir() else []
    sample_path = samples[0] if samples else None

    man = {}
    try:
        man = load_manifest(root) or {}
    except Exception:
        man = {}
    category = man.get("category") or root.parent.name
    source = man.get("source") or root.name.split("_", 1)[0]
    batch = man.get("batch") or root.name.split("_", 1)[-1]

    stats_lines = []
    if qc_path and qc_path.is_file():
        st = _count_labels(qc_path)
        if st.get("missing_col"):
            stats_lines.append(f"- QC 文件: `{qc_path}`（无 qc_text_result 列，rows={st['rows']}）")
        else:
            stats_lines.append(f"- QC 文件: `{qc_path}`")
            stats_lines.append(
                f"- 标签: T={st['T']} F={st['F']} U={st['U']} ERROR={st['ERROR']} "
                f"(n={st['rows']})"
            )
    else:
        stats_lines.append("- QC 文件: （未找到；请先跑 qc/text.py）")

    body = "\n".join([
        f"# 规则依据 NOTES — {category} / {source}_{batch}",
        "",
        f"- 日期: {date.today().isoformat()}",
        f"- 样本: `{sample_path}`" if sample_path else "- 样本: （无）",
        *stats_lines,
        "",
        "## 拟改规则",
        "",
        "- 仅 certain-noise 写入 `categories/%s/rules/blacklist.toml`" % category,
        "- 无法确认的模式勿写进黑名单（对应 QC 的 U）",
        "",
        notes_extra.strip(),
        "",
        "---",
        "本阶段（机采第一层）停在 `04_rules`；有 NOTES + 规则依据后再 `--rules-ready` clean。",
        "",
    ])
    out = rules_dir / "NOTES.md"
    out.write_text(body, encoding="utf-8")
    try:
        update_stage(
            root,
            "rules",
            paths={"notes": str(out)},
        )
    except FileNotFoundError:
        pass
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="写 04_rules/NOTES.md（text QC 统计）")
    p.add_argument("-o", "--batch-root", required=True)
    p.add_argument("--qc-csv", default=None, help="指定 text QC 结果表")
    p.add_argument("--extra", default="", help="追加到 NOTES 的备注")
    args = p.parse_args()
    path = write_notes(
        Path(args.batch_root),
        qc_csv=Path(args.qc_csv) if args.qc_csv else None,
        notes_extra=args.extra,
    )
    print(f"已写 {path}")


if __name__ == "__main__":
    main()

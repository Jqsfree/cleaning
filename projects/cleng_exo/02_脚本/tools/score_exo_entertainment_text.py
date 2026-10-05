#!/usr/bin/env python3
"""娱乐表演 Ettin reranker 文本打分 — (query, title) → ml_score。

替换 MiniLM 前端：只换「文本如何得到 score」；下游仍是
score → 阈值 τ → drop/remain → 人标验证。特征仅 title（不含 channel）。

默认模型 cross-encoder/ettin-reranker-68m-v1；阈值来自
models/exo_entertainment_ettin_reranker_calibration.json 的 t_like。
ml_score 为 relevance logit（非 [0,1]）；score < τ → ml_auto_drop。

用法:
  # 金标冒烟
  PYTHONPATH=02_脚本 .venv/bin/python3 02_脚本/tools/score_exo_entertainment_text.py \\
    data/runs/exo_entertainment/machine_0818/03_qc/labeled.csv \\
    -o work/exo_entertainment_ettin_0921/labeled_scored.csv --sample 250 --low 10

  # 大表推荐 --lite：只写 video_id+score，再 DuckDB join 拆 keep/drop
  PYTHONPATH=02_脚本 .venv/bin/python3 02_脚本/tools/score_exo_entertainment_text.py \\
    data/runs/exo_entertainment/machine_0818/05_clean/run10_v10/娱乐表演_merged_0818_clean_0921.csv \\
    -o data/runs/exo_entertainment/machine_0818/06_tools/text_gov_v01/娱乐表演_merged_0818_ettin_scores_0921.csv \\
    --lite --split-keep-drop --resume
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from core.ettin_reranker import (  # noqa: E402
    ENTERTAINMENT_QUERY,
    EttinReranker,
    build_doc,
    resolve_model_id,
)

DEFAULT_CALIB = _REPO_ROOT / "models/exo_entertainment_ettin_reranker_calibration.json"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_LENGTH = 256
DEFAULT_MODEL_SIZE = "68m"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_threshold(
    calib_path: Path,
    *,
    legacy_recall: bool = False,
) -> float:
    if not calib_path.is_file():
        raise FileNotFoundError(f"calibration missing: {calib_path}")
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    if legacy_recall:
        section = calib.get("recall") or {}
    else:
        section = calib.get("t_like") or calib.get("recall") or {}
    if section.get("drop_threshold") is None:
        raise ValueError(f"drop_threshold missing in {calib_path}")
    return float(section["drop_threshold"])


def load_query(calib_path: Path, override: str | None) -> str:
    if override:
        return override
    if calib_path.is_file():
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
        q = calib.get("query")
        if isinstance(q, str) and q.strip():
            return q.strip()
    return ENTERTAINMENT_QUERY


def score_dataframe(
    df: pd.DataFrame,
    model: EttinReranker,
    *,
    query: str,
    threshold: float,
    batch_size: int,
    max_length: int,
) -> pd.DataFrame:
    if "title" not in df.columns:
        raise ValueError("输入 CSV 需要 title 列")
    out = df.copy()
    docs = [build_doc(t) for t in out["title"].tolist()]
    pairs = [(query, d) for d in docs]
    scores = model.predict(pairs, batch_size=batch_size, max_length=max_length)
    out["ml_score"] = scores
    out["ml_drop_threshold"] = threshold
    out["ml_auto_drop"] = scores < threshold
    out["ml_query"] = query
    out["ml_model"] = model.model_id
    return out


def score_file(
    inp: Path,
    out: Path,
    model: EttinReranker,
    *,
    query: str,
    threshold: float,
    chunksize: int,
    sample: int,
    batch_size: int,
    max_length: int,
    resume: bool = False,
    split_keep_drop: bool = False,
    run_identity: dict | None = None,
):
    from core.scoring_checkpoint import score_csv

    if not math.isfinite(threshold):
        raise ValueError("Invalid scoring threshold")
    if not run_identity:
        raise ValueError("Model/code identity required")
    split_paths = None
    if split_keep_drop:
        if "_ettin_scored" in out.stem:
            split_paths = tuple(
                out.with_name(out.stem.replace("_ettin_scored", suffix, 1) + out.suffix)
                for suffix in ("_ettin_keep", "_ettin_drop")
            )
        elif "_ml_scored" in out.stem:
            split_paths = tuple(
                out.with_name(out.stem.replace("_ml_scored", suffix, 1) + out.suffix)
                for suffix in ("_ml_keep", "_ml_drop")
            )
        else:
            split_paths = tuple(
                out.with_name(out.stem + suffix + out.suffix)
                for suffix in ("_ettin_keep", "_ettin_drop")
            )
    return score_csv(
        inp,
        out,
        lambda frame: score_dataframe(
            frame,
            model,
            query=query,
            threshold=threshold,
            batch_size=batch_size,
            max_length=max_length,
        ),
        identity=dict(run_identity, threshold=threshold, query=query),
        chunksize=chunksize,
        sample=sample,
        resume=resume,
        split_paths=split_paths,
    )


def score_lite_file(
    inp: Path,
    scores_out: Path,
    model: EttinReranker,
    *,
    query: str,
    threshold: float,
    chunksize: int,
    sample: int,
    batch_size: int,
    max_length: int,
    resume: bool,
    run_identity: dict,
) -> dict:
    """Score only video_id + ml_score (fast path for multi-million-row pools)."""
    from core.data_profile import file_hash
    from core.runtime_files import atomic_json, file_lock

    inp = inp.resolve()
    scores_out = scores_out.resolve()
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = scores_out.with_suffix(scores_out.suffix + ".checkpoint.json")
    usecols = ["video_id", "title"]
    header = pd.read_csv(inp, nrows=0, encoding="utf-8-sig").columns.tolist()
    if "duration_seconds" in header:
        usecols.append("duration_seconds")

    signature = {
        "input_sha256": file_hash(inp),
        "mode": "lite_scores",
        "identity": dict(run_identity, threshold=threshold, query=query),
        "sample": sample,
        "usecols": usecols,
    }
    with file_lock(scores_out.with_suffix(scores_out.suffix + ".lock"), blocking=False):
        state = {
            "schema_version": 1,
            "signature": signature,
            "rows": 0,
            "state": "running",
        }
        if resume:
            if not checkpoint.is_file():
                raise ValueError("Lite output has no checkpoint; start fresh")
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            if state.get("schema_version") != 1 or state.get("signature") != signature:
                raise ValueError("Resume fingerprint mismatch")
            if not scores_out.is_file():
                raise ValueError("Checkpoint exists but scores file missing")
        else:
            if scores_out.exists() or checkpoint.exists():
                raise FileExistsError("Lite scores already exist; use --resume or new path")
            scores_out.write_text(
                "video_id,ml_score,ml_drop_threshold,ml_auto_drop\n",
                encoding="utf-8",
            )
            atomic_json(checkpoint, state)

        completed = int(state["rows"])
        skipped = 0
        t0 = time.time()
        for chunk in pd.read_csv(
            inp,
            encoding="utf-8-sig",
            usecols=usecols,
            chunksize=chunksize,
            nrows=sample or None,
            dtype=str,
        ):
            if skipped + len(chunk) <= completed:
                skipped += len(chunk)
                continue
            if skipped < completed:
                chunk = chunk.iloc[completed - skipped :]
                skipped = completed
            scored = score_dataframe(
                chunk,
                model,
                query=query,
                threshold=threshold,
                batch_size=batch_size,
                max_length=max_length,
            )
            slim = scored[["video_id", "ml_score", "ml_drop_threshold", "ml_auto_drop"]].copy()
            slim["ml_auto_drop"] = slim["ml_auto_drop"].map(lambda x: "true" if bool(x) else "false")
            slim.to_csv(
                scores_out,
                mode="a",
                header=False,
                index=False,
                encoding="utf-8",
            )
            state["rows"] = completed = completed + len(slim)
            state["state"] = "running"
            atomic_json(checkpoint, state)
            elapsed = max(time.time() - t0, 1e-6)
            _log(
                f"lite {completed:,} rows  "
                f"{completed / elapsed:.0f} rows/s  "
                f"τ={threshold:.4f}"
            )

        if file_hash(inp) != signature["input_sha256"]:
            raise ValueError("Input changed during lite scoring")
        if state["rows"] == 0:
            raise ValueError("Empty lite scoring input")
        state["state"] = "complete"
        atomic_json(checkpoint, state)
        return {
            "n_rows": state["rows"],
            "scores": str(scores_out),
            "checkpoint": str(checkpoint),
            "elapsed_sec": round(time.time() - t0, 1),
        }


def apply_lite_join(
    inp: Path,
    scores_path: Path,
    *,
    threshold: float,
    keep_path: Path,
    drop_path: Path,
    scored_full_path: Path | None = None,
) -> dict:
    """Join lite scores back onto the full pool and split keep/drop via DuckDB."""
    import duckdb
    from core.sql_builder import sql_escape

    inp = inp.resolve()
    scores_path = scores_path.resolve()
    keep_path = keep_path.resolve()
    drop_path = drop_path.resolve()
    keep_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=2")
    tmp = keep_path.parent / ".ettin_duckdb_tmp"
    tmp.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{sql_escape(str(tmp))}'")

    inp_s = sql_escape(str(inp))
    sc_s = sql_escape(str(scores_path))
    keep_s = sql_escape(str(keep_path))
    drop_s = sql_escape(str(drop_path))

    con.execute(
        f"""
        CREATE OR REPLACE TABLE scores AS
        SELECT
          video_id,
          TRY_CAST(ml_score AS DOUBLE) AS ml_score,
          TRY_CAST(ml_drop_threshold AS DOUBLE) AS ml_drop_threshold,
          lower(CAST(ml_auto_drop AS VARCHAR)) IN ('true','1') AS ml_auto_drop
        FROM read_csv_auto('{sc_s}', header=true, all_varchar=true, ignore_errors=true)
        """
    )
    # Re-apply threshold in case scores file used a different τ
    con.execute(
        f"""
        CREATE OR REPLACE TABLE scored AS
        SELECT
          k.*,
          s.ml_score,
          {float(threshold)} AS ml_drop_threshold,
          (s.ml_score < {float(threshold)}) AS ml_auto_drop
        FROM read_csv_auto('{inp_s}', header=true, all_varchar=true, ignore_errors=true) k
        INNER JOIN scores s USING (video_id)
        """
    )
    n = con.execute("SELECT COUNT(*) FROM scored").fetchone()[0]
    n_drop = con.execute("SELECT COUNT(*) FROM scored WHERE ml_auto_drop").fetchone()[0]
    n_keep = n - n_drop
    hours = con.execute(
        """
        SELECT
          COALESCE(SUM(CASE WHEN NOT ml_auto_drop THEN TRY_CAST(duration_seconds AS DOUBLE) END),0)/3600,
          COALESCE(SUM(CASE WHEN ml_auto_drop THEN TRY_CAST(duration_seconds AS DOUBLE) END),0)/3600
        FROM scored
        """
    ).fetchone()

    if scored_full_path is not None:
        full_s = sql_escape(str(scored_full_path.resolve()))
        con.execute(
            f"COPY scored TO '{full_s}' (HEADER, DELIMITER ',')"
        )

    con.execute(
        f"COPY (SELECT * FROM scored WHERE NOT ml_auto_drop) TO '{keep_s}' (HEADER, DELIMITER ',')"
    )
    con.execute(
        f"COPY (SELECT * FROM scored WHERE ml_auto_drop) TO '{drop_s}' (HEADER, DELIMITER ',')"
    )
    return {
        "n_rows": int(n),
        "n_keep": int(n_keep),
        "n_drop": int(n_drop),
        "hours_keep": float(hours[0]),
        "hours_drop": float(hours[1]),
        "keep": str(keep_path),
        "drop": str(drop_path),
        "threshold": float(threshold),
    }


def print_low_scores(df: pd.DataFrame, n: int) -> None:
    if n <= 0 or "ml_score" not in df.columns:
        return
    idx = np.argsort(df["ml_score"].to_numpy())[:n]
    print(f"\n最低分 {min(n, len(idx))} 条（人工抽检）:")
    for i in idx:
        title = str(df.iloc[i].get("title", ""))[:100]
        print(f"  [{df.iloc[i]['ml_score']:.4f}] {title}")


def print_high_scores(df: pd.DataFrame, n: int) -> None:
    if n <= 0 or "ml_score" not in df.columns:
        return
    idx = np.argsort(-df["ml_score"].to_numpy())[:n]
    print(f"\n最高分 {min(n, len(idx))} 条:")
    for i in idx:
        title = str(df.iloc[i].get("title", ""))[:100]
        print(f"  [{df.iloc[i]['ml_score']:.4f}] {title}")


def main() -> int:
    ap = argparse.ArgumentParser(description="exo_entertainment Ettin reranker 文本打分")
    ap.add_argument("input", nargs="?", help="输入 CSV（含 title；可选 channel）")
    ap.add_argument("-o", "--output", type=Path, help="输出 CSV（默认 *_ettin_scored.csv）")
    ap.add_argument(
        "--glob", dest="glob_pattern", type=str,
        help="批量打分 glob",
    )
    ap.add_argument(
        "--model-size",
        default=DEFAULT_MODEL_SIZE,
        help="32m / 68m / 或完整 HF model id（默认 68m）",
    )
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--threshold", type=float, default=None, help="覆盖 calibration 阈值")
    ap.add_argument("--query", type=str, default=None, help="覆盖 calibration query")
    ap.add_argument(
        "--legacy-recall", action="store_true",
        help="使用 calibration.recall（量优先），忽略 t_like",
    )
    ap.add_argument("--chunksize", type=int, default=4096)
    ap.add_argument("--sample", type=int, default=0, help="只处理前 N 行（预览）")
    ap.add_argument("--low", type=int, default=0, help="打印分数最低的 N 条")
    ap.add_argument("--high", type=int, default=0, help="打印分数最高的 N 条")
    ap.add_argument("--device", type=str, default=None, help="mps / cuda / cpu")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--split-keep-drop", action="store_true",
        help="同步写出 *_ettin_keep / *_ettin_drop（按 ml_auto_drop）",
    )
    ap.add_argument(
        "--lite",
        action="store_true",
        help="大表快路径：只写 video_id+score，再用 DuckDB join 拆 keep/drop",
    )
    ap.add_argument(
        "--no-full-scored",
        action="store_true",
        help="--lite 时不写出完整 *_ettin_scored（仅 keep/drop + scores）",
    )
    args = ap.parse_args()

    if not args.input and not args.glob_pattern:
        ap.error("请指定 input 或 --glob")

    threshold = (
        args.threshold
        if args.threshold is not None
        else load_threshold(args.calibration, legacy_recall=args.legacy_recall)
    )
    query = load_query(args.calibration, args.query)
    model_id = resolve_model_id(args.model_size)

    from core.data_profile import file_hash

    _log(f"loading {model_id} device={args.device or 'auto'}")
    model = EttinReranker(model_id, device=args.device)
    _log(
        f"ready device={model.device} batch={args.batch_size} "
        f"max_len={args.max_length} τ={threshold:.4f} lite={args.lite}"
    )
    _log(f"query: {query[:120]}{'…' if len(query) > 120 else ''}")

    paths: list[Path] = []
    if args.glob_pattern:
        import glob as globmod

        paths = sorted(Path(p) for p in globmod.glob(args.glob_pattern))
    elif args.input:
        paths = [Path(args.input)]

    if not paths:
        print(f"[ERROR] glob 无匹配: {args.glob_pattern!r}", file=sys.stderr)
        return 1

    identity = {
        "model_id": model_id,
        "calibration_sha256": (
            file_hash(args.calibration) if args.calibration.exists() else None
        ),
        "scoring_code_sha256": file_hash(__file__),
        "ettin_code_sha256": file_hash(_SCRIPT_DIR / "core" / "ettin_reranker.py"),
        "device": model.device,
        "encoder_batch_size": args.batch_size,
        "max_length": args.max_length,
    }

    summaries = []
    for inp in paths:
        out = args.output
        if out is None:
            out = inp.with_name(f"{inp.stem}_ettin_scored.csv")
        elif len(paths) > 1:
            out = out.parent / f"{inp.stem}_ettin_scored.csv"

        if args.lite:
            if "_ettin_scored" in out.stem:
                scores_out = out.with_name(
                    out.stem.replace("_ettin_scored", "_ettin_scores", 1) + out.suffix
                )
                keep_out = out.with_name(
                    out.stem.replace("_ettin_scored", "_ettin_keep", 1) + out.suffix
                )
                drop_out = out.with_name(
                    out.stem.replace("_ettin_scored", "_ettin_drop", 1) + out.suffix
                )
            elif "_ettin_scores" in out.stem:
                scores_out = out
                keep_out = out.with_name(
                    out.stem.replace("_ettin_scores", "_ettin_keep", 1) + out.suffix
                )
                drop_out = out.with_name(
                    out.stem.replace("_ettin_scores", "_ettin_drop", 1) + out.suffix
                )
                out = out.with_name(
                    out.stem.replace("_ettin_scores", "_ettin_scored", 1) + out.suffix
                )
            else:
                scores_out = out.with_name(out.stem + "_ettin_scores" + out.suffix)
                keep_out = out.with_name(out.stem + "_ettin_keep" + out.suffix)
                drop_out = out.with_name(out.stem + "_ettin_drop" + out.suffix)
                out = out.with_name(out.stem + "_ettin_scored" + out.suffix)

            lite_summary = score_lite_file(
                inp,
                scores_out,
                model,
                query=query,
                threshold=threshold,
                chunksize=args.chunksize,
                sample=args.sample,
                batch_size=args.batch_size,
                max_length=args.max_length,
                resume=args.resume,
                run_identity=identity,
            )
            summary = dict(lite_summary)
            if args.split_keep_drop:
                join_summary = apply_lite_join(
                    inp,
                    scores_out,
                    threshold=threshold,
                    keep_path=keep_out,
                    drop_path=drop_out,
                    scored_full_path=None if args.no_full_scored else out,
                )
                summary.update(join_summary)
            summaries.append(summary)
            _log(
                f"lite done {inp.name}: rows={summary.get('n_rows')} "
                f"keep={summary.get('n_keep')} drop={summary.get('n_drop')}"
            )
            continue

        summary = score_file(
            inp,
            out,
            model,
            query=query,
            threshold=threshold,
            chunksize=args.chunksize,
            sample=args.sample,
            batch_size=args.batch_size,
            max_length=args.max_length,
            resume=args.resume,
            split_keep_drop=args.split_keep_drop,
            run_identity=identity,
        )
        summaries.append(summary)
        _log(
            f"done {inp.name}: rows={summary.get('n_rows')} "
            f"keep={summary.get('n_keep')} drop={summary.get('n_drop')} "
            f"→ {out}"
        )

        if (args.low > 0 or args.high > 0) and (
            args.sample > 0 or summary.get("n_rows", 0) <= 5000
        ):
            preview = pd.read_csv(out, encoding="utf-8-sig")
            print_low_scores(preview, args.low)
            print_high_scores(preview, args.high)

    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

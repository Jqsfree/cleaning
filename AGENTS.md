# Repository Guidelines

## Project Overview

This is a data-cleaning pipeline for YouTube language-teaching video metadata. Raw CSV exports flow through sequential phases (normalize → QC → dedup → clean → evaluate → recall) and land in `data/runs/<name>/`. The project targets Python 3.13+ and uses DuckDB as its primary data engine.

## Project Structure

```
02_脚本/           Main scripts — one `phaseN_*.py` per pipeline stage
02_脚本/core/      Shared library (cleaner, playlist, scoring, sop, progress, reports)
02_脚本/rules/     Rule definitions used during cleaning
raw/               Raw input CSVs from YouTube exports
data/runs/         Pipeline outputs, one subdirectory per run
monitor.py         Queue monitoring helper
项目记录.md         Run log (auto-appended by each phase)
```

## Key Commands

**Phase 0 — Normalize raw data into a baseline Parquet:**
```bash
python3 02_脚本/phase0_normalize.py raw/input.csv -o data/runs/001_baseline/ --min-duration 10
```

**Phase 2 — Run LLM-based text quality check on a sample:**
```bash
python3 02_脚本/phase2_qc.py data/runs/001_baseline/audit_sample.parquet -o data/runs/002_audit/ -w 20
```

**Cross-batch dedup against prior deliveries:**
```bash
python3 02_脚本/phase_dedup.py baseline.parquet -d data/runs/old_run/deliver/ -o deduped.parquet
```

**Run subsequent phases** (clean, evaluate, recall) by passing the output of the prior stage as input, following the same `-o <dir>` pattern.

## Coding Conventions

- **Python 3.13+**. Keep imports at the top of each file; `02_脚本/core/` is added to `sys.path` locally in each phase script rather than installed as a package.
- Indentation is 4 spaces. Scripts are self-contained CLIs using `argparse`.
- Phase scripts follow a thin-wrapper pattern: they import core logic from `02_脚本/` modules (`chunk_text_qc_v2.py`, `core/sop.py`, etc.) and add CLI + run-logging boilerplate.
- Use `duckdb` for large-table work and `pandas` for smaller post-processed frames.
- Chinese is used for inline comments, docstrings, and log output; English for variable names, function names, and file paths.

## Testing

No formal test suite exists. Validate phases by running them against a small known input and inspecting the output Parquet/CSV. Spot-check row counts and retention percentages against prior baselines.

## Commit & Branch Conventions

This repository is not currently under Git version control. If initialized, prefer short Chinese commit messages describing the phase change (e.g., `phase0: 增加 duration 过滤`). Keep raw data files out of the repo via `.gitignore`.


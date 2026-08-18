# exo_service cascade L1+L2 Implementation Plan

> **For agentic workers:** Implement task-by-task. L3/L4 are out of scope this round.

**Goal:** From quality CSV, high-precision text DROP (L1) then industry route (L2) → `commercial_service_candidates`.

**Architecture:** TOML rules + DuckDB CLI under `categories/exo_service/` and `tools/`.

**Tech Stack:** Python 3.13, DuckDB, pytest, tomllib.

## Files

| Path | Role |
|------|------|
| `categories/exo_service/rules/cascade_l1_drop.toml` | L1 certain-noise patterns |
| `categories/exo_service/rules/cascade_l2_route.toml` | L2 industry patterns |
| `categories/exo_service/cascade_text.py` | L1+L2 runner |
| `tools/run_exo_service_cascade_text.py` | CLI thin wrapper |
| `tests/test_exo_service_cascade_text.py` | unit tests |
| `recipe.toml` | point flow to new cascade |

## Tasks

- [x] Design doc
- [ ] L1/L2 TOML
- [ ] `cascade_text.py` + CLI
- [ ] Tests
- [ ] Run on quality → `06_tools/cascade_v2/`
- [ ] Update recipe notes

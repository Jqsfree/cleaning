# Baseline Statistics

**Input:** `raw/化妆美妆教程类_11c8ae10_records6.10.csv`
**Generated:** 2026-06-10 16:30:34
**Elapsed:** 6.2s

| Stage | Retained | Dropped |
|-------|----------|--------|
| Raw input | 1,803,890 | -- |
| Null filter | 1,789,012 | 14,878 |
| Dedup | 1,789,012 | 0 |
| Damaged | 1,640,525 | 148,487 |
| Duration | 1,397,661 | 242,864 |
| **Final** | **1,397,661** | **406,229** |

**Retention:** 77.5%
**Total Duration:** 36.4万h

**Next:** Phase 2
```bash
python3 phase2_sample.py data/runs/beauty/001_baseline/化妆美妆教程类_11c8ae10_records6.10_raw.parquet -o runs/002_audit/
```

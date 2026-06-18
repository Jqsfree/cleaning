# Baseline Statistics

**Input:** `../raw/tow-person/双人对话-3-ty_67c24de5_records (1).csv`
**Generated:** 2026-06-16 17:33:44
**Elapsed:** 1.2s

| Stage | Retained | Dropped |
|-------|----------|--------|
| Raw input | 137,097 | -- |
| Null filter | 125,904 | 11,193 |
| Dedup | 125,904 | 0 |
| Damaged | 125,797 | 107 |
| Duration | 112,818 | 12,979 |
| **Final** | **112,818** | **24,279** |

**Retention:** 82.3%
**Total Duration:** 4.6万h

**Next:** Phase 2
```bash
python3 phase2_sample.py ../data/runs/tow-person/001_baseline/双人对话-3-ty_67c24de5_records (1)_raw.parquet -o runs/002_audit/
```

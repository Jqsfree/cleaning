# Baseline Statistics

**Input:** `raw/单人口播_cd7de25d_records (5).csv`
**Generated:** 2026-06-10 09:40:01
**Elapsed:** 0.9s

| Stage | Retained | Dropped |
|-------|----------|--------|
| Raw input | 282,990 | -- |
| Null filter | 282,328 | 662 |
| Dedup | 282,328 | 0 |
| Damaged | 280,576 | 1,752 |
| Duration | 279,895 | 681 |
| **Final** | **279,895** | **3,095** |

**Retention:** 98.9%
**Total Duration:** 5.9万h

**Next:** Phase 2
```bash
python3 phase2_sample.py data/runs/broadcast/001_baseline/单人口播_cd7de25d_records (5)_raw.parquet -o runs/002_audit/
```

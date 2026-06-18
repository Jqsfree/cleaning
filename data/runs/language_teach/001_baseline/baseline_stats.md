# Baseline Statistics

**Input:** `../raw/language_teach/语言教学视频13大批量_42185ccf_records.csv`
**Generated:** 2026-06-15 16:55:26
**Elapsed:** 1.0s

| Stage | Retained | Dropped |
|-------|----------|--------|
| Raw input | 144,854 | -- |
| Null filter | 138,407 | 6,447 |
| Dedup | 138,407 | 0 |
| Damaged | 138,040 | 367 |
| Duration | 126,180 | 11,860 |
| **Final** | **126,180** | **18,674** |

**Retention:** 87.1%
**Total Duration:** 4.9万h

**Next:** Phase 2
```bash
python3 phase2_sample.py ../data/runs/language_teach/001_baseline/语言教学视频13大批量_42185ccf_records_raw.parquet -o runs/002_audit/
```

# Baseline Statistics

**Input:** `raw/语言教学课_ea79a3b9_records.csv`
**Generated:** 2026-06-09 11:25:48
**Elapsed:** 5.4s

| Stage | Retained | Dropped |
|-------|----------|--------|
| Raw input | 1,492,862 | -- |
| Null filter | 1,487,929 | 4,933 |
| Dedup | 1,487,929 | 0 |
| Damaged | 1,400,507 | 87,422 |
| Duration | 951,087 | 449,420 |
| **Final** | **951,087** | **541,775** |

**Retention:** 63.7%

**Next:** Phase 2
```bash
python3 phase2_sample.py data/runs/001_baseline/语言教学课_ea79a3b9_records_raw.parquet -o runs/002_audit/
```

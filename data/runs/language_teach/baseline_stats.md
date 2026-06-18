# Baseline Statistics

**Input:** `../raw/language_teach/语言教学-视频11_669b7433_records.csv`
**Generated:** 2026-06-12 15:47:40
**Elapsed:** 0.5s

| Stage | Retained | Dropped |
|-------|----------|--------|
| Raw input | 15,310 | -- |
| Null filter | 15,310 | 0 |
| Dedup | 15,310 | 0 |
| Damaged | 15,283 | 27 |
| Duration | 13,319 | 1,964 |
| **Final** | **13,319** | **1,991** |

**Retention:** 87.0%
**Total Duration:** 4,798.5h

**Next:** Phase 2
```bash
python3 phase2_sample.py ../data/runs/language_teach/语言教学-视频11_669b7433_records_raw.parquet -o runs/002_audit/
```

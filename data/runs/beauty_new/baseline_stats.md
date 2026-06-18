# Baseline Statistics

**Input:** `../raw/beauty_NEW/化妆美妆教程-视频9-中文_7b1fd80d_records.csv`
**Generated:** 2026-06-12 15:51:17
**Elapsed:** 1.4s

| Stage | Retained | Dropped |
|-------|----------|--------|
| Raw input | 89,569 | -- |
| Null filter | 88,329 | 1,240 |
| Dedup | 88,329 | 0 |
| Damaged | 87,892 | 437 |
| Duration | 72,269 | 15,623 |
| **Final** | **72,269** | **17,300** |

**Retention:** 80.7%
**Total Duration:** 1.1万h

**Next:** Phase 2
```bash
python3 phase2_sample.py ../data/runs/beauty_new/化妆美妆教程-视频9-中文_7b1fd80d_records_raw.parquet -o runs/002_audit/
```

# 排除类资产目录

由 `export_reject_assets.py` / `accumulate_reject_assets.py` 写入；**勿手改大量 CSV**。

## 结构

```text
data/assets/rejects/
  {reject_tag}/
    proposed.csv           # 自动提案（弱监督）
    human_validated.csv    # 抽样 confirm/correct（金标；可暂缺）
    manifest.json          # n_proposed / n_by_modality / registry_version
  _metrics/
    by_source.json         # overturn / trust_status
  README.md
```

- 同一 `reject_tag`，行内用 `modality=text|thumb|fusion` 区分通道  
- Registry：`02_脚本/categories/_shared/reject_registry.toml`（可演进，勿删历史 id）

## 初版种子（2026-07-27）

- 工作批：`data/runs/film_tv/_reject_seed_0727/`（`03_qc/reject_proposed.csv`）  
- 来源：存量 `human_0724` drop + `005_clean` 多批 drop/thumb_qc  
- 规模：约 28.8 万 proposed（text ~10 万 / thumb ~18.5 万，thumb 多为 `provisional:thumb_fail`）  
- **尚无** `human_validated` → metrics 会报 `proposed_without_validation`（预期内）  
- 复跑：`PYTHONPATH=02_脚本 python scripts/_seed_reject_assets_once.py`

训练 / 收紧阈值前请先抽样验证再 export。

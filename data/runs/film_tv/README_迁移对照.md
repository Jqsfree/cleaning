# film_tv 路径迁移对照

新跑强制：`data/runs/film_tv/{human|machine}_{batch}/`（见仓库根 `AGENTS.md`）。

本文件记录 **2026-07-27** 存量整理（不改历史 parquet 内部路径）。

## 批次目录

| 旧路径 | 新路径 | 说明 |
|--------|--------|------|
| `data/runs/影视剧人工724/` | `data/runs/film_tv/human_0724/` | 已迁；根上保留同名 **符号链接** 便于旧命令 |
| `data/runs/影视剧人工724-2/` | `data/runs/film_tv/human_0724_2/` | 同上 |
| （建议）723 交付 | `data/runs/film_tv/human_0723/07_deliver/` | 清晰度产物原在 `film_tv/resolution/` |

## 根目录交付物

| 旧路径 | 新路径 |
|--------|--------|
| `data/runs/影视剧_723_724_724-2_合并去重_大于720.csv` | `data/runs/film_tv/07_deliver/` |
| `data/runs/影视剧_723_724_处理报告.md` | `data/runs/film_tv/07_deliver/` |

## 阶段编号对照

| 旧 | 新 |
|----|----|
| `005_clean` | `05_clean` |
| `002_audit` | `02_sample` + `03_qc` |
| 批次内 `*_大于720.csv`（夹在 quality） | 拷贝/登记到本批 `07_deliver/` |

## 规则

- **禁止**再在 `data/runs/` 根新增中文批目录或合并 CSV。
- 对外只从 `07_deliver/`（批次内或 `film_tv/07_deliver/`）取数。
- 新开批：`…/pipeline/run.py … --source human|machine -o data/runs/film_tv/{source}_{batch}/`

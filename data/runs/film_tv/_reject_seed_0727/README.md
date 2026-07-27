# 排除类资产种子批 `_reject_seed_0727`

非交付批次；仅用于从**存量** drop / thumb_qc 灌 `proposed` → `data/assets/rejects/`。

| 路径 | 说明 |
|------|------|
| `03_qc/reject_proposed.csv` | text+thumb 合并提案 |
| `04_rules/reject_opt_suggestions.md` | 自动建议（未改生产配置） |
| 复跑 | 仓库根：`PYTHONPATH=02_脚本 python scripts/_seed_reject_assets_once.py` |

详见 [AGENTS.md](../../../AGENTS.md)「共享排除类资产」与 [data/assets/rejects/README.md](../../assets/rejects/README.md)。

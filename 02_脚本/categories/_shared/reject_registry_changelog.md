# reject registry changelog（与 reject_registry.toml [changelog] 同步备忘）

## v1 (2026-07)

- 首版：id 对齐 `film_tv/rules/blacklist.toml` 的 `category=`
- 增加示例弃用项 `docu_legacy` → `maps_to = documentary`（演示演进，非业务必用）
- 原则：弃用勿删；未知标签可 `provisional:*`；人工不做全量细类标注

# src/dataclean — WIP / 冻结

**不是生产入口。** 日常请用 [`02_脚本/`](../../02_脚本/)，见仓库根 [`AGENTS.md`](../../AGENTS.md)。

## 状态

- Phase CLI（`dataclean.cli`）**已冻结**，调用须 `--allow-wip`。
- **禁止在本包加新功能**（新 cleaner、新门禁、新批次契约一律进 `02_脚本/`）。
- 品类 cleaner 注册表**单源**：[`02_脚本/core/category_registry.py`](../../02_脚本/core/category_registry.py)。
  本包 `pipeline/registry.py` 仅 re-export；勿再维护第二份 `_CLEANERS`。

## 与生产的关系

```text
src/dataclean/ ──sys.path──▶ 02_脚本/categories|core
                 （实验包装）     （权威实现）
```

长期若要包化，只考虑把 `02_脚本/core` 做成可安装薄包，**不要**复活统一 Phase0–7 CLI。

## 允许做什么

- 读代码、对照实验、`--allow-wip` 下跑旧 Phase 包装做对比。
- 修明显阻塞实验的导入错误（仍须不漂移生产行为）。

## 不要做什么

- 不要把 `src/dataclean` 写成文档/脚本的默认命令。
- 不要在此复制 `clean_gates` / `batch_layout` / reject 闭环。
- 不要 `register_category()`（已禁用）。

# tools/ — 脚本工程约定

`tools/` 放**非阶段化**的工具：批次的合并/抽样运维、品类专用的级联与校准脚本。
凡是「按 recipe 阶段顺序跑」的都在 `pipeline/`，不放这里。

## 目录组织

| 目录 | 用途 |
|------|------|
| `batch_ops/` | 跨品类批次运维：`merge_csvs`（合并去重 → Bronze）、`sample_qc`（抽样 QC） |
| `agri_v1/` | 农业 v1 工具套件（spec §4~8） |
| `*.py`（平铺） | 品类专用一次性/校准脚本（`run_exo_*_cascade_*.py`、`eval_*`、`apply_*`） |

新工具优先归入 `batch_ops/`（跨品类）或 `<品类>/`（单品类）子包；只有确实
一次性、且无复用价值的脚本才平铺。

## 单脚本硬性约定

1. **解释器**：`#!/usr/bin/env python3`。禁止写死 `#!/home/jqs/miniconda3/envs/...`
   （机器相关，换机即失效）。仓库根 `.venv/bin/python3` 为本机可用解释器。
2. **文档**：模块 docstring 必须写清「用途 / 口径 / 用法」，用法给出**可直接复制**的命令。
3. **类型**：`from __future__ import annotations`；函数签名带类型标注。
4. **分层**：**纯函数 + `main()` 薄 CLI**。所有业务逻辑放纯函数（不打印、不 `sys.exit`），
   `main()` 只做 argparse → 调纯函数 → 汇总打印 → 退出码。纯函数须可被 `tests/` 直接 import。
   参照 `agri_v1/make_human_qc_template.py` 的 `build_template()` / `main()`。
5. **路径**：由 `_SCRIPT = Path(__file__).resolve().parents[N]`、`_REPO = _SCRIPT.parent`
   推导。**禁止硬编码绝对路径**（用户目录路径只能作为 argparse 的默认值常量，且须可覆盖）。
6. **CLI**：`argparse`，路径参数用 `type=Path`，布尔开关用 `action="store_true"`。
   失败打 `[ERROR]` 并 `return 1`（由 `raise SystemExit(main())` 传出）。
7. **复用**：公共能力走 `core.*`（`core.io` / `core.log` / `core.sql_builder` /
   `core.batch_layout` / `core.category_labels` / `core.sop` / `core.progress` /
   `core.run_manifest`），**禁止在脚本内重复实现**。
8. **SQL**：字面量一律经 `core.sql_builder.sql_escape` 转义，禁止 f-string 直接拼路径。
9. **命名**：产物遵循「原文件名 stem + 操作」，见仓库根 `CLAUDE.md` 产物命名规范；
   品类中文名统一查 `core.category_labels`，禁止在各脚本内散落第二份映射。
10. **留痕**：跑完落 `core.sop.write_run_log`（追加 `项目记录.md` + 写 `run_log.md`）。
    冒烟测试/单测须用 `--no-log` 或不走 CLI，避免污染项目记录。

## 测试

纯函数在 `tests/test_<suite>_<script>.py` 覆盖，写法定型参考
`tests/test_exo_service_cascade_text.py`：

```python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_脚本"))

from tools.batch_ops.merge_csvs import merge_csvs  # noqa: E402
```

运行（本机无 conda，用仓库 `.venv`）：

```bash
.venv/bin/python3 -m pytest tests/ -q
```

## 临时产物

试跑产物一律落 `work/{category}_{主题}_{MMDD}/`，收尾后归档结论、删除中间文件
（见 `work/README.md`）。

## 标题主题闭环（metadata人标）

新增入口：`batch_ops/topic_loop.py`，支持 import-gold/train/benchmark/run/verify/evaluate/propose/revise/close。
操作与验收协议：`docs/title_topic_loop_v1.md`。农业已结束，仅历史对照；机器候选keep不等同人标验收完成。

## metadata 管线 v2（2026-09-21）

复用 `batch_ops/topic_loop.py`，新增 profile/replay/compare/next/release，并给 run 增加历史人标回放和冻结验收目标。操作、门禁、退出码及实测结果见 `docs/exo_metadata_pipeline_v2.md`。每轮记录保存在自身目录，避免把回归实验追加到全局人工进度表。

metadata 排错新增 `batch_ops/topic_loop.py enrich-gold`（追溯补字段）和 `diagnose`（规则误伤/误保留诊断）；`propose/revise` 用 `--validation-gold` 提供独立频道回归。用法及本轮实际结果见 `docs/exo_metadata_diagnostics_20260921.md`。它们共用现有入口，不需要新建品类脚本。

metadata 日常入口统一到 `02_脚本/pipeline/metadata.py`；此处 `batch_ops/topic_loop.py` 是兼容包装。批次管理使用 start/status/retry/specialist/publish/catalog，完整用法见仓库 README 和 `docs/exo_metadata_infrastructure_v3.md`。cook 旧打分的 --resume 现在要求可信 checkpoint，已有无 checkpoint 输出需要另选输出路径重新计算。

## 文本向量缓存（MiniLM 编码复用）

`score_*_text.py` 的 MiniLM 编码是唯一重活（百万行 ≈ 20 分钟），LR 头瞬时。
`core/text_embeddings.py` 把 `sha1(title) → 向量` 存进 DuckDB，**同 title 只编码一次**：

- 位置：`data/assets/embeddings/text/{encoder_key}.duckdb`（fp16；100 万 title ≈ 0.9 GB）
- 键只与 title 文本有关 → 换 LR 头、换 τ、重打分、增量批次、跨 run 全部复用
- `encoder_key` 由 snapshot 路径 + 维度派生 → 换编码器自动换表，不会串味
- 关闭：`--no-embed-cache` 或 `TEXT_EMBED_CACHE=0`；改目录用 `TEXT_EMBED_CACHE_DIR`
- 并发：库被另一进程独占时自动降级为「只读复用 → 直接编码」，不会因锁冲突报错
- 预热：`python 02_脚本/tools/score_*_text.py <clean.csv> -o work/prime.csv` 跑一遍即把全量 title 灌进缓存

约定：**重训模型不必重跑编码**；只有 title 从未见过才会走 MiniLM。

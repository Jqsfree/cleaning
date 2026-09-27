# 公开数据项目目录

本仓库集中保留公开的数据采集和清洗项目，各项目保持自己的运行目录、配置与环境。

- **通用清洗**：根目录 `02_脚本/`，使用原 README 和 AGENTS.md。
- **EXO 专项清洗**：`projects/cleng_exo/`。
- **体育清洗 SOP v4**：`projects/sport-live/`。
- **早期体育清洗**：`legacy/sport_event/`，用于历史批次复现。
- **体育数据采集**：`projects/tiyu_project/`，包含原采集脚本及已入库的批次 manifest/state 资料。

运行前先进入对应项目目录；各入口的依赖与参数以原文件为准，本次没有将不同流程强行串联或统一算法。
源项目已提交的运行记录按原样保留，不代表本机存在相应数据文件或可以直接恢复远端任务。
子项目内部旧仓库地址和机器绝对路径仅作历史参考，需要按新克隆位置调整。

`tiyu_project` 的全部原始文件与历史均保留；所有原 refs 额外存为 `imported/tiyu_project/*` 标签。
迁移清单见 `.repository-history/final-organization.json`；第一轮体育/EXO 合并记录见 `docs/REPOSITORY_CONSOLIDATION.md`。
本公开仓库未导入任何私有项目内容。

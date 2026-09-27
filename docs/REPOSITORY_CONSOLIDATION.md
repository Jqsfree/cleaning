# 清洗仓库合并说明

本次将三个公开仓库的默认分支内容和完整祖先提交历史合入 `Jqsfree/cleaning`。
采用不 squash 的 Git subtree 导入，原始文件内容、可执行位及目录内部结构均保持一致。
原始提交标识、目录树标识、分支和文件数量记录在 `repository-sources.json`。

## 项目入口

- `projects/cleng_exo/`：EXO 专项版本，包含通用版本没有的农业元数据过滤、缩略图处理及模型校准参数。
- `projects/sport-live/`：体育清洗 SOP v4 流程、规则版本、测试和批次记录。
- `legacy/sport_event/`：早期体育清洗脚本和记录，保留旧流程复现能力。
- 根目录 `02_脚本/`：现有通用清洗流程。

请进入相应项目目录再运行原有命令；不同项目的 Python 路径、配置和依赖独立。
EXO 的 167 个文件中有 132 个内容已出现在原 cleaning 默认分支，但其余 35 个仍需保留。
本次保留独立运行副本，不将不同版本的规则或核心实现直接覆盖到通用入口。
Git 对相同 blob 自身去重；后续统一代码需要独立的行为回归验证。

## 测试

根目录 `pytest` 默认只发现根目录 `tests/`，避免导入子项目的同名模块导致测试环境串用。
根目录现有检查命令：

```bash
python -m pytest tests/test_adaptive_api.py tests/test_fetch_resolution_height.py tests/test_fetch_yt_definition.py -q
```

体育项目测试独立运行（安装该项目依赖后）：

```bash
cd projects/sport-live
python -m pytest tests/ -q
```

源仓库 `.github/workflows/` 位于子项目内部，不会自动作为本仓库工作流执行。
本次没有将外部服务任务或旧部署工作流接入主仓库。

## 完整性与恢复

三个源仓库各只有一个分支，没有标签、Issues、PR 或 Releases（迁移准备时检查）。
源仓库完整 Git bundle 和仓库元数据另存于本地备份；它们不包含未提交的本地数据或 GitHub 密钥、Webhook 等管理设置。

合并必须使用 merge commit，不能 squash/rebase，否则本次导入的源历史可能不再可达。
每个来源的提交必须是最终主分支的祖先，并且以下目录树应与源提交的根目录树一致：

```bash
git rev-parse HEAD:projects/cleng_exo
git rev-parse HEAD:projects/sport-live
git rev-parse HEAD:legacy/sport_event
```

旧仓库删除前需重新检查源分支是否变化，确认主分支已包含上述历史和文件。
外部机器的 git remote、绝对路径、定时任务需按实际使用位置更新；这些外部配置未在本次迁移中检查或修改。

## 范围

未导入任何私有仓库内容；私有项目保留独立。
`tiyu_project` 的数据采集与 `youtube_down` 的下载承担不同功能，也保留独立。

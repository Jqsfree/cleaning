# exo_service 四层级联设计（2026-08-17）

## 问题

旧路径（文本语义小模型 keep_for_visual → 泛化 VL「是否商业服务」）在人工锚点上不合格：语义留存仍约 37% 人工合格；泛化缩略图 prompt 不稳定。

## 目标

从 `01_quality` 重跑，按层职责切开：

| 层 | 职责 | 输出 |
|----|------|------|
| L1 | 文本**高精度 DROP**（仅确定噪声） | drop / remain |
| L2 | 文本**行业路由**（不判合格，允许误报） | `commercial_service_candidates` + `industry` |
| L3 | 缩略图/故事板 **CLIP** 粗筛：有人？操作？服务/劳动场景？ | clip_pass |
| L4 | **仅子类边界** VLM，子类专用 prompt | thumb/sb T |

交付 KPI 仍只认人工合格率。U/中间带不自动当 drop。

## 废弃（本 cascade 路径）

- `exo_service_text_clf_f` 全量语义否决 → keep_for_visual 作为主池
- 单一 `vision_thumb`「是否商业服务」对全量/大样本一刀切

存量 `06_tools/text_semantic/` 只读归档，不作为权威 keep。

## L1 — 确定噪声 DROP

只删标题/频道上可确定的非目标话术（词边界优先）。用户清单：

podcast, interview, CEO, business tips, marketing, startup, company profile,
commercial, advertisement, review, tour, documentary, news, lecture,
conference, webinar, how to start, talking head, panel discussion, presentation

**风险词**（需 dry-run 看误杀）：`commercial` / `review` / `tour` 可能撞到店内内容。默认词边界；若误杀偏高再收紧（例如 `company commercial`、`product review`）。

不做：泛化 `how to`、美妆教程整类抹杀（交给 L2+L3+L4）。

## L2 — 行业路由

正则/关键词 → 多标签行业（允许一条多行业；主标签取优先序）：

hair | beauty | food_service | retail | repair | cleaning | hospitality | healthcare | automotive | pet_service

- 命中任一 → 进入 `commercial_service_candidates`
- 未命中 → `unrouted`（不出候选，**不是** L1 certain-noise；可另表统计）

## L3 — CLIP（本阶段后置）

零样本缩略图，三问独立打分后合取（阈值标定后再锁）：

1. person present  
2. clear manipulation / service action  
3. service / labor scene  

不全量 VLM。

## L4 — 子类 VLM（本阶段后置）

仅对 L3 边界带 / 抽样子类跑 VLM，例如：

- hair/beauty：「是否有人正在为顾客理发、美容或个人护理？」
- food_service：「是否有人在餐饮后厨/柜台执行服务劳动？」
- repair / retail / cleaning：各专用一句

## 分阶段落地

1. **本轮**：L1+L2 规则 + DuckDB 工具，对 `machine_0813` quality 全量出 candidates  
2. **下轮**：L3 CLIP 抽样标定阈值  
3. **再下轮**：L4 子类 VLM + 人工双过池  

## 输入

`data/runs/exo_service/machine_0813/01_quality/商业服务_merged_0813_quality_0813.csv`

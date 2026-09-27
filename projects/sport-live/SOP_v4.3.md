# 体育赛事数据集构建 SOP v4.3

---

## 核心原则

**主目标：最大化真实体育赛事数据量**

**约束：QC Pass Rate ≥ 70%**

在满足 70% 精度约束的前提下，尽可能保留更多真实体育内容。

禁止为了提升 Precision 而过度加黑名单——疯狂加黑名单会导致 Retention 崩溃，得到一个高精度但数据量极少的废掉的数据集。

注意：

这里的 70% 指的是最终 Keep 数据集随机抽样后的体育赛事占比（QC Pass Rate），不是 Keep Rate，不是 Retention Rate。

**v4.3 关键修正 (2026-06-06):**
- pass2 = 硬过滤，r2 = 软减分。非体育频道必须用 pass2
- merge_rules 已修复 (v4.2): scoring.py 改用 pattern 列表避免 split("|") bug，专属 pass2 规则写共享 `rules/blacklist.toml` 是正确 SOP
- Phase 8 召回后必须再次 pass2 过滤再 QC
- keep + recall 独立交付，禁止自动合并

---

# 命名规范

## 批次目录

```
data/runs/{sport}_{batch}/
```

`batch` 用英文序数：`one` / `two` / `three` / `four` ...

示例：
```
data/runs/shotput_one/    ← 体育解说-铅球_c875f76d
data/runs/shotput_two/    ← 体育解说-铅球_17a842ff
data/runs/marathon_one/   ← 马拉松第一批
```

每个批次对应**一个原始文件**，禁止一个批次目录混入多个原始文件。

## 目录内文件命名

```
data/runs/shotput_one/
├── 001_baseline/
│   └── {原始文件名}_raw.parquet
│
├── 005_clean/
│   ├── {原始文件名}_clean.parquet          ← Phase 2 共享规则输出
│   ├── run01/
│   │   ├── {原始文件名}_run01_keep.parquet
│   │   └── {原始文件名}_run01_drop.parquet
│   ├── run02/
│   │   ├── {原始文件名}_run02_keep.parquet
│   │   └── {原始文件名}_run02_drop.parquet
│
├── qc/
│   ├── {原始文件名}_run01_keep_qc.parquet
│   ├── {原始文件名}_run01_drop_qc.parquet
│   ├── {原始文件名}_run02_keep_qc.parquet
│   └── {原始文件名}_run02_drop_qc.parquet
│
├── rules/
│   ├── blacklist.toml        ← 专属频道名黑名单
│   ├── entities.toml         ← 专属实体词（如有）
│   ├── whitelist.toml        ← 专属白名单（如有）
│   └── rule_log.md
│
├── rules_backup/             ← 冻结时备份，供同运动新批次复用
│   └── blacklist.toml
│
├── deliver/
│   ├── {原始文件名}_run{N}_keep_final.parquet   ← 最终 KEEP（全部列）
│   ├── {原始文件名}_run{N}_keep_final.csv       ← 最终 KEEP（全部列，CSV）
│   └── {原始文件名}_run{N}_keep_final_ids.csv   ← 仅 video_id
│
└── project_log.md
```

## 临时召回文件（Phase 8）

```
005_clean/run{N}/
├── {原始文件名}_run{N}_recovered_candidate.parquet
└── recover_rules_temp.py    ← QC 通过后立即删除
```

## 禁止事项

- 禁止在文件名里混用哈希 ID 和批次号
- 禁止跨批次目录引用中间产物
- 禁止未经 QC 直接合并 recovered_candidate

---

# 共享规则版本管理

## 目录结构

```
02_脚本/rules/
├── current/                  ← 当前生效版本（软链接或复制）
│   ├── blacklist.toml
│   ├── entities.toml
│   └── whitelist.toml
└── releases/
    ├── v001/
    │   ├── blacklist.toml
    │   ├── entities.toml
    │   └── whitelist.toml
    ├── v002/
    └── v003/
```

## 版本号规则

每次共享规则任何文件发生修改，必须：

1. 将 `current/` 整体复制到 `releases/v{N+1}/`
2. 在 `releases/v{N+1}/` 里修改规则文件
3. 将 `current/` 更新为新版本内容
4. 在 project_log 记录使用的版本号

```bash
# 升版本
cp -r 02_脚本/rules/current/ 02_脚本/rules/releases/v019/
# 修改 releases/v019/ 里的规则文件
# 更新 current/
cp -r 02_脚本/rules/releases/v019/ 02_脚本/rules/current/
```

## project_log 必须记录版本号

每次 Phase 2 运行，project_log 必须写明：

```
Shared Rules Version: v019
```

否则该批次结果无法复现。

---

# 规则分层管理

## 三层结构

```
02_脚本/rules/current/                 ← 共享规则（所有数据集共用）
├── blacklist.toml                     ← 只放内容类型规则
├── entities.toml                      ← 跨运动通用实体词
└── whitelist.toml                     ← scoring 规则

data/runs/{sport}_{batch}/rules/       ← 专属规则（单一运动）
├── blacklist.toml                     ← 频道名黑名单（全部放这里）
└── rule_log.md
```

## 共享黑名单准入条件

**允许进入共享 blacklist.toml：**
- 内容类型规则：kpop、gaming 通用词、宗教、动漫、播客等
- 与具体运动无关，所有数据集都会遇到的污染

**频道名进入共享黑名单的条件：**
频道名规则并非一律禁止。同时满足以下条件可进入共享库：

1. 该频道是纯非体育内容（无任何体育赛事视频）
2. 在来源数据集的 QC 样本里 TP = 0
3. 必须标注 `# ⚠️ 人工审核加入` 及来源

**禁止进入共享 blacklist.toml：**
- 带运动前缀的专属 title 规则
- 未满足上述条件的频道名规则（放专属目录）

## 共享 entities.toml 准入条件

- 跨运动通用的赛事名、组织名、同义词
- 每条加 `@sport` 注释标记来源：

```toml
"SW561",        # @sport: american_football | @source: run02 DROP FN | @date: 2026-06-04
"FEI",          # @sport: equestrian | @source: run01 DROP FN | @date: 2026-06-03
```

## 专属规则生命周期

```
Phase 5 新增专属规则
  → Phase 6 过滤 → Phase 7 QC ≥ 70%
  → 冻结，备份至 rules_backup/
  → 同运动新批次直接复用 rules_backup/
  → 频道名规则永久留在专属目录，不推广
  → 仅跨运动有效的内容类型规则，经确认后推广至共享库并升版本
```

## 规则退役机制

每处理 10 个批次，执行一次规则审查：

1. 统计共享黑名单每条规则在最近 10 个批次的命中次数
2. 连续 5 个批次命中为 0 的规则 → 标记为 `deprecated`
3. 标记 deprecated 的规则下一版本删除，升版本号

```toml
# [DEPRECATED v023 - 连续5批次命中0] 待 v024 删除
# pattern = "\\b(old_channel_name)\\b"
```

---

# 规则变更记录（强制）

任何规则新增、删除、修改，必须记录以下全部字段：

```
## 2026-06-03

### Rule: r4_esports_channels

- 操作: 新增 5 个 esports 频道正则
- 来源样本:
    - run02_keep_qc, video_id: abc123, 标题: "ESL Pro League S18 - G2 vs NaVi", 判定: FP
    - run02_keep_qc, video_id: def456, 标题: "StarLadder Berlin Major Highlights", 判定: FP
- 预期: 减少 ~20 FP, 0 TP 损失
- 实际: run02 QC Precision 74.2%，+5.1pp
```

**来源样本字段必须包含：QC轮次、video_id、标题、判定结果。**

禁止只写"来源：run02 FP 样本"而不列具体条目。

单轮新增规则数超过 **20 条**，必须人工审批后才能执行。

---

# 规则文件改动前必须备份

**任何规则文件改动前，必须先备份当前版本。**

```bash
# 共享规则升版本即为备份（见共享规则版本管理章节）

# 改动专属规则前
cp data/runs/{sport}_{batch}/rules/blacklist.toml \
   data/runs/{sport}_{batch}/rules/blacklist.toml.bak.$(date +%Y%m%d)
```

备份文件保留至该批次冻结交付后删除。

禁止在未备份的情况下直接修改任何规则文件。

---

# Phase 1：基础过滤与去重

脚本：

`phase0_normalize.py`

输出：

`001_baseline/{原始文件名}_raw.parquet`

处理：

* video_id 去重
* 空标题过滤
* 损坏数据过滤
* 超短视频过滤
* 无效 URL 过滤

---

# Phase 2：共享规则过滤

脚本：

`clean_sports_v3.py`

规则来源：

`02_脚本/rules/current/`

project_log 必须记录：

`Shared Rules Version: v{N}`

输出：

`005_clean/{原始文件名}_clean.parquet`
`005_clean/run01/{原始文件名}_run01_keep.parquet`
`005_clean/run01/{原始文件名}_run01_drop.parquet`

---

# Phase 3：KEEP + DROP 随机抽样 QC

脚本：

`chunk_text_qc_v2.py`

## 抽样量根据数据规模决定

| KEEP 规模 | 抽样量 |
|-----------|--------|
| < 50,000 | 300 条 |
| 50,000 ~ 500,000 | 500 条 |
| > 500,000 | 1,000 条 |

DROP 抽样量与 KEEP 相同。

统计：

* KEEP 中：体育赛事数量 / 非体育数量 → QC Pass Rate (Precision)
* DROP 中：体育赛事数量 / 非体育数量 → FN Rate (误杀率)

## Retention Rate 监控

每轮 QC 必须同时记录 Retention Rate：

```
run{N} keep = {数量}
run{N-1} keep = {数量}
Retention Rate = run{N} / run{N-1} = {%}
```

Retention Rate 下降超过 **30%**，必须人工复核原因，确认不是规则误杀导致后才能继续。

---

# Phase 4：决策

## Precision 决策

| Precision | 决策 |
|-----------|------|
| ≥ 70% | 满足约束，进入 FN Rate 判断 |
| < 70% | → Phase 5（加专属黑名单，减少 FP） |

## FN Rate 决策

| FN Rate | 决策 |
|---------|------|
| < 5% | 可接受，冻结交付 |
| 5% ~ 8% | 人工判断是否处理 |
| ≥ 8% | 必须进入 FN 分析，补充 entities.toml |

## Precision ≥ 70% 且 FN Rate 可接受时

冻结当前规则。进入 Done。

完成项目记录，产出交付文件到 `deliver/`。

## FN Rate ≥ 8% 时

→ 补充共享 `02_脚本/rules/entities.toml`（加缺失的体育实体/频道/同义词），降低 `no_signal` 误杀

→ 升共享规则版本号

→ 从 Phase 2 重新开始完整流程，project_log 记录新版本号

---

# Phase 5：构建数据集专属规则（FP 黑名单）

仅处理 FP（非体育被放行）。

规则来源仅允许：

* false_positive 样本
* false_negative 样本
* audit_sample
* QC 结果

禁止凭主观感觉新增规则。

**单轮新增规则数超过 20 条，必须人工审批。**

改动前先备份专属 blacklist.toml。

输出：

`data/runs/{sport}_{batch}/rules/blacklist.toml`

---

# Phase 6：重新过滤

脚本：

`clean_sports_v3.py`

规则：

shared rules（current/） + sport rules

输出：

`005_clean/run{N+1}/{原始文件名}_run{N+1}_keep.parquet`
`005_clean/run{N+1}/{原始文件名}_run{N+1}_drop.parquet`

---

# Phase 7：再次随机抽样 QC

抽样量规则同 Phase 3。

重新计算 Precision、FN Rate、Retention Rate → 回 Phase 4 决策

---

# Phase 8（可选）：DROP 召回

## 触发条件

人工决定执行即可触发。不再强制 FN Rate 阈值（召回的目标是最大化全体育数据量，不仅限于当前运动）。

## 执行流程

1. **过滤 DROP**：对 DROP 应用当前 pass2 规则，先移除已知 FP
2. **wide 召回**：用 `phase8_recall.py --mode wide` + 频道白名单正则，从 DROP 中筛选候选
   - 信号：`vs` / `final` / `championship` / `olympic` / `world cup` / `highlights` / `NBA` / `NFL` 等
   - 频道白名单：已知体育频道正则
   - 同时保留运动专属信号（如攀岩需 `climb|boulder|IFSC`）
   - 输出：`{原始文件名}_run{N}_recovered_wide.parquet`
3. **再次过滤 recovered**：对召回结果应用 pass2 规则（wide 模式会把游戏频道带回来）
4. **抽样 300 条**，跑 QC
5. **QC T rate ≥ 70%** → 落盘 recovered CSV，等指令合并
6. **QC T rate < 70%** → 分析剩余 FP → 加 pass2 规则 → 回到步骤 3

## 禁止事项

- 召回前不先过滤 DROP（会带回大量已知垃圾）
- 召回后不再次 pass2 过滤（游戏频道会通过 wide 信号逃逸）
- 未经 QC 直接合并或交付
- QC 不通过仍强行合并

---

# 停止条件

满足任意一个：

1. QC Pass Rate ≥ 70% 且 FN Rate < 8%
2. 已完成 3 轮**黑名单**迭代（Phase 5→6→7 循环）

注意：补充 entities.toml 处理 FN 不计入黑名单迭代轮次，但 entities 修改最多执行 2 次，超过后强制停止。

禁止无限优化规则。

---

# 交付

每批次达标后，输出到该批次的 `deliver/` 目录：

```
deliver/
├── {原始文件名}_run{N}_keep_final.parquet   ← 最终 KEEP（全部列）
├── {原始文件名}_run{N}_keep_final.csv       ← 最终 KEEP（全部列，CSV）
├── {原始文件名}_run{N}_keep_final_ids.csv   ← 仅 video_id
├── {原始文件名}_run{N}_recovered_wide.csv   ← 召回（如有，独立文件，不合并）
└── {原始文件名}_run{N}_recovered_wide.parquet
```

## 交付规则

- **keep 和 recall 独立交付，不自动合并。** 合并由人工跨批次完成
- **不自动复制到 data/交付/。** 等明确指令
- **QC 通过 ≠ 自动交付，必须等人确认**

最终跨批次合并由人工按 video_id 去重完成。

---

# 项目记录（强制）

每完成一个 Phase，必须立即写项目记录。

禁止积压记录。禁止事后补写。

记录位置：

`data/runs/{sport}_{batch}/project_log.md`

模板：

```
## 2026-06-03 15:30

### Phase 2 - 共享规则过滤

Shared Rules Version: v019
输入：001_baseline/{原始文件名}_raw.parquet
输出：005_clean/run01/

---

### Phase 3 - KEEP + DROP 抽样 QC

输入：
005_clean/run01/{原始文件名}_run01_keep.parquet（keep=23,404）
005_clean/run01/{原始文件名}_run01_drop.parquet

抽样量：300（keep < 50k）

结果：
QC Pass Rate: 57.0%
FN Rate: 12.3%
Retention Rate: — （首轮无对比）

结论：
Precision 未达 70%，FN Rate ≥ 8%

下一步：
1. Phase 5 分析 FP 样本加黑名单
2. 补充 entities.toml 处理 FN，升版本号
```

---

# FN 处理原则

当 DROP FN Rate ≥ 8% 时:

1. **禁止修改黑名单** — 黑名单控制 Precision，松了会引入污染
2. **优先补充共享信号到 `02_脚本/rules/entities.toml`**:
   - 缺失的体育实体词 → 加入 lexicon，附 @sport 注释
   - 缺失的同义词 → 加入 synonyms
   - 缺失的频道 → 加入 channel_whitelist
3. **修改后升共享规则版本号**
4. 目标: 降低 `no_signal` 与 `score_threshold` 导致的误杀
5. 验证: 重跑 Phase 2 + Phase 3，对比 Recall 变化

---

# 子规则备份与复用

冻结数据集时，将专属规则备份到 `rules_backup/`：

```bash
cp data/runs/{sport}_{batch}/rules/blacklist.toml \
   data/runs/{sport}_{batch}/rules_backup/blacklist.toml
```

同运动新批次进来，直接复用备份：

```bash
cp data/runs/{sport}_one/rules_backup/blacklist.toml \
   data/runs/{sport}_two/rules/blacklist.toml
```

噪声结构相似的同类型运动也可复用：

```bash
cp data/runs/archery_one/rules_backup/blacklist.toml \
   data/runs/shooting_one/rules/blacklist.toml
```

约束：

1. 复用后仍需跑完整的 Phase 2-3 QC 验证
2. Precision < 70% 则按标准流程迭代，不硬套
3. 复用的规则如有修改，必须写 rule_log

---

# Agent 执行规范

## 每次启动必读

1. 当前 `project_log.md` 最后一条记录（确认所在 Phase 和共享规则版本号）
2. 本次任务对应的 Phase 说明

## 禁止事项

- 禁止跨 Phase 连续执行（每个 Phase 完成后等待人工确认）
- 禁止在未读 project_log 的情况下执行任何过滤操作
- 禁止同时操作两个数据集目录
- 禁止修改其他批次目录下的任何文件
- 禁止在未备份的情况下修改任何规则文件
- 禁止单轮新增超过 20 条规则（必须人工审批）

## 多数据集并行

- 每个批次独立目录，互不引用中间产物
- 合并只在所有子集单独达标、交付到 `deliver/` 后由人工完成
- 合并前按 video_id 去重

---

# 附录：当前实现差异 (2026-06-06)

| 项目 | SOP 规范 | 当前实际 | 说明 |
|------|---------|---------|------|
| 共享规则目录 | `02_脚本/rules/current/` + `releases/` | 双层：管道读 `rules/`，版本管理在 `current/` | 修改规则后需 `\cp current/*.toml rules/` 同步 |
| baseline 文件 | `{原始文件名}_raw.parquet` | ✅ 已修复 | `phase0_normalize.py` |
| clean 输出 | `{原始文件名}_clean.parquet` | ✅ 已修复 | `core/cleaner.py` |
| run keep/drop | `{原始文件名}_run{N}_keep.parquet` | ✅ 已修复 | `core/cleaner.py` |
| QC 目录 | run 下的 keep_qc/drop_qc/ | ✅ 已对齐 | `phase2_qc.py` |
| 交付目录 | 每批次 `deliver/` | ✅ 已对齐 | — |
| 批次目录 | `data/runs/{sport}_{batch}/` | ✅ 已对齐 | — |
| **merge_rules bug** | 专属 pass2 合并到共享 | ✅ 已修复 (v4.2) — 改用 pattern 列表避免 `split("\|")` | 专属 pass2 直接写 `rules/blacklist.toml` |
| **pass2 vs r2** | 未区分 | pass2=硬过滤 r2=软减分 | `[[pass2]]` 直接移除，`[[r2]]` 只减 scoring |


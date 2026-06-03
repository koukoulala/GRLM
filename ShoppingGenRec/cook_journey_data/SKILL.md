---
name: shopping-journey-pipeline-compare
description: "Shopping推荐Pipeline对比分析。对比两条推荐链路（Copilot/SLM/Offline Ranker）的Journey和Product差异，逐用户深入分析，追踪L1-L4根因。Use when asked to compare shopping pipelines, analyze journey quality differences, or diff recommendation results."
argument-hint: "[data_directory containing pipeline files]"
---

# Shopping 推荐 Pipeline 对比分析

---

## 核心概念：对称对比 P1 vs P2

本技能用<strong>同一套维度</strong>对两端 pipeline 进行评估，不预设"基线 vs 待优化"的偏向：

- **P1 / P2** 只是命名标签，对应数据目录中的两份输入；本身不暗示孰优孰劣
- 所有维度（journey/product/seller/query）都用<strong>相同的 lens</strong>评估两端
- 报告呈现的是<strong>每个维度上两端各自的优势与不足</strong>，结论由数据决定

> **避免单边视角**。不要写成"P2 差距 + 反向优势"或"P1 差距 + 反向优势"——而是"在维度 X 上 P1 占优、维度 Y 上 P2 占优、维度 Z 互补"。
>
> **避免单边归因**。L1-L4 根因分析对两端都做。如果某差距是双向的（例如 P1 缺 X、P2 缺 Y），都要点出。

## 同一用户的两个 history 切片：联合视角

P1 和 P2 看的是同一个用户。如果 P1 用 conversation history、P2 用 browse history，那这两份 history 是<strong>同一用户的两个互补切片</strong>，不是两组不同的数据源。

因此 "P1/P2 journey 重叠率低" 通常不是 bug，而是反映了对话信号和浏览信号天然抓住了用户不同侧面的购物意图。两端联合后才是用户的完整画像。

正确的评估问题：
1. 对各自的 history 切片，pipeline 是否充分捕获了用户意图？
2. 给定同一用户的合并意图集合，每端在召回质量、seller authority、品牌覆盖、价格区间等维度上各自表现如何？
3. 能否用 P1 的强项 + P2 的强项混合召回？

## 语言规范

**所有分析输出（per-user 分析、综合总结、差距分析）统一使用中文撰写。** 涉及源数据时，列出英文原文并附中文含义，便于中文读者理解英文数据内容。

具体规则：

| 内容类型 | 格式 | 示例 |
|---------|------|------|
| Journey 标题 | 英文原文 + 中文翻译 | "A midi shirtdress moment"（优雅中长款衬衫裙） |
| 产品名称 | 英文原文 + 中文简述 | "Sasttie Wedge Pillow Set"（楔形枕头套装） |
| Seller / Brand | 英文原文（知名品牌无需翻译） | Nordstrom、Amazon.com |
| 用户历史事件 | 英文原文 + 中文概括 | "Browsed: Nike Dunk Low Retro White Black"（浏览了 Nike Dunk Low 复古黑白配色） |
| Query / TID | 英文原文 + 中文语义解读 | query: "L'Agence Kylo belted shirtdress women"（L'Agence 品牌腰带衬衫裙，女款） |
| reason / description | 英文原文 + 中文翻译 | reason: "You recently browsed multiple shirtdresses"（你最近浏览了多款衬衫裙） |
| 分析文字 | 纯中文 | 该用户在 P1 中 journey 覆盖面较广，P2 中 query 粒度更精细… |

> **原则：分析论述用中文，源数据保留英文原文+中文注释。** 这确保报告既可被中文读者流畅阅读，又能追溯到原始数据。

---

## 执行流程

### 输出目录约定（run-name）

**每次运行的所有产物隔离到独立的 run 子目录，避免覆盖历史结果。**

- 本地根目录：`<skill_data_root>/analysis/`
- 单次运行的产物全部写入：`<skill_data_root>/analysis/<run_name>/`
- 命名建议：`run_<YYYY-MM-DD>_<p1_name>_vs_<p2_name>`（如 `run_2026-05-08_copilot_vs_slm_v3`）

**远程输出目录（`--output-dir`）：**
- 所有脚本（`parse_pair.py` / `split_users.py` / `gen_per_user.py`）均支持 `--output-dir <path>` 参数
- 指定后，脚本会在本地 `analysis/<run_name>/` 写入的同时，将产物**镜像复制**到 `<output-dir>/<run_name>/`
- 典型用法：`--output-dir /cosmos/.../vip_case_study_IDB/`，使结果同时存在于本地工作区和远端数据目录
- Phase 3-5 的 `analyze_sellers.py`、`build_html_report.py` 和手写的 `.md` 文件需在两端目录都执行/复制

**Phase 0 执行前必须确定 `run_name`：**
1. 默认按上述命名规则构造（用当天日期 + 两端 pipeline 简称）
2. 如果用户显式提供了 run name，使用用户的名字
3. 如果同名 run 目录已存在，提示用户：覆盖 / 改名 / 中止

**所有后续 phase 的路径：**
- `paired_data.json` → `analysis/<run_name>/paired_data.json`
- 单用户 JSON（Phase 2 输入）→ `analysis/<run_name>/user_data/user_<id8>.json`
- per-user 分析 → `analysis/<run_name>/per_user/user_<id8>.md`
- 综合 / 差距报告 → `analysis/<run_name>/comprehensive_summary.md` / `gap_analysis.md`
- seller 统计 → `analysis/<run_name>/seller_analysis.json`

**给 Phase 2 / 3 / 4 子 agent 的提示词必须使用 `analysis/<run_name>/...` 的完整绝对路径**，不能写成裸 `analysis/...`，否则 agent 会写错位置。

### Phase 0：文件探测与格式识别

自动识别数据文件格式，避免解析试错。

1. 列出目标目录下所有数据文件
2. 读取每个文件前 2-3 行，判断格式（TSV / JSONL / JSON），采样提取字段结构
3. 与 [Schema Registry](references/schema-registry.md) 比对，确定具体 schema：
   - **Schema A**：Copilot Homepage TSV（含 `journeyProfiles`，产品通过 Bing Shopping API 实时搜索）
   - **Schema B**：SLM Query-based JSONL（`products[].query` 自然语言搜索词，ANN 检索）
   - **Schema C**：SLM TID-based JSONL（`products[].tid[]` 7-slot 结构化数组，目录匹配）
   - JSONL 文件需进一步区分 B/C：检查 `products[0]` 是否含 `query`（→B）还是 `tid`（→C）
4. **用户 ID 与映射表判断：**
   - 提取两端的用户 ID 字段和类型（Schema A 用 PicassoId，Schema B/C 用 StableId）
   - **需要映射表**：两端 ID 类型不同（如 A+B、A+C），需要一个含 `StableId ↔ PicassoId` 对应关系的 TSV 文件
   - **不需要映射表**：两端 ID 类型相同（如 B+C，都是 StableId），直接按 StableId 配对
   - 如需映射表但目录中未找到，提示用户提供
5. 输出探测摘要：`Pipeline1: [文件] | Schema [A/B/C] | ID类型` / `Pipeline2: ...` / `映射表: [文件] 或 不需要`

> **Windows 注意：** `csv.field_size_limit(2**30)`（不用 `sys.maxsize`）；`sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')`；文件统一 `encoding='utf-8'`

### Phase 1：数据解析与用户配对

1. 按 Phase 0 识别的 schema 解析两端数据
2. **用户配对：**
   - 两端 ID 类型相同（如都是 StableId）→ 直接按 ID 配对，无需映射表
   - 两端 ID 类型不同（如 PicassoId vs StableId）→ 使用映射表转换后配对
   - 只保留两端都有数据的用户（内连接）
3. 提取每用户的：历史信号、Journey 列表、产品详情
4. **预计算统计字段**（供 Phase 2 校验用，必须由 Python 脚本精确计算，禁止 LLM 估算）：
   - `p1.journey_count`: `len(p1.journeys)`
   - `p1.shopping_history_count`: `len(p1.shopping_history)`
   - `p2.journey_count`: `len(p2.journeys)`
   - `p2.recent_events_count`: 按换行符拆分 `p2.recent_events` 并过滤空行后的行数
   - `p2.profile_retailer_preferences`: 直接从 profile 提取的列表（或 `[]`）

   > **为什么需要预计算：** `recentShoppingEvents` 是一个含换行符的**单一长字符串**（非数组），LLM 在计数时极易出错（已发生过将 94 行误报为 0 行的案例）。由脚本预计算后写入 JSON，Phase 2 agent 直接引用即可。

5. **统一产品字段**（不同 schema 的产品结构不同，统一到 paired_data.json 中）：
   - Schema A 产品：保留 `name/seller/price`
   - Schema B 产品：保留 `query/matched_products`
   - Schema C 产品：保留 `tid/match_type/matched_products`
5. **用户分层（Triage）：**

| 分层 | 条件 | 处理 |
|------|------|------|
| **Deep** | 双端都有 journey（>0） | → Phase 2 深度分析 |
| **Skip** | 双端都 0 journey | 跳过，仅计入统计 |
| **Skip** | 单端有数据、另一端为空 | 跳过，仅计入覆盖率统计 |

6. 输出 `analysis/<run_name>/paired_data.json`（含 `p1_schema`/`p2_schema` 标记 + 预计算统计字段）+ 分层摘要

   **运行命令示例：**
   ```bash
   py analysis/parse_pair.py --run-name <run_name> [--data-dir <dir>] \
     [--p1-file <path>] [--p2-file <path>] [--map-file <path>] \
     [--output-dir <remote_data_dir>]
   py analysis/split_users.py --run-name <run_name> [--output-dir <remote_data_dir>]
   ```
   `split_users.py` 会从 `analysis/<run_name>/paired_data.json` 拆出每个 Deep 用户的独立 JSON 到 `analysis/<run_name>/user_data/`，供 Phase 2 各 agent 读取。

> **实现要求：** 预计算统计字段（`journey_count`、`recent_events_count` 等）**必须在 Phase 1 的 Python 数据解析脚本中计算并写入 JSON**，不得由 LLM agent 在 Phase 2 中估算。

### Phase 2：逐用户深入对比

**仅对 Deep 用户执行。** 可并行启动多个 agent（每批 3-4 个用户）。

对每个用户生成 `analysis/<run_name>/per_user/user_{stableId前8位}.md`，章节结构见 [用户分析模板](references/user-template.md)。

**参考范例：** [example-per-user-analysis.md](references/examples/example-per-user-analysis.md)（用户 CE04DD86 的完整分析，展示了 7 个章节的预期深度和格式，包含数据校验、语义匹配矩阵、相似 Journey 深入对比、L1-L4 归因、Seller Authority L1→L3→L4 流转追踪）

**关键原则：**
- **对称对比视角**：每个维度都列出 P1 和 P2 各自的表现与差距，不预设哪一端是"待优化"的
- 分析用中文，源数据（journey/product/seller）列出英文原文 + 中文含义
- Journey 匹配必须用 agent 语义理解，**不用** Jaccard 等词汇匹配
- 差异归因参考 [分析框架](references/analysis-framework.md)
- **Schema-specific 深度分析**：分析重点是 **Journey 内容（title/description/reason）和产品本身（query 质量、product 相关性、seller/brand authority）**。Schema C 关注 `match_type` 分布和 `tid[5]` seller 退化。Schema B 的诊断字段（`stats.reasoning` / `ANNScore` / `IsFiltered` / `FilterReason` / `Rank`）若数据中存在则可作为辅助证据，**若为空或缺失直接跳过，不作为必须项**
- **证据链格式**：每个差异点需提供 L1→L3→L4 的具体数据流转证据（如：用户浏览 X seller → TID 生成 Y seller → 匹配到 Z seller）
- **浏览记录计数规则**：`recentShoppingEvents` 是含换行符的**单一字符串**（非数组）。**禁止**自行计算行数。必须直接引用 `p2.recent_events_count` 预计算值。如该字段缺失，使用 Python 脚本 `len([l for l in s.split('\n') if l.strip()])` 精确计算，不得目测估算。

### Phase 3：综合总结

1. 运行 seller 分析脚本：`py <skill_dir>/scripts/analyze_sellers.py analysis/<run_name>/paired_data.json analysis/<run_name>/`
   - 使用 7-档 tier 分类：`luxury / department / specialty / brand_dtc / mass / marketplace / other`
   - **重要**：不要把垂直专家（Albee Baby / Clive Coffee / Williams Sonoma / B&H / CDW / Sephora / Hobby Lobby 等）归为 "other"。`specialty` 是 P2 ANN 索引常见的优势区，必须独立计量；`brand_dtc`（Nike.com / Apple / Anker / Hoka）同理。错误归类会得到 "other 占比暴涨 = tier 退化" 的伪结论
   - 脚本输出 `seller_analysis.json`，含 `tier_taxonomy` / `p1_tiers` / `p2_tiers` / `p1_tier_sellers` / `p2_tier_sellers` / `p1_unclassified_top` / `p2_unclassified_top`
   - 检查 `*_unclassified_top` — 如果有大量未分类的 seller 出现 ≥5 召回，扩展 `analyze_sellers.py` 的 LUXURY/DEPARTMENT/SPECIALTY/BRAND_DTC/MASS/MARKETPLACE 集合
2. **数据一致性交叉校验（必须先于综合分析）：**
   对每个 Deep 用户，从 `analysis/<run_name>/paired_data.json` 读取预计算字段（`p2.recent_events_count`、`p2.journey_count`），与对应 per-user .md 中"零、数据校验"报告的数值比对。如发现不一致（如 per-user 报告 0 条浏览记录但 paired_data 显示 94 条），**标记该用户分析为不可靠**，在综合总结中注明，并以 paired_data.json 的预计算值为准。
3. 生成 `analysis/<run_name>/comprehensive_summary.md`：

   **参考范例：** [example-comprehensive-summary.md](references/examples/example-comprehensive-summary.md)（12 用户的综合总结，展示了用户统计表、Journey 匹配统计、L1-L4 归因综合、Seller Authority 综合、P0/P1/P2 问题模式等章节的预期内容）
   - 用户分层统计表（Deep / Skip-双空 / Skip-仅P1 / Skip-仅P2）
   - Journey 语义匹配统计（HIGH / MEDIUM / NO MATCH 分布）
   - L1-L4 差异归因综合分析
   - Seller Authority 综合分析（引用 `seller_analysis.json` 的 7 档分布；指出 specialty/brand_dtc 是否上升、marketplace 是否缺失）
   - 共性问题模式（P0/P1/P2 优先级）+ 改进建议

### Phase 4：差距分析报告

**重要：Phase 4 agent 必须直接读取 `analysis/<run_name>/per_user/` 目录下的所有用户分析文件，而非接收 orchestrator 的摘要。** 这确保 agent 能从原始分析中提取具体证据和数据。

> **交叉校验要求：** Phase 4 agent 在引用任何 per-user 分析的定量数据（浏览记录数、journey 数、匹配数）时，必须与 `analysis/<run_name>/paired_data.json` 中的预计算值交叉验证。如发现偏差，以 paired_data.json 的预计算值为准，并在报告中标注 "[已修正：per-user 报告 X，实际为 Y]"。

生成 `analysis/<run_name>/gap_analysis.md`：

**参考范例：** [example-gap-analysis.md](references/examples/example-gap-analysis.md)（SLM vs Copilot 差距分析，展示了按 L1→L4 链路排序的 7 个差距、每个差距的定量摘要/表现/根因/反向优势结构、Seller Authority L1→L3→L4 流转图、核心归因判断、优化建议）
- **对称对比视角**：每个差距章节描述"在该维度上 P1 vs P2 各自表现"，不预设"基线 vs 待优化"
- 按 L1→L2→L3→L4→跨层 组织差距，定量指标参考 [分析框架](references/analysis-framework.md)
- 每个差距标注根因层级：**表现→根因→双向反向优势**
- **双向反向优势必须包含**：每个差距章节末尾列出 P1 和 P2 各自在该维度上的优势案例（避免"只有一端有缺点"的单边叙事）
- **每个差距必须包含定量摘要**，格式：

```markdown
## 差距N：[名称] ｜ 根因层级：LX
> **P1 vs P2 表现差异：[定量指标] ｜ 影响用户：M/N（X%） ｜ 优先级：PX**
```

- Seller Authority 追踪 L1→L3→L4 seller 流转
- 末尾附优先级排序表 + 优化建议

**子问题拆解要求：** 当某差距影响 >50% 用户时，必须将该差距拆解为具体子问题（如"TID 质量"拆解为品牌虚构/语义漂移/品牌遗漏/seller 退化），每个子问题独立列出影响用户数、案例表格和根因分析。

**核心归因判断（必须包含）：** 报告末尾必须有一个"核心归因判断"章节，综合所有差距，判断 **每端 pipeline 主要短板源自 pipeline 的哪一层**（如"P2 的 L4 是主要瓶颈，因为 L3 query 通常正确；P1 的 L1 信号源依赖对话历史导致实时意图缺失"），并提供跨差距的交叉证据。判断要对两端对称。

### Phase 5：HTML 可视化报告（最终交付物）

把所有 markdown 产物 + 数据汇总成一份单文件 HTML dashboard，作为<strong>给非技术 stakeholder 看的最终交付物</strong>。

**步骤：**
1. 准备 `analysis/<run_name>/dimensions.json` —— 由 orchestrator 编写（**不要给子 agent 做**，因为这是综合判断），包含：
   - `executive_summary`：开篇执行摘要（HTML 段落），分别点评 P1 和 P2 的核心特点 + 短板，并给出"互补 vs 替代"的判断
   - `journey_axes[]`：Journey 4 维评估表（quality / type diversity / coverage diversity / relevance）
   - `product_axes[]`：Product 4 维评估表（quality / seller authority 按场景细分 / diversity / relevance）
   - `dimensions[]`：完整 N 维评分表（每维含 P1/P2/winner/evidence）
   - `scenarios[]`：场景推荐表
   - `p2_advantages[]` / `p1_advantages[]`：各 4 张 KPI 卡片
   - schema 见 [build_html_report.py](scripts/build_html_report.py) 顶部 docstring
2. 运行：`py <skill_dir>/scripts/build_html_report.py analysis/<run_name>/`
3. 产出 `analysis/<run_name>/report.html`（单文件，含 7 个 tab）

**HTML 7 个 tab 结构：**
| Tab | 内容 |
|-----|------|
| **概览** | 核心 KPI + Tier 横向条形图 + 用户卡片网格（点击跳转） |
| **⭐ 最终结论** | 执行摘要 → Journey 维度 → Product 维度 → 总评分 → N 维数据 → 关键数字 → P1/P2 优势 → 场景推荐 |
| **差距分析** | 嵌入 `gap_analysis.md` |
| **综合总结** | 嵌入 `comprehensive_summary.md` |
| **逐用户深入分析** | 14 个 per_user/*.md 切换 |
| **Seller / 价格分析** | 7 档 tier 表 + Top sellers + 价格分布 |
| **📋 评估方法** | history-union 视角 + 7 档分类说明 + 评分规则 + per-user 7 章结构 |

**最终结论 tab 的设计原则（重要）：**
- 这是给业务方看的，不是给开发者看的
- **必须有开篇执行摘要**：直接点评两端 pipeline，至少覆盖 (1) 各自核心特点 + 短板、(2) 是否互补、(3) 一句话建议
- **从 Journey 和 Product 两个维度评估**：Journey 4 个子维度 + Product 4-5 个子维度（按 seller authority 场景再细分）
- **数据 + 证据**：每个判断都要有定量证据（pp 差距、影响用户数、具体案例）；少 thinking，多 facts
- 方法论解释挪到"📋 评估方法" tab，不要污染最终结论

---

## 质量检查点

- [ ] `run_name` 已确定，输出目录 `analysis/<run_name>/` 已创建
- [ ] Phase 0 探测结果已输出，schema 已确认
- [ ] 用户分层摘要已输出（Deep / Skip 各类数量）
- [ ] `analysis/<run_name>/paired_data.json` 包含预计算统计字段（`journey_count`、`recent_events_count`）
- [ ] 所有 Deep 用户都有独立分析文件（`analysis/<run_name>/per_user/`）
- [ ] 每个 per-user 分析的"零、数据校验"与 paired_data.json 预计算值一致
- [ ] Journey 匹配使用语义理解
- [ ] 每个差异点都有 L1-L4 归因
- [ ] Seller Authority 追踪了 L1→L3→L4 完整链路
- [ ] 每个差距都有定量摘要（影响用户数/占比）
- [ ] Phase 3/4 已对 per-user 定量数据做交叉校验
- [ ] analyze_sellers.py 已执行并生成 `analysis/<run_name>/seller_analysis.json`（7-档 tier，含 specialty / brand_dtc / marketplace 独立计量）
- [ ] `seller_analysis.json` 中 `*_unclassified_top` 已 review，高频未分类 seller（≥5 召回）已扩展到 `analyze_sellers.py` 的 tier 集合中
- [ ] orchestrator 已写 `analysis/<run_name>/dimensions.json`（含 executive_summary / journey_axes / product_axes / dimensions / scenarios / p2_advantages / p1_advantages）
- [ ] `build_html_report.py` 已执行并生成 `analysis/<run_name>/report.html`
- [ ] HTML 最终结论 tab 包含开篇执行摘要 + Journey 4 维 + Product 4-5 维评估
- [ ] 所有给子 agent 的 prompt 都使用 `analysis/<run_name>/...` 完整路径，未出现裸 `analysis/...`

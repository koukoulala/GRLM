# 分析框架

> **分析视角：对称对比。** P1 和 P2 用<strong>同一套维度</strong>评估；每个维度都列出两端各自表现，由数据决定胜负，不预设"基线 vs 待优化"。
>
> **语言规范：** 分析论述用中文。引用源数据时保留英文原文并附中文含义（如 journey title、product name、query、reason、seller name）。

## 同一用户的两个 history 切片：联合视角（核心前提）

P1 和 P2 看的是<strong>同一个用户</strong>。如果 P1 用 conversation history（对话主动表达），P2 用 browse history（浏览行为），那这两份 history 是<strong>同一用户的两个互补切片</strong>，不是两组不同的数据源。

因此 "P1/P2 journey 重叠率低" 通常不是 bug，而是反映了对话信号和浏览信号天然抓住了用户不同侧面的购物意图。<strong>两端联合后才是用户的完整画像</strong>。

正确的评估问题：

1. 对各自的 history 切片，pipeline 是否充分捕获了用户意图？
2. 给定同一用户的合并意图集合，哪一端的<strong>召回质量</strong>（query 准确性、seller authority、长尾品牌覆盖、价格区间）更优？
3. 能否用 P2 的强项 + P1 的强项混合召回？

> **不要把"两端 journey 不重叠"直接归为某一端的问题**。在归因前先问：是 input 信号源天然不交叉，还是 pipeline 算法出错？前者属于 L1 数据源差异（属客观特征），后者才是真问题。归因要对两端对称——P1 和 P2 各自在哪些层级有不足，分开点出。

## Journey 维度评估（4 个子维度）

| 子维度 | 评估方法 | 典型问题 |
|--------|----------|----------|
| **Quality 主题准确性** | journey title/description/reason 是否准确反映用户意图，是否过度泛化 | 单点信号被扩成"全家桶"；主题命名宽泛 |
| **Type Diversity 类型多样性** | 覆盖几类 journey type（explicit / related / trending / all_in_one） | 缺 trending（趋势）/ all_in_one（一站式 capsule） |
| **Coverage Diversity 品类覆盖** | 跨品类宽度，与用户实际意图（联合 history）的吻合度 | conversation-only 和 browse-only 各失一半 |
| **Relevance 用户意图吻合** | 主动表达意图 vs 实时行为意图的捕获率 | 对话信号衰减失时效；profile 字段瘦身致长期偏好缺失 |

## Product 维度评估（4-5 个子维度）

| 子维度 | 评估方法 | 典型问题 |
|--------|----------|----------|
| **Quality 产品-query 匹配** | 召回产品的品类/品牌/性别/属性是否与 query 一致；长尾品牌覆盖率 | 长尾品牌 ANN 索引 0 命中；ANNScore 高分但语义漂移 |
| **Seller Authority — 垂直专业** | specialty tier 占比（母婴/咖啡/相机/B2B 等） | 错误归到 mass / other |
| **Seller Authority — 探索/百货** | department + marketplace tier 占比 | 跨百货比价能力丢失；marketplace（Etsy）完全无召回 |
| **Diversity seller/brand 多样性** | top-k unique sellers，consec_seller 分布 | top-k 单 seller monoculture；consec_seller_after 居高 |
| **Relevance 价格/偏好执行** | 价格区间与用户预算吻合度；retailerPreferences 在 L4 的执行率 | 用户偏好 Apple 但 L4 仅极少召回 Apple Store |

## 根因层级（L1-L4）

```
L1 用户历史      L2 Journey 生成     L3 TermID/Query 生成   L4 产品匹配
────────────    ────────────────    ──────────────────    ──────────────
用户行为数据 →   抽象出 Journey   →   生成搜索词/TID    →   匹配具体产品
(浏览/对话/      (主题/类型/          (品牌/seller/          (Title/Seller/
 Profile)         reason)              属性等)               Brand/Price)
```

| 层级 | 典型问题 |
|------|---------|
| L1 | 两端数据源不同导致信号差异（对话 vs 浏览） |
| L2 | Journey 品类泛化、过度扩展或遗漏、类型缺失 |
| L3 | 品牌虚构、语义漂移、品牌遗漏、seller 降级、字段退化 |
| L4 | 产品目录覆盖不足、价格/品牌层级偏差、IndexReRanker 缺 retailer/brand boost |

## Seller Tier 分类（7 档）

> **重要：** 不要使用旧的 4 档分类（luxury/premium/mass/other），否则会把垂直专家与品牌官店错误归入 "other"，得出"P2 'other' 占比上升 = tier 退化"的伪结论。

| Tier | 定义 | 代表 seller |
|------|------|------------|
| **luxury** | 奢华百货 / 奢侈品 | Bloomingdale's / Saks / Neiman Marcus / FARFETCH / Bergdorf Goodman |
| **department** | 主流高端百货 | Macy's / Nordstrom / Nordstrom Rack / Dillard's / Anthropologie |
| **specialty** | 垂直专家零售商（按品类专业服务）| REI（户外）/ Williams Sonoma（厨房）/ B&H Photo（影像）/ Albee Baby（母婴）/ Clive Coffee（咖啡）/ CDW（B2B IT）/ Sephora（美妆）/ Chewy（宠物）/ Home Depot（家装）/ Hobby Lobby（手工）|
| **brand_dtc** | 品牌官方 DTC | Nike.com / Apple / Anker / The North Face / Hoka / Ann Taylor / Quince |
| **mass** | 大众综合零售 | Amazon / Walmart / Target / Kohl's / Wayfair / JCPenney |
| **marketplace** | P2P / UGC / 长尾聚合 | Etsy / eBay / Poshmark |
| **other** | 未匹配规则的零散站点 | （review `*_unclassified_top` 决定是否要扩展 tier 集合）|

**对比分析时注意：**
- `specialty` 上升（如 +16pp）通常是 P2 ANN 索引的<strong>正向迁移</strong>，不是退化
- `brand_dtc` 上升（如 +7pp）通常是 per-brand query 拆分的优势
- `mass` 下降可以是中性偏正向（Amazon 依赖减少）
- `marketplace` 完全为 0 是真短板（手作/复古/长尾品类失声）
- `department` 下降是真短板（跨百货比价能力丢失）
- `other` 占比应保持在 15-20% 以下；如果 ≥30%，说明 tier 集合需要扩展

## L3 子问题拆解指南

当 L3 差距影响 >50% 用户时，**必须**拆解为以下子问题，每个独立分析：

| 子问题 | 定义 | 检测方法 | 严重度 |
|--------|------|----------|--------|
| **品牌虚构（Brand Hallucination）** | TID/query 中出现不存在的品牌 | 品牌名无法在产品目录/搜索引擎中找到 | P0 |
| **语义漂移（Semantic Drift）** | 生成的搜索词/TID 偏离用户原始意图 | 对比用户搜索词 vs TID/query 语义 | P1 |
| **品牌遗漏（Brand Omission）** | 用户明确搜索/浏览的品牌未出现在 TID/query 中 | 对比用户历史品牌 vs TID[4]/query 品牌 | P1 |
| **Seller 字段退化（Schema C only）** | TID[5] 被非 seller 值占据（颜色、型号等）| 检查 TID[5] 是否为有效零售商名称 | P0 |

每个子问题需要独立的：影响用户数/占比、案例表格（用户/Journey/具体问题/匹配率）、根因分析。

## 差距维度

1. **意图理解深度**（L1）— 预算/否定信号/场景/品牌忠诚/用户专业度
2. **Journey 覆盖面**（L1+L2）— 数量、类型多样性、品类重叠度
3. **TID/Query 生成质量**（L3）— 品牌虚构、语义漂移、品牌遗漏、seller 字段退化
4. **Seller Authority**（L3）— seller 档次降级，追踪 L1→L3→L4 流转
5. **产品目录覆盖**（L4）— 小众/专业品类匹配率
6. **价格定位**（L3→L4 连锁）— seller 降级驱动的价格下移
7. **冷启动与过度扩展**（跨层）— 稀疏数据下的行为差异

## 差距分析在不同 Schema 组合下的适配

差距三（L3）和差距四（Seller Authority）的分析方法取决于 Pipeline 使用的 Schema：

| 分析维度 | Schema A (Copilot) | Schema B (Query-based) | Schema C (TID-based) |
|---------|-------------------|----------------------|---------------------|
| **L3 品牌分析** | 检查 queries[] 中的品牌是否被保留 | 检查 query 中的品牌关键词是否与用户历史一致 | 检查 tid[4] 品牌是否真实（虚构检测）、是否与用户历史一致（遗漏检测） |
| **L3 语义分析** | 对比 queries 语义与用户对话意图 | 对比 query 语义与用户浏览意图 | 对比 tid 各 slot 语义与用户浏览意图（漂移检测） |
| **L3 Seller 追踪** | queries 中通常不含 seller，seller 由 Bing API 决定 | query 中可能隐含 seller（品牌/retailer 关键词） | tid[5] 为 seller slot，可直接追踪 L1→L3 seller 流转 |
| **L3 字段退化** | 不适用 | 不适用 | 检查 tid[5] 是否被非 seller 值占据 |
| **L4 产品质量** | 按产品列表评估 | 按召回产品的相关性、品牌、seller、价格人工判定 | 按 match_type (exact/fuzzy/none) 统计匹配率 |
| **Seller Authority L4** | 直接从 products[].seller 提取 | 从 matched_products[].Seller 提取 | 从 matched_products[].Seller 提取，与 tid[5] 对比 |

**Schema B 的 Seller Authority 分析特别说明：**
- Query-based 模式下 seller 不在 L3 结构化指定，而是由 ANN 检索 + 排序/过滤决定
- 重点对比 `matched_products[].Seller` 与用户 `profile.retailerPreferences` / 浏览历史中的 seller 的一致性
- 关注是否大量召回了用户从未浏览过的灰渠道（如 eBay/Poshmark/海外站点）

## 差距定量指标参考

| 差距 | 定量指标 |
|------|---------|
| 意图理解深度（L1） | 影响用户占比、各意图维度受影响用户数 |
| Journey 覆盖面（L1+L2） | P2<P1 的用户占比、journey 总量比、零重叠用户占比 |
| TID/Query 生成质量（L3） | Schema C: 品牌虚构/语义漂移/品牌遗漏/seller退化 各自影响用户数；Schema B: query 品牌保真率、语义偏移用户数 |
| Seller Authority（L3） | 影响用户占比、三种降级模式各自影响用户数 |
| 产品目录覆盖（L4） | Schema C: 匹配率<50%品类数、累计 none TID 数；Schema B: 召回与 query 不相关的产品占比、长尾 SKU 0 召回案例数 |
| 价格定位（L3→L4） | 价格偏离>N×用户数（N 为参考阈值，默认 5×，可根据数据调整）、中位价差距百分比 |
| 冷启动/过度扩展（跨层） | 冷启动用户数（浏览<10条）、过度扩展用户数（1信号→5+journey） |

## Schema B 分析重点

**Schema B 的核心分析对象是 Journey 内容和产品本身**，不依赖任何特殊诊断字段。

| 维度 | 位置 | 分析用途 |
|------|------|---------|
| Journey title / description / reason | `journeys[].title/description/reason` | 判断 journey 主题是否准确反映用户意图、覆盖范围、是否过度扩展或缺漏 |
| 产品 query | `journeys[].products[].query` | 判断 query 生成是否准确（品牌保真、语义不漂移、限定词不丢失） |
| 召回产品 | `journeys[].products[].matched_products[]` | 判断召回产品是否与 query 匹配（品类/品牌/性别/属性），seller/brand 权威度，价格区间 |

> **可选辅助字段**：以下 Schema B 字段若数据中存在且非空，可作为补充证据；若全空则注明并跳过，不影响主分析：
> - `stats.reasoning` / `stats.totalCandidates` / `stats.selectedCount` / `stats.filteredCount`
> - `matched_products[].ANNScore` / `IsFiltered` / `FilterReason` / `Rank`

**Schema B Seller Authority 分析路径：**
```
L1: profile.retailerPreferences + recentShoppingEvents 中的 seller
L3: query 中是否包含 seller / 品牌关键词
L4: matched_products[].Seller 实际分布
差距判断: L4 seller 分布是否偏离 L1 偏好？是否大量召回用户从未浏览的灰渠道？
```

## 核心归因判断（gap_analysis.md 必须包含）

报告末尾必须有"核心归因判断"章节，<strong>对两端对称回答</strong>：每端 pipeline 的主要短板源自哪一层？

判断方法：
1. 分别统计 P1 和 P2 各自被归因的差距数量与影响用户数（按 L1/L2/L3/L4/跨层）
2. 检查 L4 是否"忠实执行"L3 的输出（如果是，则 L4 不是主要问题）
3. 提供跨差距的交叉证据（如：seller 降级、品牌虚构、语义漂移是否都源于同一层）
4. 得出对称结论，例如：
   - P2：L4 是主要瓶颈（ANN 索引覆盖不足；L3 query 通常正确但 L4 召回背叛 L3）
   - P1：L1 信号源依赖对话历史（无法捕捉实时浏览意图，trending 类别也受限）

## 双向反向优势要求

每个差距章节末尾必须包含<strong>双向反向优势</strong>小节——既列 P1 在该维度上的优势案例，也列 P2 的优势案例，确保读者不会得出"只有一端有缺点"的单边结论。格式：

```markdown
### 反向优势
**P1 在此维度的优势：**
| 优势场景 | 具体案例 | 用户 |
|---------|---------|------|
| [场景] | [具体表现] | [用户ID] |

**P2 在此维度的优势：**
| 优势场景 | 具体案例 | 用户 |
|---------|---------|------|
| [场景] | [具体表现] | [用户ID] |
```

如果某维度某端确实没有优势可写，明确标注"无显著优势"，不要为了对称而虚构。

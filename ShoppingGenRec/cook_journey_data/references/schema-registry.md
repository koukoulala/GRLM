# Schema Registry

Phase 0 探测文件格式时，与此处已知 schema 比对，匹配则使用对应解析逻辑。

---

## Schema A：Copilot Homepage (TSV)

**识别特征:** TSV 格式 + `response` 列含 `journeyProfiles`

```
列: user_id, version, turns_count, conversation_history, response

response 解析:
  用户画像:  response.userShoppingProfile
  Journey:  response.journeyProfiles[]
    .title .description .journeyType .confidenceScore .reason .queries[]
  产品:     response.journeyProfiles[].bingShoppingApiProducts[]
    .name .seller .price .imageUrl（通常无 brand）
  用户历史:  conversation_history (JSON string)
    .ShoppingConversationHistory[] / .OtherConversationHistory[]

Journey类型: explicit, related, trending, all_in_one
用户ID: PicassoId（需映射表转 StableId）
产品检索方式: Bing Shopping API 实时搜索（基于 .queries[] 自然语言）
```

---

## Schema B：SLM Query-based Journey (JSONL)

**识别特征:** JSONL + 含 `stableid` + `journeys` + `products[].query`（无 `tid`）

```
顶层: stableid, journeys[], userShoppingProfile{}, recentShoppingEvents

用户画像:  userShoppingProfile (或 userShoppingProfile.userShoppingProfile)
  .shoppingGenderPreference
  .categoryPreferences[]
  .brandPreferences[]
  .retailerPreferences[]        ← Seller Authority 分析必需
  .priceSensitivity
  .fashionStyle[] .fashionFit[] .shoppingValues[]
  .contextualShoppingInterests[]
  .suggestedRelatedBrands[]

用户浏览历史:  recentShoppingEvents (string, 多行文本)
  格式: "序号 | 时间 | 行为(Searched/Browsed) | 产品标题/搜索词 [| Seller]"
  Seller 提取: 从产品标题末尾解析（如 "DKNY Women's Midi Dress - Macy's" → Macy's）

Journey:  journeys[]
  .title .journeyType .description .reason .conversationStarter .stats{}

Journey stats:  journeys[].stats（可选辅助字段，可能为空）
  .totalCandidates .selectedCount .filteredCount .reasoning

产品:     journeys[].products[]
  .query              ← 自然语言搜索词（如 "L'Agence Kylo belted shirtdress women"）
  .matched_products[]
    .Title .Seller .Brand .OriginalPrice .global_offer_id .Description
    .CategoryName .Gender .AgeGroup .ImageUrl .OfferUrl
    （可选辅助字段，可能为空）.Rank .ANNScore .IsFiltered .FilterReason

Journey类型: explicit, related
用户ID: StableId
产品检索方式: 自然语言 query → ANN 向量检索 → 排序/过滤
```

**与 Schema C（TID-based）的关键区别：**
- 使用 `.query`（自然语言）而非 `.tid[]`（结构化数组）检索产品
- 无 `.match_type` 字段（exact/fuzzy/none），产品质量通过**召回产品与 query 的相关性人工判定**
- Journey 有 `.description` 和 `.conversationStarter`（Schema C 无）
- matched_products 有 `.Description`（产品描述，Schema C 无）

**Schema B 核心分析维度（Phase 2/4）：**
- **Journey 内容**：title/description/reason 是否准确反映用户意图、覆盖范围
- **Query 质量**：是否准确（品牌保真、语义不漂移、限定词不丢失、性别/品类约束）
- **召回产品**：与 query 的相关性、品牌真实性、seller/brand 权威度、价格区间
- **Seller Authority**：`matched_products[].Seller` 是否符合用户 `retailerPreferences` 和浏览历史

> **可选辅助字段（仅在数据中存在且非空时使用）**：`stats.reasoning` / `stats.totalCandidates` / `stats.selectedCount` / `stats.filteredCount` / `matched_products[].ANNScore` / `IsFiltered` / `FilterReason` / `Rank`。若全空则注明"该字段在本数据中未提供"后跳过，不影响主分析。

---

## Schema C：SLM TID-based Journey (JSONL)

**识别特征:** JSONL + 含 `stableid` + `journeys` + `products[].tid`（无 `query`）

```
顶层: stableid, journeys[], userShoppingProfile{}, recentShoppingEvents

用户画像:  与 Schema B 相同

用户浏览历史:  与 Schema B 相同

Journey:  journeys[]
  .title .journeyType .reason .stats{}
  （无 .description、无 .conversationStarter）

产品:     journeys[].products[]
  .tid[]              ← 7-slot 结构化数组
  .match_type         ← "exact" / "fuzzy" / "none"
  .matched_products[]
    .Title .Seller .Brand .OriginalPrice .global_offer_id
    .CategoryName .Gender .AgeGroup .ImageUrl .OfferUrl
    （可选辅助字段，可能为空）.Rank .ANNScore .IsFiltered .FilterReason
    （无 .Description）

TID 结构（7 个 slot）:
  idx 0: 品类 (e.g., "blouse", "carrier", "pants")
  idx 1: 属性1 (e.g., "floral", "hands free", "bootcut")
  idx 2: 属性2 (e.g., "ruffle trimmed", "wrap", "mid rise")
  idx 3: 属性3 (e.g., "silk", "lightweight", "stretch")
  idx 4: 品牌 (e.g., "PAIGE", "BABYBJÖRN") ← 品牌虚构分析用
  idx 5: Seller (e.g., "Nordstrom Rack", "Amazon.com") ← Seller Authority 分析用
  idx 6: 附加属性 (e.g., "bell sleeve", "women", "infant")
  注意: 小众品类中 idx 5 可能退化为产品属性值（如 "aqua", "CH563WN#140"）

Journey类型: explicit, related
用户ID: StableId
产品检索方式: TID 结构化属性 → 产品目录精确/模糊匹配
```

---

## Schema 识别逻辑

```
输入文件 → 判断格式
  │
  ├─ TSV + response 列含 journeyProfiles
  │   → Schema A (Copilot Homepage)
  │
  ├─ JSONL + stableid + journeys
  │   │
  │   ├─ products[0] 含 "query" 字段（无 "tid"）
  │   │   → Schema B (SLM Query-based)
  │   │
  │   ├─ products[0] 含 "tid" 字段（无 "query"）
  │   │   → Schema C (SLM TID-based)
  │   │
  │   └─ 同时含 "query" 和 "tid"
  │       → 优先按 query 处理，tid 作为辅助
  │
  └─ 均不匹配
      → 输出字段结构，agent 自行推断，标注"未知 schema"
```

**对比分析时的 Schema 差异处理：**

| 分析维度 | Schema A (Copilot) | Schema B (Query) | Schema C (TID) |
|---------|-------------------|-----------------|---------------|
| Journey 内容对比 | .queries[] | .query | .tid[] → 需还原为语义描述 |
| 产品质量评估 | 无评分，按产品列表顺序 | 召回产品与 query 的相关性人工判定 | .match_type (exact/fuzzy/none) |
| Seller 追踪 L3 | .queries 中隐含 seller | .query 中可能含 seller | .tid[5] 为 seller slot |
| Seller 追踪 L4 | .seller | .matched_products[].Seller | .matched_products[].Seller |
| 品牌虚构检测 | 不适用（query 不含品牌slot） | 检查 query 中品牌是否真实 | 检查 tid[4] 品牌是否真实 |
| seller 字段退化 | 不适用 | 不适用 | 检查 tid[5] 是否为非 seller 值 |
| Filtering 分析 | 无 | 可选：若 stats.reasoning / IsFiltered / FilterReason 字段非空可参考 | IsFiltered/FilterReason |

---

## paired_data.json 中间产物 Schema

Phase 1 输出的 `analysis/paired_data.json` 供 Phase 2 agent 和 `analyze_sellers.py` 消费。
需根据输入 schema 组合（A+B / A+C / B+C）适配产品字段：

```json
[
  {
    "stableid": "CE04DD86...",
    "picassoid": "fEXcQ9z...",
    "triage": "deep",
    "p1_schema": "A",
    "p2_schema": "C",
    "p1": {
      "journey_count": 15,
      "shopping_history_count": 30,
      "shopping_history": [],
      "other_history_count": 239,
      "profile": {},
      "journeys": [
        {
          "title": "...",
          "journeyType": "explicit",
          "reason": "...",
          "queries": ["..."],
          "products": [
            { "name": "...", "seller": "Nordstrom", "price": 144.0 }
          ]
        }
      ]
    },
    "p2": {
      "journey_count": 7,
      "recent_events_count": 84,
      "recent_events": "1 | 8 hours ago | Browsed | ...",
      "profile": {
        "retailerPreferences": ["Macy's", "Nordstrom"],
        "brandPreferences": ["BABYBJÖRN"],
        "priceSensitivity": "general"
      },
      "journeys": [
        {
          "title": "...",
          "journeyType": "explicit",
          "reason": "...",
          "products": [
            {
              "query": "L'Agence Kylo belted shirtdress women",
              "tid": ["blouse", "floral", "ruffle", "silk", "PAIGE", "Nordstrom Rack", "bell sleeve"],
              "match_type": "exact",
              "matched_products": [
                { "Title": "...", "Seller": "Nordstrom Rack", "Brand": "PAIGE", "OriginalPrice": "$59.97" }
              ]
            }
          ]
        }
      ]
    }
  }
]
```

**关键字段说明：**
- `p1.journey_count` / `p2.journey_count`: **预计算值**，由 Phase 1 脚本精确计算。Phase 2 agent 用于数据校验。
- `p2.recent_events_count`: **预计算值**，`recentShoppingEvents` 字符串按换行拆分后的非空行数。Phase 2 agent **必须直接引用此值**，禁止自行解析字符串计数（LLM 对长字符串计行极易出错）。
- `p1_schema` / `p2_schema`: 标记各端使用的 schema 类型（"A" / "B" / "C"），Phase 2 agent 据此决定分析逻辑
- `p2.products[].query`: Schema B 时存在，Schema C 时不存在
- `p2.products[].tid[]`: Schema C 时存在，Schema B 时不存在
- `p2.products[].match_type`: 仅 Schema C 存在
- 两个字段可能同时存在（如果同时含 query 和 tid），agent 按 schema 类型决定主分析路径

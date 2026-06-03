# SLM 与 Copilot Shopping Journey 差距分析报告

**分析范围：** 12 个双端用户（同时拥有 Copilot 和 SLM Journey 输出）
**分析日期：** 2026-04-23

---

## 核心结论

SLM 与 Copilot 在 Shopping Journey 生成上存在七个维度的系统性差距，但两者各有明显优势、互补性极强。

本报告按 **Pipeline 链路顺序（L1→L2→L3→L4→跨层）** 组织差距分析，从数据源层逐步追溯到产品匹配层，使根因链条清晰可见：

| 差距 | Pipeline 层级 | 修复优先级 | 影响用户 |
|------|-------------|-----------|---------|
| 差距一：意图理解深度不足 | L1 数据源层 | P2（结构性） | 12/12（100%） |
| 差距二：Journey 覆盖面窄 | L1+L2 | P1 | 10/12（83.3%）SLM < Copilot |
| 差距三：TermID 生成质量问题 | L3 | **P0+P1** | 8/12（66.7%） |
| 差距四：Seller Authority 降级 | L3 | **P0** | 10/12（83.3%） |
| 差距五：小众品类产品目录覆盖不足 | L4 | P1 | 6/12（50%） |
| 差距六：商品 Popularity 与价格定位 | L3→L4 连锁 | P2（连锁） | 3-6/12（25-50%） |
| 差距七：冷启动与过度扩展 | 跨层 | P2 | 3/12（25%）各端 |

---

## 根因层级框架（L1-L4）

本报告采用 L1-L4 四层归因框架，对应 SLM Pipeline 从用户数据到最终推荐产品的完整链路。每个差距均追溯到具体层级，以定位问题的真正来源。

```
L1 用户历史      L2 Journey 生成     L3 TermID 生成      L4 产品匹配
────────────    ────────────────    ───────────────    ──────────────
浏览记录         从用户信号中        为每个 Journey      用 TID 去产品
搜索事件    →    抽象出 Journey  →   生成结构化的    →   目录中检索
Shopping         主题、类型、         TermID（品牌/       匹配产品
Profile          reason              seller/属性等）
```

| 层级 | 名称 | 输入 | 输出 | 典型问题 |
|------|------|------|------|---------|
| **L1** | 用户历史差异 | 用户的浏览/搜索/对话行为 | 可用于推荐的购物信号集合 | 两端采集到的信号本身不同（如 Copilot 有 20 组对话，SLM 仅 2 条浏览） |
| **L2** | Journey 生成差异 | L1 的购物信号 + Profile | Journey 列表（标题、类型、reason） | 品类泛化、journey 过度扩展或遗漏、trending/related 类型缺失 |
| **L3** | TermID 生成差异 | Journey 主题 + 用户偏好 | 结构化 TID（7 个 slot：品类/属性×3/品牌/Seller/附加属性） | 品牌虚构、语义漂移、品牌遗漏、seller 字段退化、seller 系统性降级 |
| **L4** | 产品匹配差异 | TID | 匹配到的具体产品（Title/Seller/Brand/Price） | 产品目录覆盖不足（小众品类 none 率高）、价格/品牌层级偏差 |

**关键洞察：** 本分析发现大部分差距的根因集中在 **L3（TermID 生成）**，而非 L4（产品匹配）。L4 对 L3 给出的 TID 通常忠实执行——问题在于 L3 给出的 TID 本身质量不足（品牌虚构、seller 降级、语义漂移等）。修复应优先聚焦 L3。

---

## 差距一：意图理解深度不足 ｜ 根因层级：L1（数据源层）

> **影响用户：12/12（100%） ｜ 优先级：P2**
>
> 这是最核心的结构性差距，但属于 P2 优先级——因为它源于数据源性质差异，无法通过优化 SLM 模型单独解决。

### 表现

SLM 仅能从浏览/搜索行为中提取表层购物信号，无法理解用户对话中表达的深层意图：

| 意图维度 | Copilot 表现 | SLM 表现 | 涉及用户 |
|---------|-------------|---------|---------|
| **预算约束** | 从对话提取精确金额："$45 存钱买 $107 打印机"、"$300 including grinder"、"need to grind a cpl more paychecks" | 完全无法感知预算，依赖浏览价位推断 | 00DB9B18, AA7F1661, A6264706 |
| **否定信号** | "denim tops but NOT shirt style"、"I like this bag but its too small" | 无法识别否定偏好，仅做正向泛化 | CE04DD86, 1745833B |
| **使用场景** | "breakroom 员工生日公告"、"Seattle weather 跑鞋"、"spring NY work event dress" | 仅推断泛化品类，缺乏场景语境 | 1AFEFCE5, AA7F1661, FF1353FD |
| **礼物 vs 自用** | "what to gift a new mother?"、"BTS T恤 women's medium"（为妻子购买） | 无法区分，所有浏览均视为自用 | CE04DD86, E4785BFA |
| **品牌忠诚度** | "portofino shirts from express"、"why are you not suggesting from express.com?" | 搜索 "express tops" 泛化为通用 blouse，Express 品牌丢失 | 1745833B |
| **用户专业度** | 从京都购买 sashiko 用品的经历识别为进阶刺绣爱好者 → 推荐专业丝线 | 泛化为通用刺绣入门 → 推荐 Hobby Lobby $3 套件 | CE04DD86 |
| **风格偏好** | "moody-luxe, business-casual"、"pastel, creative, flowy, designer brands" | Profile 风格标签较浅，无法关联到具体产品选择 | CE04DD86, FF1353FD |
| **用户身份** | 识别出 Paparazzi 销售顾问（文案写作模式）、测量公司行政人员 | 无法区分"卖家"与"买家"：将进货浏览当消费者购物 | 3BC2C939, 1AFEFCE5 |

### 根因

**数据源性质的本质差异（L1 层）。** 对话交互天然携带深层语义（预算、场景、否定条件），浏览记录天然只有行为事实（看了什么、点了什么）。这是**架构层面的差距**，无法通过优化 SLM 模型本身解决，需要：
- 短期：增强 Profile 建设，从浏览行为推断预算区间（如用户浏览的产品价格分布）、品牌忠诚度（如同品牌反复浏览）
- 长期：融合 Copilot 对话信号，将深层意图传递给 SLM

### SLM 反向优势

SLM 在以下场景下意图捕捉优于 Copilot：

| 优势场景 | 具体案例 | 用户 |
|---------|---------|------|
| **实时行为捕捉** | Avery.com 标签浏览（35 分钟前），Copilot 无此信号 | 1AFEFCE5 |
| **对比购物识别** | Nike Dunk Low 多色系密集浏览、BABYBJÖRN vs Lillebaby vs Happy! | 6614E456, CE04DD86 |
| **无对话式品类** | 美甲 30+ 条浏览、涂色书/画笔、LEGO、家庭健身——Copilot 对话中完全不存在 | AA7F1661, 00DB9B18 |
| **尺码精确需求** | Ann Taylor Petite Eva Pant、curvy fit 等精确信号 | FF1353FD |
| **Copilot 意图误读纠正** | Copilot 将推广文案写作当购物意图（14 个 journey 全 0 产品），SLM 生成了有效推荐 | 3BC2C939 |

---

## 差距二：Journey 主题覆盖面窄 ｜ 根因层级：L1+L2

> **影响用户：10/12（83.3%）SLM Journey 数 < Copilot ｜ SLM 总计 74 vs Copilot 151（49%） ｜ 零重叠用户：5/12（41.7%） ｜ 优先级：P1**

### 表现

| 用户 ID | Copilot | SLM | SLM/Copilot | 品类重叠度 |
|---------|---------|-----|-------------|----------|
| 00DB9B18 | 8 | 12 | **150%** | 低（部分重叠） |
| 0BD7ADCE | 7 | 6+ | ~86% | **零重叠** |
| 1745833B | 20 | 2 | 10% | 极低 |
| 1AFEFCE5 | 13 | 2 | 15% | 极低 |
| 3BC2C939 | 14 | 5 | 36% | 中 |
| 58767B1A | 9 | 5 | 56% | **零重叠** |
| 6614E456 | 7 | 7 | 100% | **零重叠** |
| A6264706 | 12 | 6 | 50% | 低 |
| AA7F1661 | 18 | 8 | 44% | **零重叠** |
| CE04DD86 | 15 | 7 | 47% | 中 |
| E4785BFA | 14 | 4 | 29% | **零重叠** |
| FF1353FD | 14 | 10 | 71% | 中 |
| **总计** | **151** | **74** | **49%** | — |

### 根因

三个层面共同导致覆盖面差距：

**1. L1：历史数据量差距悬殊。** 最极端的 1745833B 仅 2 条 SLM 浏览 vs 20 组 Copilot 对话；1AFEFCE5 仅 10 条浏览 vs 8 组深度对话。数据量直接限制了 SLM 可生成的 journey 数量。

**2. L2：SLM 缺少 trending/all_in_one 类型。** Copilot 通过 related（关联品类）、trending（趋势推荐）、all_in_one（跨品类搭配）三种扩展类型显著增加 journey 数量。SLM 几乎不生成这些类型——这不仅是模型能力问题，更是**SLM 的 Profile 在冷启动用户中为空白**（如 1745833B、58767B1A），无法支撑 trending 生成。

**3. L1（数据源性质）：两端信号本质不同导致零重叠。** 12 个用户中 5 个（42%）的两端 journey **完全不重叠**——这不是"SLM 覆盖不足"，而是两个 pipeline 看到了同一用户的不同购物面。说明两端数据源具有极强互补性。

### SLM 覆盖面优于 Copilot 的案例

| 用户 | SLM 独有品类 | Copilot 遗漏原因 |
|------|------------|----------------|
| 00DB9B18 | HP 墨盒/打印机（100%）、投影仪、涂色书、SodaStream | Copilot 有 $107 打印机存钱对话但**未生成 journey** |
| CE04DD86 | 家居装饰（73%）、裤装（78%）、搭配鞋（100%） | Copilot 有 14 轮 cupboard 对话但**未生成 journey** |
| FF1353FD | 手拿包/耳饰/高跟鞋/丝巾/桌旗（多个 100%） | Copilot 对话未覆盖这些品类 |
| 6614E456 | 球鞋/理发器/边桌/CT6 配件/街头卫衣 | Copilot 从 1 组腕表对话生成 7 个全腕表 journey，品类多样性为零 |

---

## 差距三：TermID 生成质量问题 ｜ 根因层级：L3（P0 级）

> **影响用户：8/12（66.7%） ｜ 优先级：P0（品牌虚构/seller 退化）+ P1（语义漂移/品牌遗漏）**
>
> 四类子问题统计：品牌虚构 1/12 用户、语义漂移 4/12 用户、品牌遗漏 4/12 用户、seller 字段退化 3/12 用户（42+ TID 受影响）。

### 3.1 品牌虚构（Brand Hallucination）

| 用户 | Journey | 虚构品牌 | 用户实际浏览品牌 | 匹配率 |
|------|---------|---------|---------------|-------|
| CE04DD86 | 婴儿背带 | Babylondon, Zazamalls, Klutch, Babycocoon, InfantGear, Kaliuli | BABYBJÖRN, Lillebaby, Happy! | **0%**（12/12 none） |
| CE04DD86 | 婴儿配套用品 | Poloo（虚构婴儿枕品牌） | — | 37.5% |

**根因（L3 模型层）：** 模型在婴儿背带品类的品牌知识库为空。当模型对某品类品牌不确定时，**倾向于编造看起来合理的品牌名，而非使用用户历史中的真实品牌或留空**。这是模型训练数据在小众品类上覆盖不足 + fallback 策略缺失的双重问题。

### 3.2 语义漂移（Semantic Drift）

| 用户 | Journey | 用户搜索 | TID 实际描述 | 漂移方向 | 匹配率 |
|------|---------|---------|------------|---------|-------|
| 58767B1A | 宝石手链 | "healing gemstone bracelets"（疗愈水晶） | crystal beaded, heart stretch（时尚饰品） | 疗愈水晶→时尚饰品 | 83% 但品类错 |
| 3BC2C939 | 狗零食 | Blue Buffalo Nudges 10oz treats | dog food/dry 24lb 主食 | 零食→主食 | 36% exact |
| E4785BFA | Supreme×TNF | Supreme x The North Face Steep Tech Fleece | 普通 TNF insulated 功能外套 | 联名限量→普通功能 | 100% 但 0 件正确 |
| CE04DD86 | 刺子绣 | DARUMA sashiko 专业丝线 | 通用 embroidery kit 入门套件 | 专业级→入门级 | 54.5% |
| 58767B1A | 水壶 | "ello water bottle"（Ello 品牌） | Hydro Flask, JoyJolt（Ello 缺失） | 品牌特定→品牌泛化 | 高匹配但品牌错 |

**根因（L3 格式+模型层）：** TID 的结构化 7-slot 格式（`[品类, 属性1, 属性2, 属性3, 品牌, Seller, 附加属性]`）在以下场景下系统性丧失信息：
- **无法表达否定语义**（"not shirt style"）
- **无法携带联名限定词**（"Supreme x" 被丢弃，仅保留 "The North Face"）
- **品牌搜索被泛化**（"ello water bottle" → 通用 water bottle）
- **品类粒度不足**（treats 零食 vs food 主食无法区分）

这是 TID 格式本身的表达力限制，加上模型对长尾搜索关键词的保真度不足。

### 3.3 品牌遗漏

| 用户 | 用户搜索的品牌 | TID 中替代品牌 | 影响 |
|------|-------------|-------------|------|
| 1745833B | Express Portofino | PAIGE, Inspire Chic, Allegra K | Express 品牌忠诚度丢失 |
| E4785BFA | Rancilio Silvia | 全部 Breville | 对比购物中的关键品牌缺失 |
| 58767B1A | Ello | Hydro Flask, JoyJolt | 品牌搜索被泛化 |
| 0BD7ADCE | BUNN（商用咖啡机） | Keurig, Cuisinart（家用） | 商用→家用品类降级 |

**根因（L3 模型层）：** TID 生成模型**未将用户历史中的品牌作为硬约束**。当用户明确搜索某品牌时，TID 应优先包含该品牌；但当前模型倾向于用训练数据中出现频率更高的"替代品牌"覆盖，丢失了用户的品牌特异性需求。

### 3.4 TID Seller 字段（Index 5）结构退化

TID 第 5 个字段（index 5）应填入零售商名称，但在多个品类中被产品属性值占据：

| 用户 | Journey | Index 5 退化值示例 | 应填写的 Seller |
|------|---------|------------------|--------------|
| 00DB9B18 | HP 墨盒 | CH563WN#140, N9H62AN#140（产品型号） | HP Store, Amazon |
| 00DB9B18 | 涂色书/画笔 | "adult", "12-piece", "black" | Amazon, Hobby Lobby |
| AA7F1661 | 美甲 | "aqua", "sparkle", "sunrise"（颜色） | Amazon, Sally Beauty |
| FF1353FD | 手拿包 | "black", "champagne", "elegant", "party", "wedding"（10 个属性值，仅 3 个有效 seller） | Nordstrom Rack, Amazon |
| FF1353FD | 丝巾 | "14-momme", "cream", "floral"（4 个属性值，仅 1 个有效 seller） | Nordstrom Rack, J.Crew |

**根因（L3 模型层）：** 模型在小众/配饰品类中**无法正确区分 TID 各 slot 的语义角色**。在女装/电子等主流品类中，模型能准确区分品牌/seller/属性等维度；但在美甲、配饰、手工等小众品类中，维度边界模糊，导致属性值"溢出"到 seller 字段。本质是**模型在小众品类上的 TID 格式理解不足**。

### 与 Copilot Query 生成的对比

Copilot 使用自然语言 query 检索 Bing Shopping API，天然不存在 TID 结构退化问题：
- "women's white Express Portofino shirt" → 精确品牌+系列+颜色
- "silk sashiko embroidery thread jewel tones" → 材质+品类+色调
- "large cork bulletin boards office use" → 场景+尺寸+材质

SLM 的 TID 结构化格式在简单品类中高效（女装、电子），但在复杂品类中表达力不足且易退化。

---

## 差距四：Seller Authority 降级 ｜ 根因层级：L3（P0 级）

> **影响用户：10/12（83.3%） ｜ 优先级：P0**
>
> 三种降级模式：全价→折扣（3 用户）、垂直/专业→大众（3 用户）、品牌官网→第三方（3 用户）。Seller 降级与差距三同源于 L3，但影响维度不同——差距三影响品牌/品类精准度，差距四影响零售渠道权威性和价格定位。

### 表现

**Seller 档次分布：**

| 档次 | Copilot 占比 | SLM 占比 | 差距 |
|------|------------|---------|------|
| Luxury 奢侈品（Nordstrom, Bloomingdale's, Saks 等） | **10.3%** | **4.6%** | Copilot 2.2 倍 |
| Premium 高端（Macy's, Nordstrom Rack, REI 等） | **17.9%** | **7.0%** | Copilot 2.6 倍 |
| Mass 大宗零售商（Amazon, Walmart, Target, Kohl's 等） | 31.2% | 29.5% | 持平 |
| Other（专业/小众渠道） | 40.6% | 58.8% | SLM 偏高 |

**Top Seller 对比——注意第 4 位的完美对比：**

| 排名 | Copilot | 次数 | SLM | 次数 |
|------|---------|------|-----|------|
| 1 | Amazon.com | 319 | Amazon.com | 271 |
| 2 | Macy's | 179 | Home Depot | 99 |
| 3 | **Etsy** | 147 | B&H Photo | 56 |
| 4 | **Nordstrom** | 144 | **Nordstrom Rack** | 55 |
| 5 | Walmart | 141 | Kohl's | 50 |

Copilot 独有高权威 Seller：**Etsy**（147）、**Nordstrom 全价**（144）、Seattle Coffee Gear（19）、Neiman Marcus（7）
SLM 独有 Seller：Newegg（30）、Academy Sports（14）、Famous Footwear（10）、Hobby Lobby（9）

### 根因：L3（TID 生成）是主要问题，不是 L4（匹配）

**核心发现：Seller 降级发生在 L3 阶段，L4 匹配引擎是忠实执行者。**

**证据 1：L4 忠实执行 TID sellers**

当 TID 指定了有效 seller，matched_products 基本 100% 执行：

| 用户 | Journey | TID Seller | Matched Seller | L4 保真度 |
|------|---------|-----------|---------------|----------|
| AA7F1661 | 复古运动鞋 | Famous Footwear, Zappos, Nike.com, DSW, DICK'S | 完全一致 | 100% |
| E4785BFA | 咖啡研磨机 | Whole Latte Love, Sur La Table, Clive Coffee | 完全一致 | 100% |
| 1AFEFCE5 | 公告板 | Staples ×10, Amazon ×1 | Staples ×8, Amazon ×1 | ~100% |

**证据 2：L3 系统性替换用户偏好 seller**

TID 生成器未将用户历史中的 seller 作为约束，自主选择产品目录中容易匹配的大众零售商：

| 用户 | 用户浏览 Seller | L3 TID 替换为 | 降级模式 |
|------|---------------|-------------|---------|
| FF1353FD | Neiman Marcus, Bergdorf Goodman | Nordstrom Rack, belk, LOFT | 全价→折扣 |
| 1745833B | Express（用户明确要求 express.com） | Kohl's ×6, Nordstrom Rack | 品牌官网→折扣 |
| 6614E456 | IKEA | Wayfair, Amazon | 品牌官网→综合电商 |
| CE04DD86 | Etsy, Pottery Barn Kids | Hobby Lobby, Home Depot, 虚构 seller | 专业/垂直→大众 |
| A6264706 | Apple Store（几乎 100% 浏览来源） | B&H, Adorama, Best Buy | 品牌官网→授权经销商 |

**证据 3：L4 偶尔纠正 L3 错误**

少数案例中 L4 在 TID 未指定的情况下恢复了正确 seller：
- **FF1353FD 裤装**：TID 未包含 Ann Taylor，但 matched_products 恢复了 5 个 Ann Taylor 产品

**根因总结（L3 模型层）：** TID 生成器倾向于选择**训练数据中出现频率最高的大众零售商**（Amazon、Kohl's、Nordstrom Rack、Home Depot），即使用户 Profile 中有明确的 retailerPreferences（如 FF1353FD 的 Neiman Marcus）。Profile 中已存在的信号未被利用，是**模型层面的约束缺失**。

### 三种降级模式

**模式 1：全价百货→折扣百货**

| 用户 | 用户偏好 | L3 替换为 | 价格影响 |
|------|---------|----------|---------|
| FF1353FD | Neiman Marcus ($150-500) | Nordstrom Rack ($60-120) | 降 50-70% |
| FF1353FD | Bergdorf Goodman ($200-650) | LOFT ($60-90) | 降 60-80% |
| 1745833B | Express ($50-63) | Kohl's ($12-40) | 降 40-75% |
| CE04DD86 | Nordstrom (正价) | Nordstrom Rack (折扣) | 档次下移 |

**模式 2：垂直/专业→大众零售**

| 用户 | 用户偏好 | L3 替换为 | 品类影响 |
|------|---------|----------|---------|
| CE04DD86 | Etsy/eBay（sashiko 专业丝线） | Hobby Lobby, Amazon（入门套件） | 专业级→入门级 |
| E4785BFA | StockX/GOAT（Supreme 限量平台） | DICK'S, Macy's（大众运动零售） | 潮流限量→大众功能 |
| 0BD7ADCE | BUNN/WebstaurantStore（商用咖啡机） | Keurig/Cuisinart/Amazon（家用） | 商用→家用 |

**模式 3：品牌官网→第三方经销商**

| 用户 | 用户偏好 | L3 替换为 | 说明 |
|------|---------|----------|------|
| A6264706 | Apple Store | B&H, Adorama, Best Buy | Apple 产品统一定价，影响中等 |
| 6614E456 | Nike.com, adidas 官网, IKEA | Scheels, DSW, Wayfair | 品牌官网全面消失 |
| 00DB9B18 | HP 官网 | Adorama, B&H, Staples | HP 丢失但价格类似 |

### CE04DD86 Seller 流转示例图

```
婴儿背带 Journey：
L1 用户浏览                L3 TID 生成                   L4 匹配结果
┌─────────────────┐   ┌────────────────────┐   ┌──────────────────┐
│ Pottery Barn Kids│   │ Bella Luna Toys ───│──→│ none（虚构）      │
│ (BABYBJÖRN)     │   │ Zazamalls ─────────│──→│ none（虚构）      │
│                 │   │ Athleta ───────────│──→│ none（不卖背带）  │
│ Amazon          │   │ belk ──────────────│──→│ none              │
│ (Happy!)        │   │ Babylist ──────────│──→│ none              │
└─────────────────┘   └────────────────────┘   └──────────────────┘
  用户真实渠道           全部虚构/不相关            12/12 = 0%
  ↑ 完全丢失            Pottery Barn Kids          匹配全失败
                        从未出现在 TID

刺子绣 Journey：
L1 用户浏览                L3 TID 生成                   L4 匹配结果
┌─────────────────┐   ┌────────────────────┐   ┌──────────────────┐
│ Etsy            │   │ Amazon.com ────────│──→│ Amazon.com ✅    │
│ (sashiko kits)  │   │ Hobby Lobby ───────│──→│ Hobby Lobby ✅   │
│ eBay            │   │ Michaels ──────────│──→│ none              │
│ (日本进口套件)  │   │                    │   │                  │
└─────────────────┘   └────────────────────┘   └──────────────────┘
  Etsy/eBay 专业渠道     替换为大众渠道            L4 忠实执行 L3
  ↑ 完全丢失             Etsy 从未出现             但渠道已降级
```

---

## 差距五：小众品类产品目录覆盖不足 ｜ 根因层级：L4

> **影响用户：6/12（50%） ｜ 匹配率 < 50% 的品类：7 个，累计 54 个 TID 匹配失败 ｜ 优先级：P1**
>
> 这是唯一根因在 L4（产品匹配/目录层）的差距。即使 L3 的 TID 完全正确，如果产品目录中缺少对应品类，匹配仍然会失败。

### 表现

| 用户 | 品类 | 匹配率 | 问题描述 |
|------|------|-------|---------|
| CE04DD86 | 婴儿背带 | **0%** | TID 品牌全虚构（L3 问题），但即使品牌正确，目录中也可能缺失 |
| 6614E456 | 理发配件 | **9.1%** | barber cape、disinfectant jar 等专业配件在目录中空白 |
| 00DB9B18 | SodaStream 苏打水机 | **18.2%** | "soft drink maker" 品类覆盖极其有限 |
| 3BC2C939 | 狗零食 | **36%** | TID 漂移为主食（L3 问题）+ 零食品类目录覆盖不足 |
| AA7F1661 | 美甲 | **37.5%** | 猫眼磁性胶、3D sculpting gel 等专业美甲产品缺失 |
| 58767B1A | Hello Kitty 授权 | **46.2%** | Stoney Clover Lane×HK、BAGGU×HK 等 IP 联名产品缺失 |
| 00DB9B18 | Cisco VoIP 电话 | **50%** | 企业级 VoIP 设备在消费级目录中覆盖有限 |
| 6614E456 | CT6 汽车配件 | **55.6%** | 车型专用配件目录有限 |

### 根因

**架构差异是核心原因（L4 目录层）。** Copilot 通过 Bing Shopping API **实时搜索**全网索引产品，天然覆盖长尾和小众品类；SLM 通过 TID **匹配预构建的产品目录**，目录未覆盖的品类必然匹配失败。以下品类存在系统性覆盖缺口：

- **MLM 直销品牌**（Paparazzi $5 饰品——不在主流电商渠道）
- **IP 联名/限量产品**（Hello Kitty×Loungefly、Supreme×TNF——限时限量、非标准 SKU）
- **专业/商用设备**（BUNN 商用咖啡机、Cisco VoIP 电话、理发专业配件——面向 B2B 市场）
- **手工/独立品牌**（Etsy 手工吊灯、sashiko 专用丝线——长尾商品）

注意：部分低匹配率品类的问题同时涉及 L3（如 CE04DD86 婴儿背带的品牌虚构、3BC2C939 狗零食的语义漂移），需要 L3+L4 联合修复。

### SLM 在主流品类的卓越表现

| 用户 | 品类 | 匹配率 | 说明 |
|------|------|-------|------|
| 0BD7ADCE | 女装上衣 | **100%**（17/17） | 大众女装品类目录完善 |
| 3BC2C939 | 项链/耳环/手链/裤装 | **100%**（46/46） | 主流饰品/服装 |
| 1745833B | 女装上衣 | **100%**（19/19） | 同上 |
| A6264706 | iMac/iPad/iPhone | **100%** | Apple 产品目录完善 |
| 00DB9B18 | HP 墨盒/打印机 | **100%** | 主流电子耗材 |
| FF1353FD | 手拿包/高跟鞋/丝巾/桌旗 | **100%** | 主流配饰/家居 |
| E4785BFA | 咖啡研磨机 | **90.9%** | 主流咖啡设备 |

**规律：主流消费品类（女装、运动鞋、电子、咖啡设备）匹配率通常 85%+；小众/专业品类通常 < 50%。**

---

## 差距六：商品 Popularity 与价格定位差距 ｜ 根因层级：L3→L4 连锁

> **Copilot trending 价格偏离 >5× 的用户：3/12（25%） ｜ SLM 价格系统性偏低的用户：3/12（25%） ｜ 中位价差距：Copilot $110 vs SLM $69（59%） ｜ 优先级：P2（修复差距四后自动缓解）**

### 表现

| 统计量 | Copilot | SLM | Copilot/SLM |
|--------|---------|-----|-------------|
| P25 | $45 | $30 | 1.5× |
| **中位数** | **$110** | **$69** | **1.6×** |
| P75 | $279 | $160 | 1.7× |
| **均价** | **$291** | **$185** | **1.6×** |
| 最高价 | $7,725 | $4,199 | — |

Copilot 产品中位价比 SLM 高出 **59%**。

### 根因

**L3 Seller 降级（差距四）的直接连锁反应：**

```
L3 Seller 降级              →  产品价格下移
Neiman Marcus ($150-500)    →  Nordstrom Rack ($60-120)     降 50-70%
Etsy 专业丝线 ($15-63)      →  Hobby Lobby 入门套件 ($3-5)   降 90%+
Express 全价 ($50-63)       →  Kohl's 折扣 ($12-40)         降 40-75%
```

当 L3 将高端渠道替换为折扣渠道时，产品价格**必然**随之下移。因此修复 Seller Authority（差距四）将同时缓解价格定位差距。

### 重要注意：Copilot 的高端定位并非总是正确

Copilot 的 trending journey 存在**价格严重偏离用户实际消费水平**的问题：

| 用户 | Trending 推荐 | 价格 | 用户实际水平 | 偏差 |
|------|-------------|------|------------|------|
| 1AFEFCE5 | Miu Miu/Prada/Balenciaga 手包 | $2,100-$4,200 | Michael Kors $279-$358 | **10-12×** |
| 6614E456 | Tudor/Omega 机械表 | $5,475-$6,700 | Andre Rivalle 古董表 $100-$250 | **22-27×** |
| 00DB9B18 | Breville 咖啡机 | $800-$1,500 | 中学生 $5/周零花钱 | **160-300×** |

Copilot 的 trending journey 未参考用户 explicit journey 的价位锚点。在 00DB9B18 等价格敏感用户身上，**SLM 的大众渠道（Amazon、Walmart）反而更合适**。

---

## 差距七：冷启动与过度扩展 ｜ 根因层级：跨层（L1+L2+Profile）

> **SLM 冷启动（浏览 < 10 条）：3/12（25%） ｜ Copilot 过度扩展（1 信号→5+ journey）：3/12（25%） ｜ 优先级：P2**

### SLM 冷启动问题

当浏览记录极少时，SLM 的 Profile 通常为空，无法支撑 trending/related journey 生成：

| 用户 | 浏览记录数 | SLM Journey | Copilot Journey | 问题 |
|------|----------|------------|----------------|------|
| 1745833B | **2 条** | 2 | 20 | Profile 全空，仅覆盖用户 10% 的购物兴趣 |
| 1AFEFCE5 | **10 条** | 2 | 13 | 仅覆盖 15% |
| 58767B1A | **6 条** | 5 | 9 | Profile 空白，部分 journey 语义漂移 |
| E4785BFA | **7 条** | 4 | 14 | Profile 仅 2 个品类偏好 |

**根因（L1→Profile→L2 连锁）：** SLM 的 Profile 构建**需要足够的浏览事件才能激活**。当事件数 < 10 时，Profile 多数字段为空，无法提供 trending/all_in_one 的生成基础，也无法推断用户画像（性别、年龄、消费水平）。

### Copilot 过度扩展问题

Copilot 在信号稀少时倾向于从单一信号过度生成 journey：

| 用户 | 购物信号 | 生成 Journey | 问题 |
|------|---------|-------------|------|
| 0BD7ADCE | 1 条枕头搜索 | 7 个 | 1→7 过度扩展，6 个衍生缺乏直接信号 |
| 58767B1A | 2 个品类 | 9 个 | 产品严重重复（Snap Circuits Jr. 出现 4 次） |
| 6614E456 | 1 组腕表对话 | 7 个全腕表 | 品类多样性为零 |

**根因（L2 层）：** Copilot 的 related/trending/all_in_one 扩展机制**缺少与 explicit journey 信号强度的比例约束**。1 条搜索应最多衍生 1-2 个 related，而非 6 个。

---

## 差距总结与优先级

### 按 Pipeline 层级和修复优先级排序

| 优先级 | 差距 | 根因层级 | 涉及用户 | 核心改进动作 |
|--------|------|---------|---------|------------|
| **P0** | TID 品牌虚构（差距三） | L3 模型 | CE04DD86 | 不确定品牌时使用用户历史品牌或留空，禁止编造 |
| **P0** | TID seller 字段退化（差距三） | L3 模型 | 00DB9B18, AA7F1661, FF1353FD 等 | 强制 index 5 仅生成零售商名称 |
| **P0** | L3 系统性 seller 降级（差距四） | L3 模型 | FF1353FD, 1745833B, CE04DD86, 6614E456, A6264706 | 将 Profile retailerPreferences 作为 TID seller 硬约束 |
| **P1** | 语义漂移（差距三） | L3 模型 | 58767B1A, 3BC2C939, E4785BFA, CE04DD86 | 增强对搜索关键词的保真度 |
| **P1** | 品牌遗漏（差距三） | L3 模型 | 1745833B, E4785BFA, 58767B1A, 0BD7ADCE | 用户搜索的品牌必须出现在 TID 中 |
| **P1** | 小众品类目录覆盖（差距五） | L4 目录 | 6614E456, 00DB9B18, AA7F1661, 58767B1A | 扩充美甲/VoIP/理发/婴儿背带/IP 联名等品类 |
| **P1** | Journey 数量不足（差距二） | L1+L2 | 1745833B, 1AFEFCE5 | 增加 trending/all_in_one 生成能力 |
| **P2** | 意图理解深度（差距一） | L1 数据源 | 全部 12 用户 | 长期：融合对话信号；短期：增强 Profile 建设 |
| **P2** | 冷启动策略（差距七） | L1+Profile | 1745833B, 58767B1A, 1AFEFCE5 | 极少事件时结合 trending 生成探索性 journey |
| **P2** | 价格定位（差距六） | L3→L4 连锁 | 多数用户 | 修复差距四后将自动缓解 |

### SLM 优势总结

| 优势维度 | 具体表现 | 涉及用户 |
|---------|---------|---------|
| **主流品类匹配率极高** | 女装/饰品/Apple/打印耗材等多个 100% exact | 0BD7ADCE, 3BC2C939, A6264706, 00DB9B18, FF1353FD |
| **覆盖 Copilot 遗漏品类** | 打印机（Copilot 有对话但遗漏）、家居装饰（14 轮对话但遗漏）、Petite 裤装 | 00DB9B18, CE04DD86, FF1353FD |
| **实时行为捕捉** | 35 分钟前的 Avery.com 浏览、1 天前的 Hello Kitty 搜索即时响应 | 1AFEFCE5, 58767B1A |
| **大众渠道适配价敏用户** | Amazon/Walmart 比 Breville $1,500 更适合中学生用户 | 00DB9B18 |
| **Related Journey 逻辑合理** | 咖啡机→研磨机（90.9%）、裤装→搭配鞋（100%）、球鞋→街头卫衣 | E4785BFA, CE04DD86, 6614E456 |
| **Copilot 系统故障时弥补** | 3BC2C939 Copilot 14 journey 全 0 产品（系统 bug），SLM 100% exact | 3BC2C939 |

### 最终判断

SLM 与 Copilot 的差距可分为**可修复差距**和**结构性差距**两类：

**可修复差距（应优先投入）：** TID 品牌虚构、seller 字段退化、系统性 seller 降级、语义漂移和品牌遗漏——这些都是 **L3 层面的模型质量问题**，可通过改进 TID 生成逻辑解决。特别是 retailerPreferences 信号已存在于用户 Profile 中但未被利用，修复成本相对较低。小众品类的产品目录覆盖也可通过扩充索引改善（L4 层）。

**结构性差距（需长期投资）：** 意图理解深度（预算约束、否定信号、使用场景、用户身份识别等）根本上源于 L1 数据源性质的差异——浏览记录天然无法提供对话交互中的深层语义信息。缩小这一差距需要**信号源融合**（将 Copilot 对话信号与 SLM 浏览信号交叉引用）或引入更丰富的上下文信号。

**值得强调的是，SLM 和 Copilot 展现出极强的互补性。** 12 个用户中有 5 个（42%）的 journey 完全不重叠，说明两个 pipeline 看到了同一用户完全不同的购物面。SLM 在产品匹配精度和实时行为捕捉上的优势，与 Copilot 在深度意图理解和品类扩展上的优势，可以形成 **1+1 > 2** 的效果。理想方案是将两种信号源融合，而非视为竞争关系。

---

## 具体优化建议

### P0 级：L3 TID 生成质量修复

P0 的三个问题（品牌虚构、seller 字段退化、seller 系统性降级）本质上是同一层（L3）的不同症状，可通过**统一的 TID 后处理校验管道**一并解决：

**1. TID 品牌/Seller 白名单校验（后处理）**

在 TID 生成后、送入 L4 匹配前，增加一道校验步骤：
- 从产品目录中提取品牌白名单和 Seller 白名单
- 校验 TID index 4（品牌）是否在品牌白名单中 → 不在则替换为用户历史中的真实品牌或留空（**禁止编造**）
- 校验 TID index 5（seller）是否在 Seller 白名单中 → 不在则替换（解决 "aqua"/"CH563WN#140" 等属性值占据 seller 位的问题）

**2. 用户信号约束注入（生成时）**

TID 生成时将用户 Profile 中已有的信号作为约束条件：
- `retailerPreferences`（如 Neiman Marcus、Etsy）→ 约束 seller slot，禁止降级为 Kohl's/Nordstrom Rack
- `brandPreferences`（如 BABYBJÖRN、Express）→ 约束品牌 slot，禁止替换为训练数据中的高频替代品牌
- 浏览历史中的品牌官网（Express.com、Apple Store、IKEA）→ 优先保留品牌官网 seller

**3. Seller 档次一致性检查**

当用户历史 seller 档次为 Luxury（如 Neiman Marcus），TID 中不应出现低 2 档以上的渠道（如 Kohl's）。跨档触发替换或告警。

**预期效果：**
- CE04DD86 婴儿背带匹配率 0% → 50%+
- FF1353FD seller 从 Nordstrom Rack 恢复为 Neiman Marcus
- 整体 Luxury+Premium 占比从 11.6% → 20%+
- 消除所有 seller 字段退化问题

### P1 级：TID 语义保真 + 品牌保留 + 目录扩充 + Journey 扩展

**1. 搜索关键词保真**

用户搜索中的关键语义限定词必须保留在 TID 中：
- 联名标识（"Supreme x TNF" → 不可丢弃 "Supreme"）
- 品类子类（"dog treats" → 不可漂移为 "dog food"）
- 用途限定（"healing gemstone" → 不可泛化为 "fashion bracelet"）

**2. 用户品牌强制包含**

用户明确搜索或反复浏览（≥3 次）的品牌必须出现在 TID 中，不可被训练数据中的高频替代品牌覆盖。对比购物中的多个品牌（如 BABYBJÖRN vs Lillebaby）应同时保留。

**3. 小众品类产品目录扩充**

按 none 率排序优先扩充：

| 优先级 | 品类 | none 率 | 扩充方向 |
|--------|------|---------|---------|
| 最高 | 婴儿背带 | 100% | BABYBJÖRN/Lillebaby 官网 offer |
| 最高 | 理发专业配件 | 91% | Sally Beauty/B2B 渠道 |
| 高 | SodaStream | 82% | SodaStream 官网/Best Buy |
| 高 | 专业美甲 | 63% | GAOY/Beetles 等 Amazon 店铺 |
| 中 | Hello Kitty 联名 | 54% | Sanrio 授权产品聚合 |
| 中 | Cisco VoIP | 50% | B2B 渠道 offer |

**4. Journey 数量和类型扩展**

- 基于 Profile 生成 trending journey（当 categoryPreferences 非空时）
- 为 explicit journey 自动衍生 related journey（已验证高质量：裤装→鞋 100%、咖啡机→研磨机 91%）
- 冷启动用户（浏览 < 5 条）基于品类上位类生成探索性 journey



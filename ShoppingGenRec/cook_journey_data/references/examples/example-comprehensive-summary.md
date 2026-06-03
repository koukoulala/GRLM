# Copilot vs SLM Shopping Journey 管线综合对比分析报告

**分析日期:** 2026-04-23
**分析范围:** 12 个在两条管线中均生成了 Journey 的用户

---

## 1. 概述

### 分析范围

本报告仅覆盖在 Copilot 和 SLM 两条管线中**均生成了 Journey** 的 12 个用户。通过逐用户深入对比，系统性地评估两条管线在 Journey 生成质量、产品匹配精准度、Seller 权威性等方面的差异。

### 两条管线的数据来源差异

| 维度 | Copilot | SLM |
|------|---------|-----|
| **数据来源** | 用户与 Copilot 的购物对话记录 + Profile | 用户浏览/搜索/点击行为记录 + Profile |
| **信号深度** | 多轮对话，包含意图、预算、场景、品牌偏好等深层信息 | 浏览行为，记录品类、品牌、渠道等行为轨迹 |
| **信号广度** | 受限于用户主动发起的对话话题 | 覆盖用户全渠道的浏览行为 |
| **时间窗口** | 对话历史跨度较长（3-115 天） | 近期浏览事件（1-34 天） |

### 差异归因框架（L1-L4）

| 层级 | 名称 | 描述 |
|------|------|------|
| **L1** | 用户历史差异 | 两端采集到的用户购物信号本身不同 |
| **L2** | Journey 生成差异 | 从相同/类似信号中生成的 Journey 方向、粒度、数量不同 |
| **L3** | Query/TermID 生成差异 | Journey 内部的搜索词和 TermID 质量差异 |
| **L4** | Product 匹配差异 | 最终匹配到的产品在品牌、价格、渠道上的差异 |

---

## 2. 用户数据覆盖统计

### 逐用户数据概览

| 用户 ID | Copilot 购物对话数 | SLM 浏览记录数 | Copilot Journey 数 | SLM Journey 数 |
|---------|-------------------|---------------|--------------------|----|
| 00DB9B18 | 16 | 71 | 8 | 12 |
| 0BD7ADCE | 1 | 128 | 7 | 6+ |
| 1745833B | 20 | 2 | 20 | 2 |
| 1AFEFCE5 | 8 | 10 | 13 | 2 |
| 3BC2C939 | 0（非购物对话推断） | 56 | 14 | 5 |
| 58767B1A | 1 | 6 | 9 | 5 |
| 6614E456 | 1 | 62 | 7 | 7 |
| A6264706 | 6 | 30 | 12 | 6 |
| AA7F1661 | 23 | 116 | 18 | 8 |
| CE04DD86 | 30 | 84 | 15 | 7 |
| E4785BFA | 14 | 7 | 14 | 4 |
| FF1353FD | 19 | 94 | 14 | 10 |

### 汇总统计

| 指标 | Copilot | SLM |
|------|---------|-----|
| **总 Journey 数** | 151 | 74+ |
| **平均 Journey 数/用户** | 12.6 | 6.2 |
| **中位 Journey 数/用户** | 13.5 | 6 |
| **Journey 数最多** | 20（1745833B） | 12（00DB9B18） |
| **Journey 数最少** | 7（0BD7ADCE/6614E456） | 2（1745833B/1AFEFCE5） |

**核心发现：** Copilot 平均生成的 Journey 数量是 SLM 的 **2 倍**，主要因为 Copilot 大量使用 related/trending/all_in_one 类型进行品类扩展。SLM 以 explicit Journey 为主，数量更少但更聚焦于用户实际浏览行为。

---

## 3. Journey 语义匹配统计

### 匹配结果汇总

从 12 个用户的语义匹配分析中提取所有匹配对：

| 匹配度 | 匹配对数 | 涉及用户 |
|--------|---------|---------|
| **HIGH** | 7 | 3BC2C939（项链、耳环、手链）、CE04DD86（刺绣、女装上衣）、FF1353FD（连衣裙、blouse、吊灯） |
| **MEDIUM** | 12 | 00DB9B18（Cisco/PA系统、打印×2）、1745833B（上衣、健身可穿戴）、1AFEFCE5（公告板）、CE04DD86（婴儿、裤装）、A6264706（平板、耳机）、FF1353FD（裤装） |
| **NO MATCH** | 大量 | 0BD7ADCE、58767B1A、6614E456、AA7F1661、E4785BFA 等用户两端 Journey 完全不重叠 |

### 品类匹配模式

**匹配良好的品类：**
- **女装**（上衣/连衣裙/blouse）：FF1353FD、CE04DD86、1745833B 均有 HIGH/MEDIUM 匹配
- **饰品**（项链/耳环/手链）：3BC2C939 有 3 个 HIGH 匹配
- **吊灯/照明**：FF1353FD 有 HIGH 匹配
- **办公用品**（公告板）：1AFEFCE5 有 HIGH 匹配

**匹配极差的品类（两端完全不重叠）：**
- 0BD7ADCE：Copilot→枕头 vs SLM→女装/泳衣/咖啡机（零匹配）
- 58767B1A：Copilot→STEM玩具/espresso vs SLM→Hello Kitty/宝石手链/水壶（零匹配）
- 6614E456：Copilot→古董腕表×7 vs SLM→球鞋/理发器/边桌/CT6配件（零匹配）
- AA7F1661：Copilot→护肤/咖啡机/婚礼/跑步 vs SLM→美甲/健身设备/夹灯/LEGO（零匹配）
- E4785BFA：Copilot→泳裤/厨具/EV vs SLM→Supreme×TNF/咖啡机（零匹配）

**关键结论：** 12 个用户中，**5 个用户的两端 Journey 完全不重叠**（零匹配），占比 41.7%。这说明两条管线在大多数情况下捕获的是同一用户的**完全不同的购物面**，具有极强的互补性。

---

## 4. L1-L4 差异归因综合分析

### 4.1 L1 用户历史差异

**出现频率：** 12/12 用户（100%），是最普遍、影响最大的差异来源。

**典型模式：**

| 模式 | 出现次数 | 典型用户 | 描述 |
|------|---------|---------|------|
| **Copilot 丰富 / SLM 稀疏** | 3 | 1745833B、1AFEFCE5、E4785BFA | Copilot 有 8-20 组购物对话，SLM 仅 2-10 条浏览记录 |
| **SLM 丰富 / Copilot 稀疏** | 3 | 0BD7ADCE、6614E456、58767B1A | Copilot 仅 1 组购物搜索，SLM 有 6-128 条浏览记录 |
| **双方都有数据但完全不重叠** | 4 | AA7F1661、A6264706、E4785BFA、6614E456 | 两端捕获了用户完全不同时段/品类的购物兴趣 |
| **双方有数据且部分重叠** | 3 | 00DB9B18、CE04DD86、FF1353FD | 在部分品类上重叠（如女装、吊灯），但各有独特品类 |

**典型示例：**
- **1745833B**：Copilot 有 20 组购物对话 → 20 个 Journey；SLM 仅 2 条浏览 → 2 个 Journey。SLM 覆盖了用户购物兴趣的不到 10%
- **0BD7ADCE**：Copilot 仅 1 条枕头搜索 → 7 个 Journey（过度扩展）；SLM 有 128 条浏览 → 6+ Journey（品类多样）
- **AA7F1661**：Copilot 的 23 组对话涵盖护肤/咖啡机/婚礼；SLM 的 116 条浏览集中在美甲/健身/LEGO——"两个完全不同的用户"

### 4.2 L2 Journey 生成差异

**出现频率：** 11/12 用户存在不同程度的 L2 差异。

**Copilot 常见问题：**

| 问题 | 出现用户 | 描述 |
|------|---------|------|
| **从单一信号过度扩展** | 0BD7ADCE、58767B1A、6614E456 | 从 1 条搜索生成 7-9 个 Journey（含大量 related/trending/all_in_one），产品严重重复 |
| **Journey 遗漏** | 00DB9B18、CE04DD86 | 有明确购物意图的对话未生成独立 Journey（如 00DB9B18 的打印机、CE04DD86 的家居） |
| **误读用户意图** | 3BC2C939 | 将用户的饰品推广文案写作需求解读为购物意图 |

**SLM 常见问题：**

| 问题 | 出现用户 | 描述 |
|------|---------|------|
| **品类泛化** | 1745833B、CE04DD86 | Express Portofino → 通用 blouse；sashiko 丝线 → 通用刺绣入门套件 |
| **浏览品类遗漏** | 0BD7ADCE、6614E456、AA7F1661 | 有大量浏览的品类未生成 Journey（如 0BD7ADCE 的除油剂 20+ 条、6614E456 的女性内衣 8 条） |
| **缺少 trending/all_in_one 类型** | 多数用户 | SLM 几乎不生成 trending 和 all_in_one 类型 Journey |

### 4.3 L3 Query/TermID 生成差异

**出现频率：** 8/12 用户存在 L3 差异，其中 4 个为严重问题。

**品牌虚构（最严重问题）：**
- **CE04DD86 婴儿背带 Journey**：12 个 TID 的品牌**全部虚构**（Babylondon、Zazamalls、Klutch 等），用户实际浏览的 BABYBJÖRN、Lillebaby、Happy! 完全未出现，导致 0% 匹配率

**语义偏移：**
- **58767B1A 宝石手链 Journey**：用户搜索 "healing gemstone bracelets"（疗愈水晶），TID 偏向时尚金属手链（BaubleBar、Loft、J.Crew Factory）
- **3BC2C939 狗零食 Journey**：用户浏览 treats（10oz 零食），TID 漂移为 dog food/dry（24lb 主食）

**品牌遗漏：**
- **58767B1A 水壶 Journey**：用户搜索 "ello water bottle"，TID 中完全没有 Ello 品牌
- **E4785BFA Supreme×TNF Journey**：TID 缺少 "Supreme" 联名限定词，只检索到普通 TNF 功能性外套
- **E4785BFA 咖啡机 Journey**：用户搜索 "rancilio silvia"，TID 中完全没有 Rancilio 品牌

**TID seller 位退化：**
- 多个用户的 TID index 5（seller 字段）被产品属性值占据：00DB9B18（"CH563WN#140"、"12-piece"）、AA7F1661（"aqua"、"sparkle"）、FF1353FD（"chic"、"collared"、"dry clean"）

### 4.4 L4 Product 匹配差异

**出现频率：** 10/12 用户存在 L4 差异。

**匹配率分布（SLM Journey）：**

| 匹配率区间 | Journey 数 | 典型品类 |
|-----------|----------|---------|
| **90-100%** | 30+ | Apple 产品、女装上衣/blouse、球鞋、咖啡研磨机、夹灯、裤装、手拿包 |
| **70-89%** | 10+ | 投影仪、吊灯、标签/贴纸、家居装饰、健身追踪器 |
| **40-69%** | 8+ | Cisco IP 电话(50%)、Hello Kitty(46.2%)、CT6 配件(55.6%)、刺绣套件(54.5%) |
| **0-39%** | 6+ | 婴儿背带(0%)、SodaStream(18.2%)、理发配件(9.1%)、美甲(37.5%)、狗零食(36%) |

**价格/品牌层级偏差：**
- **Copilot trending Journey 价格严重偏离**：1AFEFCE5 用户搜索 $40-360 手包，trending 推荐 Miu Miu $2,100-$3,950（10 倍差距）；6614E456 用户 $100-250 古董表，trending 推荐 Tudor $5,475、Omega $6,700（20-50 倍差距）；00DB9B18 中学生靠 $5/周零花钱，推荐 Breville $800-1500 咖啡机
- **SLM Paparazzi 用户**（3BC2C939）：用户是 $5 饰品销售顾问，SLM 推荐 $20-85 中高档品牌（3-17 倍差距）

**系统级故障：**
- **3BC2C939 Copilot 全部 14 个 Journey 产品数组为空**（零产品推荐），系统级 bug

---

## 5. Seller Authority 综合分析

### Seller 流转分析（L1→L3→L4）

从 12 个用户的 Seller 流转追踪中，综合发现以下模式：

#### 降级模式频率

| 降级模式 | 出现用户数 | 典型案例 |
|---------|----------|---------|
| **品牌官网→第三方零售** | 8/12 | A6264706（Apple→B&H）、00DB9B18（HP官网→Adorama）、6614E456（IKEA→Wayfair）、1745833B（Express→Kohl's） |
| **全价百货→折扣百货** | 5/12 | FF1353FD（Neiman Marcus/Bloomingdale's→Nordstrom Rack）、CE04DD86（Nordstrom→Nordstrom Rack） |
| **专业渠道→大众渠道** | 6/12 | CE04DD86（Etsy sashiko→Hobby Lobby）、E4785BFA（StockX/GOAT→DICK'S）、6614E456（StockX/SNIPES→DSW） |
| **用户偏好 Seller 完全丢失** | 7/12 | 3BC2C939（Paparazzi 完全消失）、6614E456（IKEA/Aeropostale 消失）、FF1353FD（Etsy/Artemis 消失） |
| **Seller 位退化为属性值** | 5/12 | 00DB9B18、AA7F1661、FF1353FD 等用户的 TID index 5 被颜色/材质等属性占据 |

#### Seller 档次对比

| 维度 | Copilot Seller 特征 | SLM Seller 特征 |
|------|-------------------|----------------|
| **典型零售商** | Bloomingdale's, Nordstrom, Macy's, REI, Williams Sonoma, Etsy, Sephora, Sur La Table | Amazon, Walmart, Kohl's, DICK'S, Nordstrom Rack, Home Depot, DSW |
| **整体档次** | 中高档，偏向品牌直营和专业渠道 | 大众化，偏向综合零售商和折扣渠道 |
| **与用户偏好匹配度** | 受对话语境影响，有时偏高（trending 价格偏离） | 受产品目录覆盖影响，系统性偏向大众渠道 |

#### Seller 保留表现较好的品类

| 品类 | 用户 | 表现 |
|------|------|------|
| 办公用品（Staples/Newegg） | 1AFEFCE5 | Staples 10/15 TID 保留 |
| 宠物用品（Amazon/Chewy） | 3BC2C939 | Amazon 保留 |
| 咖啡设备（Williams Sonoma/Sur La Table） | E4785BFA | 专业咖啡渠道精准保留 |
| 家庭健身（DICK'S） | AA7F1661 | DICK'S 和 Walmart 保留 |

#### 根因总结

1. **L3 TID 生成是 Seller 降级的核心层级**：TID 生成器倾向选择产品目录中覆盖面更广的大众零售商，即使用户 Profile 中有明确的 retailerPreferences
2. **产品目录覆盖缺口导致被动替换**：Apple Store、IKEA、Paparazzi、Supreme 联名等品牌/渠道在 Bing Shopping 目录中 offer 不足，迫使 TID 选择替代 Seller
3. **L4 匹配引擎有一定纠偏能力**：如 FF1353FD 裤装 Journey，TID 中无 Ann Taylor，但 matched_products 恢复了大量 Ann Taylor 产品

---

## 6. 商品 Popularity 与价格定位差距

### 价格偏差模式

| 偏差类型 | 出现频率 | 涉及用户 | 具体案例 |
|---------|---------|---------|---------|
| **Copilot trending 价格严重偏高** | 3/12 | 1AFEFCE5、6614E456、00DB9B18 | 手包 $360→$3,950（10x）；古董表 $250→$6,700（27x）；中学生→$1,500 咖啡机 |
| **SLM 品牌层级偏高** | 3/12 | 3BC2C939、CE04DD86、FF1353FD | Paparazzi $5→$20-85（4-17x）；FTCayanz $25→J.Crew $138；Neiman Marcus→Nordstrom Rack |
| **SLM 品牌层级偏低** | 2/12 | CE04DD86、FF1353FD | Etsy 专业 sashiko→Hobby Lobby $3 入门；Bloomingdale's→LOFT |
| **Copilot 忽略价格敏感度** | 2/12 | 00DB9B18、A6264706 | 中学生 $5/周零花钱推荐 $800+ 咖啡机；用户说"need more paychecks"但推全线 Apple 旗舰 |

### 品牌层级分布

**Copilot 倾向：**
- Explicit Journey：精准匹配用户对话中的品牌层级
- Related/Trending/All-in-one Journey：显著向高端偏移，经常脱离用户实际消费能力

**SLM 倾向：**
- 在标准消费品类（女装、电子、家居）中品牌定位较为合理
- 在小众/专业品类（sashiko 刺绣、Paparazzi 直销、healing gemstone）中品牌理解不足
- 系统性偏向大众品牌（Kohl's、Nordstrom Rack），即使用户浏览了高端渠道

---

## 7. 两条管线优劣势对比

| 维度 | Copilot 优势 | SLM 优势 |
|------|-------------|----------|
| **意图理解深度** | ✅ 从对话中提取预算、场景、品牌偏好、使用目的等深层信号 | ❌ 只能从浏览行为推断，无法获取深层意图 |
| **品类覆盖广度** | ✅ 平均 12.6 Journey/用户，related/trending 有效扩展 | ❌ 平均 6.2 Journey/用户，但更聚焦 |
| **产品匹配精准度** | ❌ 无 match_type 数据，部分用户出现系统级空产品 bug | ✅ 多个 Journey 达到 90-100% exact 匹配率 |
| **品牌特异性** | ✅ 从对话中精确锁定品牌（Express Portofino、Polar H9、OAS） | ❌ 倾向品类泛化，品牌需求被替换为竞品 |
| **价格定位** | ❌ Trending Journey 价格常偏离用户能力 10-50 倍 | ⚠️ 多数合理，但小众品类偏差大 |
| **Seller 权威性** | ✅ 渠道专业度高（Etsy古董、Seattle Coffee Gear） | ❌ 系统性降级为大众渠道（Kohl's、Nordstrom Rack） |
| **用户画像** | ✅ 能识别性别、预算、使用场景等 | ✅ 能从浏览模式识别品类偏好和尺码需求 |
| **时效性** | ❌ 依赖用户主动发起对话 | ✅ 实时捕捉最新浏览行为 |
| **Journey 类型多样性** | ✅ explicit/related/trending/all_in_one 四种类型 | ❌ 主要是 explicit + 少量 related |
| **抗"过度扩展"** | ❌ 从 1 条搜索生成 7-9 Journey，产品严重重复 | ✅ Journey 数量与数据量更匹配 |

---

## 8. 共性问题模式

### P0 — 系统级故障，需立即修复

| 问题 | 影响用户 | 描述 |
|------|---------|------|
| **Copilot 产品数组全空** | 3BC2C939 | 14 个 Journey 全部零产品推荐，系统级 bug |
| **SLM TID 品牌虚构** | CE04DD86 | 婴儿背带 Journey 12 个 TID 品牌全部虚构（Babylondon、Zazamalls），0% 匹配率 |
| **SLM TID seller 位退化** | 00DB9B18、AA7F1661、FF1353FD 等 | TID index 5 被产品属性值占据（颜色、型号、材质），seller 维度失效 |

### P1 — 严重质量问题，需优先解决

| 问题 | 影响用户 | 描述 |
|------|---------|------|
| **Copilot trending 价格严重偏离** | 1AFEFCE5、6614E456、00DB9B18 | Trending Journey 价格偏离用户消费能力 10-50 倍 |
| **SLM 小众品类产品目录覆盖不足** | 00DB9B18（SodaStream 18.2%）、6614E456（理发配件 9.1%）、AA7F1661（美甲 37.5%）、58767B1A（Hello Kitty 46.2%） | 小众/专业品类 TID 匹配率低于 50% |
| **SLM 语义偏移** | 58767B1A（healing→fashion）、3BC2C939（treats→food）、E4785BFA（Supreme联名→普通TNF） | TID 生成偏离用户搜索的实际语义 |
| **SLM retailerPreferences 未被利用** | FF1353FD | Profile 中明确记录 Neiman Marcus/Bergdorf Goodman，TID 生成时被替换为 Nordstrom Rack |
| **Copilot Journey 过度扩展** | 0BD7ADCE（1→7）、58767B1A（2→9）、6614E456（1→7） | 从 1-2 个信号过度扩展为 7-9 个 Journey，产品大量重复 |
| **Copilot Journey 遗漏** | 00DB9B18（打印机）、CE04DD86（家居） | 有明确多轮对话但未生成独立 Journey |

### P2 — 优化项，提升体验

| 问题 | 影响用户 | 描述 |
|------|---------|------|
| **SLM 品类泛化丢失品牌特异性** | 1745833B（Express→通用blouse）、CE04DD86（sashiko→通用刺绣） | 用户有明确品牌偏好但被泛化 |
| **SLM 浏览品类未生成 Journey** | 0BD7ADCE（除油剂 20+ 条）、6614E456（女性内衣 8 条）、AA7F1661（男装） | 有足够浏览信号但未生成 Journey |
| **用户身份识别** | 3BC2C939 | 用户是 Paparazzi 销售顾问，两端都未识别"卖家"身份 |
| **SLM 冷启动能力不足** | 1745833B（2 条浏览→空白 Profile） | 极少浏览事件时 Profile 为空，无法提供冷启动策略 |
| **Copilot 未体现价格敏感度** | 00DB9B18、A6264706 | 对话中有明确预算约束但 related/trending 完全忽略 |

---

## 9. 改进建议

### 针对 SLM

| 优先级 | 建议 | 影响范围 |
|--------|------|---------|
| **P0** | 修复 TID 品牌虚构 bug — 当模型不确定品牌时应留空或使用用户历史中的真实品牌 | CE04DD86 等 |
| **P0** | 修复 TID index 5 seller 位退化 — 确保 seller 字段始终填入有效零售商名称 | 5+ 用户 |
| **P1** | 扩充小众品类产品目录覆盖（美甲、SodaStream、VoIP 电话、理发配件、Hello Kitty 联名、婴儿背带） | 6+ 用户 |
| **P1** | TID 生成时参考 Profile 中的 retailerPreferences 作为 seller 约束 | FF1353FD 等 |
| **P1** | 增强联名/限量款识别能力（Supreme×TNF 等），区分普通品牌和联名品牌需求 | E4785BFA |
| **P1** | 当用户搜索包含具体品牌名时，TID 应优先包含该品牌（Ello、Rancilio 等） | 58767B1A、E4785BFA |
| **P2** | 提升从浏览信号中的品类覆盖率 — 高频浏览品类（≥5 条）应生成 Journey | 0BD7ADCE、6614E456 等 |
| **P2** | 从极少事件中最大化信息提取 — 即使仅 2 条事件，Profile 也不应为空 | 1745833B |
| **P2** | 考虑冷启动策略 — 浏览事件 <5 条时，结合 trending 生成探索性 Journey | 多数用户 |

### 针对 Copilot

| 优先级 | 建议 | 影响范围 |
|--------|------|---------|
| **P0** | 修复产品召回为空的系统级 bug | 3BC2C939 |
| **P0** | 修复 Journey 遗漏 — 有明确多轮购物对话的品类必须生成独立 Journey | 00DB9B18（打印机）、CE04DD86（家居） |
| **P1** | Trending Journey 价格锚定 — 参考用户 explicit Journey 的价位范围，限制 trending 的价格偏离倍数 | 1AFEFCE5、6614E456、00DB9B18 |
| **P1** | 控制 Journey 过度扩展 — 从 1 个信号最多衍生 1 explicit + 2 related，避免产品重复 | 0BD7ADCE、58767B1A、6614E456 |
| **P1** | 融入价格敏感度信号 — 从对话中提取预算约束并应用于 related/trending 的产品筛选 | 00DB9B18、A6264706 |
| **P2** | 区分"文案生成"与"购物意图" — 用户粘贴产品描述要求撰写帖文时，不应视为购物信号 | 3BC2C939 |

### 整体建议

| 建议 | 描述 |
|------|------|
| **两端信号融合** | 12 个用户中 5 个（41.7%）两端 Journey 零匹配。融合对话意图信号和浏览行为信号可显著提升推荐覆盖率 |
| **用户角色识别** | 增加"卖家/代购/顾问"等角色标签（如 3BC2C939 的 Paparazzi 顾问身份），区分进货浏览和消费浏览 |
| **产品目录优先扩展清单** | 美甲（GAOY 等专业品牌）、婴儿背带（BABYBJÖRN/Lillebaby）、SodaStream、VoIP 电话、Hello Kitty 联名、Supreme 联名、sashiko 专用产品 |

---

## 10. 附录：用户分析索引

| 文件名 | 用户核心特征 | 关键发现 |
|--------|------------|---------|
| user_00DB9B18_analysis.md | 年轻学生，Cisco PA 系统爱好者，预算极低 | Copilot 遗漏打印机 Journey；SLM SodaStream 18.2% 匹配率；Copilot 推荐 $800 咖啡机给 $5/周零花钱学生 |
| user_0BD7ADCE_analysis.md | 女性用户，商业清洁+女装+泳衣 | L1 差异最大 — Copilot 仅 1 条搜索生成 7 Journey vs SLM 128 条浏览；SLM 女装 100% exact |
| user_1745833B_analysis.md | 活跃女性购物者，15+ 品类 | SLM 仅 2 条浏览 → 2 Journey vs Copilot 20 个；SLM 冷启动问题严重 |
| user_1AFEFCE5_analysis.md | 工程公司行政人员，办公翻新+手包 | 唯一 HIGH 匹配（公告板）；Copilot trending 手包价格偏离 10x |
| user_3BC2C939_analysis.md | Paparazzi 饰品销售顾问 | Copilot 14 Journey 全部零产品（系统 bug）；SLM 100% exact 匹配率但价格偏高 3-17x |
| user_58767B1A_analysis.md | 兴趣分散（STEM/Hello Kitty/水壶） | 两端零匹配；SLM Hello Kitty 46.2%、healing gemstone 语义偏移 |
| user_6614E456_analysis.md | 退伍军人，球鞋/古董表 | 两端零匹配；Copilot 1 对话扩展 7 腕表 Journey；SLM 理发配件 9.1% |
| user_A6264706_analysis.md | 科技爱好者，Apple 生态+DIY PC | SLM Apple 全线 100% 匹配率；两端用户画像冲突（预算有限 vs high-tier） |
| user_AA7F1661_analysis.md | 多元兴趣（护肤/咖啡/美甲/LEGO） | 两端零匹配，互补性最强案例之一；SLM 美甲 37.5% |
| user_CE04DD86_analysis.md | 新妈妈/刺绣爱好者/女装 | SLM 婴儿背带品牌全虚构（P0）；刺绣入门 vs 进阶偏差 |
| user_E4785BFA_analysis.md | 男性生活方式（泳裤/厨具/Supreme/咖啡） | 两端零匹配；SLM 咖啡研磨机 Journey 质量极高；Supreme 联名理解不足 |
| user_FF1353FD_analysis.md | 设计师品牌女装+吊灯 | 互补性最强用户（3 个 HIGH 匹配）；SLM seller 系统性降级（Neiman Marcus→Nordstrom Rack） |

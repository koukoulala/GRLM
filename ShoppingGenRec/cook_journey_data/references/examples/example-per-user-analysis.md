# 用户 CE04DD86FA7F55FFA94D4B98FFFFFFFF 对比分析

**Copilot PicassoId:** fEXcQ9zsp7Xczi2mUFrKG
**SLM StableId:** CE04DD86FA7F55FFA94D4B98FFFFFFFF
**分析日期:** 2026-04-23

---

## 一、用户历史对比

**Copilot 端（Conversation History + Profile）：** 非常丰富，30 组购物对话 + 239 组非购物对话。购物兴趣覆盖面极广：女装（开衫/cardigans、针织衫/knit tops、牛仔上衣/denim tops、连衣裙/dresses、裤子/pants、腰带/belts）、手工（刺子绣 sashiko 丝线/套件）、包袋（手提包/satchel、挎包/shoulder bag、托特包/tote）、电子（iPhone、降噪耳机/noise-cancelling headphones）、鞋类（跑鞋/running shoes、越野鞋/trail sneakers）、家居（被褥/bedspreads、柜子/cupboards、空气净化器/air purifiers）、新妈妈礼物、护肤品/skincare、搅拌机/blender 等。Profile 记录了 38 个品类偏好和 13 个品牌偏好，风格标签 "moody-luxe, business-casual, trail-inspired streetwear"。

**SLM 端（Recent Shopping Events + Profile）：** recentShoppingEvents 有 84 条浏览记录（**非空**），覆盖：婴儿背带/baby carriers（BABYBJÖRN、Lillebaby、Happy!）、女装（Stitch Fix 上衣/blouses、Macy's 连衣裙/dresses 和裤子/pants、Nordstrom 上衣/tops）、刺子绣/sashiko（DARUMA 线、eBay/Etsy 套件）、家居装饰/home decor（花盆/planters、床头柜/nightstands、柜子/cabinets）、童书/children's books 等。Profile 记录品类偏好 4 个（女装、婴儿背带、刺绣用品、家居装饰），品牌偏好 8 个。

**历史覆盖差异总结：** SLM 的浏览记录集中在**近期实际行为**（1-28 天前），聚焦于婴儿用品、女装日常购物、刺绣手工、家居装饰这 4 个核心品类。Copilot 的对话历史覆盖时间跨度更长（3-90 天前），且包含大量**主动搜索意图**（如 "trail-inspired streetwear footwear"、"moody-luxe cardigans"、"Coach satchel handbags"），涵盖的品类远多于 SLM（额外有越野鞋、开衫、iPhone、耳机、空气净化器、跑鞋、口红等）。

---

## 二、Journey 语义匹配结果

### Journey 概览

| 来源 | Journey 数 | 类型分布 |
|------|-----------|---------|
| Copilot | 15 | explicit: 6, related: 4, trending: 3, all_in_one: 2 |
| SLM | 7 | explicit: 5, related: 2 |

### 语义匹配矩阵

| Copilot Journey | SLM Journey | 匹配度 | 说明 |
|---|---|---|---|
| J3: Silken threads for your sashiko dreams（刺子绣丝线） | J4: Sashiko kits that make stitching feel like art（刺子绣套件） | **HIGH** | 同为刺子绣/刺绣品类，但聚焦不同：Copilot→丝线，SLM→套件 |
| J10: Embroidery scissors that feel like jewelry（刺绣剪刀） | J4 同上 | **MEDIUM** | 都是刺绣配件，但剪刀 vs 套件是不同子品类 |
| J5: Light denim tops that skip the shirt vibe（浅色牛仔上衣）+ J6: Split-neck knits you'll live in（开领针织衫） | J2: Tops and dresses that turn errands into runways（日常穿搭上衣和连衣裙） | **HIGH** | 都是女装上衣/连衣裙品类。Copilot 拆分为两个 journey，SLM 合并为一个 |
| J4: Keepsakes for her first year of motherhood（新妈妈纪念礼物） | J1: Baby carriers that make every stroll a joyride（婴儿背带） | **MEDIUM** | 都源于"新妈妈"用户身份，但产品品类完全不同：纪念品/护理套装 vs 婴儿背带 |
| J15: Moody-luxe workday uniform in one go（暗黑奢华职场搭配，含裤子） | J3: Pants that work as hard as you do（百搭裤装） | **MEDIUM** | Copilot 将裤子融入全套搭配 journey，SLM 独立成裤装专题 |
| J1: Trail sneakers that own the city streets（越野街头运动鞋） | J7: Shoes that keep up with your new pants（搭配裤装的鞋） | **NO MATCH** | 越野运动鞋 vs 职场高跟鞋/平底鞋，品类和场景完全不同 |

### Copilot 独有 Journey（9 个）

| Journey | 品类 | 缺失原因 |
|---------|------|---------|
| J1: Trail sneakers that own the city streets（越野街头运动鞋） | 运动鞋 | SLM 浏览记录中无越野鞋记录 → **L1** |
| J2: Cardigans with a moody-luxe twist（暗黑奢华风开衫） | 女装开衫 | SLM 浏览记录中仅有上衣/裙子，无开衫 → **L1** |
| J7: Leather totes that mean business（商务皮革托特包） | 包袋 | SLM 浏览记录中无包袋记录 → **L1** |
| J8: Satchel handbags with quiet confidence（低调手提包） | 包袋 | 同上 → **L1** |
| J9: Trail-ready jackets for urban explorers（越野城市夹克） | 外套 | SLM 无越野外套浏览 → **L1** |
| J11: Vegan lipsticks everyone's raving about（纯素口红） | 美妆 | 来自 Copilot Profile 的 trending journey，SLM 无此信号 → **L1** |
| J12: Noise-cancelling headphones stealing the spotlight（降噪耳机） | 电子 | 同上 → **L1** |
| J13: Spring's obsession: pleated dresses（百褶裙） | 女装 | 同上 → **L1** |
| J14: From trail kicks to full street kit（越野街头全套搭配） | 搭配 | SLM 无越野品类基础 → **L1** |

### SLM 独有 Journey（2 个）

| Journey | 品类 | 说明 |
|---------|------|------|
| J5: Home accents that make a room feel finished（家居装饰点缀） | 家居 | Copilot 对话历史中有大量家居搜索（如 Group 21 关于 wooden cupboard 的 14 轮对话），但**未生成独立家居 journey**，这是 Copilot 的遗漏 |
| J6: Baby essentials that match your carrier game（婴儿配套用品） | 婴儿 | 背带 journey 的衍生 related journey |

---

## 三、相似 Journey 深入对比

### 匹配对 1：刺子绣/刺绣（HIGH）

**Copilot J3 "Silken threads for your sashiko dreams"（刺子绣丝线）**
vs **SLM J4 "Sashiko kits that make stitching feel like art"（刺子绣套件）**

| 维度 | Copilot | SLM |
|------|---------|-----|
| **聚焦方向** | 丝线（silk thread）— 用户搜索 "sashiko embroidery silk thread" 的精确延伸 | 套件（embroidery kit）— 从用户浏览 sashiko kit 记录泛化 |
| **Query/TermID** | Queries: "silk sashiko embroidery thread jewel tones", "best silk thread for Japanese sashiko" → 精准锁定丝线 | TID: ['thread', 'sewing', 'sashiko', 'DARUMA'], ['kit', 'embroidery', 'stamped', 'floral'] → 覆盖线和套件，但主体偏向通用刺绣套件而非 sashiko 专用 |
| **产品相关性** | KINKAME 日本丝线套装（$50）、Naturally Dyed Silk Floss（$16）、Gloriana Silk Floss → **精准匹配**用户对丝质 sashiko 线的搜索 | wtisan 4-Pack Embroidery Kit（$26）、Hobby Lobby $3-5 刺绣套件 → **泛化**为通用刺绣入门套件，与 sashiko 专业需求不符 |
| **品牌/卖家权威性** | Etsy 手工卖家（KINKAME、Treenway Silks）→ 刺绣手工领域的专业渠道 | Amazon、Hobby Lobby → 大众零售渠道，入门级产品 |
| **价格区间** | $7-63，中位~$15 → 符合专业丝线的合理价位 | $3-30，中位~$5 → 入门套件价格，远低于专业用品 |
| **匹配质量** | N/A（Copilot 无 match_type） | exact=6, fuzzy=2, none=3 → 54.5% exact，**none 27% 偏高** |

**差异归因：**

- **L2（Journey 生成差异）：** 用户历史中同时包含 sashiko 线和 sashiko 套件浏览记录，Copilot 精准地从对话 "sashiko embroidery silk thread" 提取出丝线需求，SLM 则泛化为通用刺绣套件。SLM 的 journey 标题虽提到 "sashiko"，但 tid 和产品已偏离为通用 embroidery kit。
- **L3（TermID 生成差异）：** SLM 的 tid 中第一个就是 ['thread', 'sewing', 'sashiko', 'DARUMA'] 但 match_type=none，说明 tid 虽然生成了正确的 sashiko 线描述，但产品目录中无法匹配到对应产品。后续 tid 转向通用刺绣套件来填充结果。
- **L4（Product 匹配差异）：** 即使在同一品类下，Copilot 推荐的是 Etsy 手工专业渠道的高端丝线，SLM 推荐的是 Hobby Lobby/Amazon 的 $3-5 入门套件，**反映出用户专业度判断的差异**——该用户有去京都购买 sashiko 用品的实际经历（见 Other Conversation Group 3/5/6/7），是进阶用户而非新手。

---

### 匹配对 2：女装上衣/连衣裙（HIGH）

**Copilot J5 "Light denim tops that skip the shirt vibe"（浅色牛仔上衣）+ J6 "Split-neck knits you'll live in"（开领针织衫）**
vs **SLM J2 "Tops and dresses that turn errands into runways"（日常穿搭上衣和连衣裙）**

| 维度 | Copilot | SLM |
|------|---------|-----|
| **Journey 粒度** | 拆分为 2 个 journey：牛仔上衣（非衬衫款）和开领针织衫 → 基于用户两次不同对话 | 合并为 1 个 journey：上衣+连衣裙 → 从浏览记录中聚合 |
| **Query/TermID** | J5: "women's light denim sleeveless tops", "non-shirt style denim blouses" → 精确到"非衬衫式牛仔上衣"; J6: "women's split neck knit tops" → 精确到开领 | TID: ['dress', 'fit-and-flare', 'midi', 'ruffle'], ['blouse', 'printed', 'split neck'] → 覆盖连衣裙+衬衫，与用户浏览的 Macy's/Nordstrom 产品一致 |
| **Copilot 产品** | J5: Splendid Denim Tank（$89）、Levi's Cami（$75）→ 牛仔背心/吊带; J6: Style & Co Split-Neck Top（$20-70）→ 开领针织 | — |
| **SLM 产品** | — | Ruffle Midi Fit & Flare Dress（$60）、Split Neck Blouse（$35）、V-Neck Fit & Flare Dress（$35）→ 连衣裙为主 |
| **品牌定位** | Splendid、Levi's、Steve Madden、Free People → 中高端休闲; Style & Co、Alfred Dunner → 中档 | Flying Tomato、CeCe、Donna Morgan、Maison Tara → 中档，Nordstrom Rack/belk 渠道 |
| **价格** | J5: $45-128; J6: $20-84 | $20-78，价格整体偏低 |
| **匹配质量** | N/A | exact=9/10（**90%**），匹配质量**优秀** |

**差异归因：**

- **L2（Journey 生成差异）：** Copilot 从对话意图中区分了 "denim tops but not shirt style" 和 "split-neck knit tops" 两个明确的子需求，拆分成两个 journey；SLM 从浏览记录中合并为一个泛化 journey。Copilot 的粒度更精准地反映了用户的具体偏好。SLM 的处理虽然粗一些，但 90% 的 exact 匹配率说明产品召回效果很好。
- **L4（Product 匹配差异）：** 两边品牌定位相近（都是中档），但 SLM 偏向连衣裙而 Copilot 偏向上衣，反映了 SLM 从浏览记录中连衣裙占比更高（DKNY Midi Dress、Robbie Bee Fit & Flare Dress 等）。

---

### 匹配对 3：新妈妈/婴儿（MEDIUM）

**Copilot J4 "Keepsakes for her first year of motherhood"（新妈妈纪念礼物）**
vs **SLM J1 "Baby carriers that make every stroll a joyride"（婴儿背带）**

| 维度 | Copilot | SLM |
|------|---------|-----|
| **品类方向** | 新妈妈礼物套装（护理品+记忆册+哺乳巾） | 婴儿背带 |
| **来源** | 对话："what to gift a new mother?" → 是**送礼场景** | 浏览记录：BABYBJÖRN、Lillebaby、Happy! 背带 → 是**自用对比购物** |
| **Copilot 产品** | Palmer's Body Butter（$30）、Baby Memory Book（$48-60）、Postpartum Care Box（$59-89） | — |
| **SLM 产品** | — | 全部 12 个产品 match_type=**none**，没有一个成功匹配 |
| **匹配质量** | N/A | **0% exact — 严重问题** |

**SLM 婴儿背带 Journey 的 TID 详情（全部 none）：**

| # | TID 品牌 | 是否真实品牌 | 用户实际浏览品牌 |
|---|---------|-------------|---------------|
| 1 | Babylondon | ❌ 虚构 | BABYBJÖRN |
| 2 | Zazamalls | ❌ 虚构 | Lillebaby |
| 3 | Bedtime Originals | ❌ 不相关（床品品牌） | Happy! |
| 4 | Klutch | ❌ 虚构 | — |
| 5 | Halo | ⚠️ 存在但非背带品牌 | — |
| 6 | Quince | ⚠️ 存在但非背带品牌 | — |
| 7-12 | Kaliuli, Munchkin, InfantGear, Babycocoon 等 | ❌ 大部分虚构 | — |

**差异归因：**

- **L1（用户历史差异）：** Copilot 捕捉的是对话中的"送礼"意图，SLM 捕捉的是浏览记录中的"对比购物"行为。两者虽然都源自"新妈妈"信号，但场景理解不同。SLM 的理解更准确——用户确实在比较背带。
- **L3（TermID 生成差异）- ⚠️ 严重问题：** SLM 为婴儿背带 journey 生成的 12 个 tid 中，品牌名**全部是虚构的或不相关的**（Babylondon、Zazamalls、Klutch 等），而用户实际浏览的品牌是 BABYBJÖRN、Lillebaby、Happy!。**tid 中的品牌信息完全错误**，导致 100% none 匹配。
- **L4（Product 匹配差异）：** 由于 tid 品牌全错，产品匹配阶段无从召回正确产品，这是一个 **L3 导致的 L4 连锁失败**。

---

### 匹配对 4：裤装（MEDIUM）

**Copilot J15 "Moody-luxe workday uniform in one go"（暗黑奢华职场搭配，含裤子）**
vs **SLM J3 "Pants that work as hard as you do"（百搭裤装）**

| 维度 | Copilot | SLM |
|------|---------|-----|
| **Journey 粒度** | 全套搭配（裤子+芭蕾鞋+托特包），裤子只是一部分 | 独立裤装 journey，专注 bootcut/high-rise |
| **Copilot 产品** | Ann Taylor Sophia Pant（$119）、Lee Chino（$55）、VIVAIA Ballet Flats（$109） → 裤子+鞋+包 | — |
| **SLM 产品** | — | INC Bootcut Pants（$45）、Levi's Wedgie Bootcut（$110）、L.L.Bean Bootcut（$60）、Banana Republic Bootcut（$100） → 全部是 bootcut 裤 |
| **精准度** | 裤子部分不够聚焦，混入了鞋和包 | 精准匹配用户 "bootcut high-rise" 的 fit 偏好 |
| **匹配质量** | N/A | exact=7, fuzzy=1, none=1 → **78% exact** |
| **价格** | $55-275（含鞋包） | $40-110（纯裤装） |

**差异归因：**

- **L2（Journey 生成差异）：** SLM 精准生成了裤装专题 journey，与用户浏览记录（Style & Co Bootcut、Banana Republic Bootcut、Athleta Bootcut）完全对应。Copilot 将裤子融入搭配 journey 而非独立品类。**SLM 在此 journey 上的表现优于 Copilot**。
- **L4：** SLM 产品匹配率高（78% exact），品牌和价位都与用户浏览历史一致。

---

### SLM 独有：Home accents（家居装饰）

**SLM J5 "Home accents that make a room feel finished"（家居装饰点缀）**

- 基于用户浏览的花盆（Tucker Ceramic Planter）、床头柜（Streamdale Nightstand）、柜子（Red Barrel Studio Cabinet）记录
- exact=8/11（**73%**），产品与用户浏览高度吻合
- 推荐产品：Ebern Designs Ceramic Planter（$41）、DESIGN ART Framed Wall Art（$176）、Mercer41 Nightstand（$95）、Crate & Barrel Planter（$43）
- Copilot 对话历史中有大量家居搜索（Group 21 关于 wooden cupboard 的 **14 轮对话**！），但**未生成独立家居 journey**，这是 Copilot 的明显遗漏

### SLM 独有：Baby essentials（婴儿配套用品）

**SLM J6 "Baby essentials that match your carrier game"（婴儿配套用品）**

- 背带 journey 的衍生 related journey，推荐睡袋/sleep sack、毯子/blanket、包被/swaddle
- exact=3, fuzzy=1, none=4 → 37.5% exact，匹配率偏低
- 产品中 Quince Bamboo Sleep Sack（$40）、Carter's Receiving Blanket（$32）质量尚可，但婴儿枕头类产品（Poloo 品牌）全部 none

### SLM 独有：Shoes（搭配鞋）

**SLM J7 "Shoes that keep up with your new pants"（搭配裤装的鞋）**

- 裤装 journey 的衍生 related journey，推荐职场高跟鞋/pumps 和平底鞋/flats
- exact=11/11（**100% exact**），匹配质量**完美**
- 产品：Jeffrey Campbell Archive Pump（$69）、LifeStride Flats（$80）、Naturalizer Pump（$60）、Bandolino Mary Jane（$89）
- 品牌以中档舒适鞋品为主（LifeStride、Naturalizer、Easy Street），与用户的 "business-casual" 风格一致

---

## 四、差异归因汇总

| 层级 | 出现次数 | 严重程度 | 具体问题 |
|------|---------|---------|---------|
| **L1 用户历史差异** | 8 处 | 高 | Copilot 对话历史覆盖面远超 SLM 浏览记录（30 组购物对话 vs 84 条浏览），导致 Copilot 独有 8+ 个 journey（越野鞋、开衫、包袋、iPhone、耳机、口红、百褶裙等） |
| **L2 Journey 生成差异** | 4 处 | 中 | ① 刺绣：Copilot→丝线 vs SLM→入门套件（对用户专业度判断不同）；② 上衣：Copilot 拆分 2 个 vs SLM 合并 1 个；③ 新妈妈：送礼 vs 自用；④ 裤装：Copilot 混入搭配 vs SLM 独立专题 |
| **L3 TermID 生成差异** | 2 处 | **严重** | ① 婴儿背带 journey 的 tid **品牌全部虚构**（100% none）；② 刺绣 journey 的 tid 偏向通用刺绣而非 sashiko 专用 |
| **L4 Product 匹配差异** | 2 处 | 中 | ① 刺绣：Copilot→Etsy 专业渠道高端丝线 vs SLM→Hobby Lobby $3 入门套件；② 上衣：两边品质相近但 SLM 偏连衣裙 |

---

## 五、关键发现

### SLM 的亮点 ✅

1. **裤装 journey**：精准匹配用户 bootcut/high-rise 偏好，78% exact，品牌价位与用户浏览一致
2. **家居装饰 journey**：Copilot 遗漏的品类，SLM 从浏览记录中准确识别并推荐，73% exact
3. **上衣/连衣裙 journey**：90% exact 匹配率，产品召回效果优秀
4. **搭配鞋 journey**：100% exact，完美匹配

### SLM 的严重问题 ❌

1. **婴儿背带 journey 的 tid 品牌全部虚构**：用户实际浏览 BABYBJÖRN、Lillebaby、Happy! 三个品牌，但 SLM 生成的 tid 中出现 Babylondon、Zazamalls、Klutch 等虚构品牌名，导致 0% 匹配率。这是 **L3 层面的系统性 bug**，疑似模型在不确定品牌名时倾向于"编造"而非使用用户历史中的真实品牌。
2. **刺绣 journey 偏向入门级**：用户是有京都购物经历的进阶刺子绣爱好者，但 SLM 推荐的是 Hobby Lobby $3-5 的入门套件，未能识别用户的专业度。

### Copilot 的亮点 ✅

1. **对话意图解析精准**：区分 "denim tops but not shirt style" 和 "split-neck knit tops" 两个子需求
2. **品类覆盖广**：15 个 journey vs SLM 的 7 个
3. **刺绣品类推荐精准**：通过 Etsy 专业渠道推荐日本 KINKAME 丝线，与用户的进阶水平匹配

### Copilot 的遗漏 ❌

1. **家居品类缺失**：有 14 轮对话讨论 wooden cupboard，但未生成家居 journey
2. **裤装未独立成 journey**：用户有明确的 bootcut 搜索行为（Group 22），但仅融入了 all_in_one 搭配 journey

---

## 六、改进建议

### 针对 SLM

1. **修复 tid 品牌虚构问题（P0）**：婴儿背带 journey 的品牌名全部虚构，应优先使用用户历史中的真实品牌（BABYBJÖRN、Lillebaby 等），当模型不确定品牌时应留空或使用通用描述而非编造
2. **提升用户专业度识别**：该用户有京都 sashiko 购物经历和多年刺绣经验，推荐入门套件不合适。建议从用户浏览的品牌层级（DARUMA 是专业品牌）和浏览深度推断用户水平
3. **扩大 sashiko 专用产品覆盖**：第一个 tid ['thread', 'sewing', 'sashiko', 'DARUMA'] 生成正确但匹配 none，说明产品目录中缺少 sashiko 专用产品

### 针对 Copilot

4. **覆盖家居品类**：用户有 14 轮 cupboard 对话但未生成 journey，需检查 journey 生成的品类覆盖逻辑
5. **独立高意图品类**：bootcut pants 有独立搜索行为，应生成独立 journey 而非仅融入 all_in_one 搭配

---

## 七、Seller Authority 根因分析

### 用户历史中的 Seller（L1）

**Copilot 端：** retailerPreferences 为空（`[]`），但对话中推荐的产品覆盖渠道广泛：Etsy（刺绣丝线）、Nordstrom/Macy's（女装）、Coach/Kate Spade（包袋）、Amazon 等。

**SLM 端：** retailerPreferences 记录为 `["Macy's", "Nordstrom", "Amazon", "Etsy", "eBay", "Pottery Barn Kids", "Stitch Fix"]`。浏览记录明确显示：婴儿背带来自 Pottery Barn Kids（BABYBJÖRN、Lillebaby）和 Amazon（Happy!）；女装来自 Macy's（DKNY、Style & Co）、Nordstrom（Wit & Wisdom、NYDJ）、Stitch Fix（Daniel Rainn）；刺绣来自 Amazon（DARUMA）、eBay/Etsy（sashiko kits）；家居来自 Macy's/Wayfair/Bloomingdale's。

### Seller 流转追踪（L1 → L3 → L4）

#### SLM Journey 1：Baby carriers that make every stroll a joyride（婴儿背带）

| TID Seller（L3，index 5） | L4 Matched Seller | 与 L1 一致？ | 说明 |
|---|---|---|---|
| Bella Luna Toys | — (none) | ❌ | 虚构 seller，用户浏览的是 Pottery Barn Kids |
| — (Zazamalls) | — (none) | ❌ | 虚构品牌+seller |
| belk | — (none) | ❌ | 用户未浏览 belk |
| Athleta | — (none) | ❌ | Athleta 不卖婴儿背带 |
| Babylist | — (none) | ❌ | 婴儿用品平台，但用户浏览的是 Pottery Barn Kids |
| Bloomingdale's | — (none) | ❌ | 不匹配 |

**⚠️ 严重问题：12 个 tid 全部 none，seller 全部虚构或不相关。** 用户实际浏览的 Pottery Barn Kids 和 Amazon 从未出现在 tid 的 seller 字段中。

#### SLM Journey 2：Tops and dresses that turn errands into runways（上衣/连衣裙）

| TID Seller（L3） | L4 Matched Seller | 与 L1 一致？ |
|---|---|---|
| — | Nordstrom Rack | ⚠️ 部分一致（用户浏览 Nordstrom，L4 是 Nordstrom Rack——折扣线） |
| — | belk | ❌ |
| — | Bloomingdale's | ⚠️ 用户浏览了 Bloomingdale's（家居），但非女装 |

#### SLM Journey 3：Pants that work as hard as you do（裤装）

| TID Seller（L3） | L4 Matched Seller | 与 L1 一致？ |
|---|---|---|
| — | Macy's | ✅ 用户浏览了 Style & Co @ Macy's |
| — | Levi's | ❌ 品牌 DTC |
| — | L.L.Bean | ❌ |
| — | H&M | ❌ |
| — | Banana Republic Factory | ⚠️ 用户浏览了 Banana Republic Factory bootcut |

#### SLM Journey 4：Sashiko kits（刺绣套件）

| TID Seller（L3） | L4 Matched Seller | 与 L1 一致？ |
|---|---|---|
| — | Amazon.com | ✅ 用户在 Amazon 浏览 DARUMA |
| — | Missouri Star Quilt Company | ❌ 刺绣专业渠道 |
| — | Hobby Lobby | ❌ 用户未浏览 |

#### SLM Journey 5：Home accents（家居装饰）

| TID Seller（L3） | L4 Matched Seller | 与 L1 一致？ |
|---|---|---|
| — | Wayfair | ✅ 用户浏览了 Wayfair（Red Barrel Studio Cabinet） |
| — | Bed Bath & Beyond | ❌ |
| — | Crate & Barrel | ❌ |
| — | Amazon.com | ✅ |
| — | Ashley | ❌ |

#### SLM Journey 7：Shoes that keep up with your new pants（搭配鞋）

| TID Seller（L3） | L4 Matched Seller | 与 L1 一致？ |
|---|---|---|
| — | Nordstrom Rack | ⚠️ 用户浏览 Nordstrom，非 Rack |
| — | Kohl's | ❌ 用户未浏览 Kohl's |
| — | Macy's | ✅ |

### 降级模式分析

**观察到四种降级模式：**

1. **婴儿背带 journey 的 seller 全面虚构**：用户浏览的是 Pottery Barn Kids 和 Amazon 的背带产品，但 tid 中的 seller 出现了 Bella Luna Toys（小众玩具店）、Athleta（运动服饰）、Bloomingdale's 等完全不相关的渠道。这与品牌虚构问题（Babylondon、Zazamalls）同源——**模型在生成 tid 时对婴儿背带品类的 seller 知识严重不足**。

2. **Nordstrom → Nordstrom Rack 降级**：用户浏览的是 Nordstrom 正价商品（Wit & Wisdom、NYDJ），但 L4 匹配到的多是 Nordstrom Rack（折扣线）。这意味着推荐的品牌档次可能偏低（Nordstrom Rack 偏向尾货/折扣品）。

3. **Kohl's 大量渗透**：鞋类 journey 中 Kohl's 出现频率极高（11 个 seller 中 Kohl's 占 5-6 个），但用户浏览历史中**从未出现 Kohl's**。Kohl's 是中低端百货，与用户浏览的 Macy's/Nordstrom 定位有落差。

4. **刺绣品类 seller 降级**：用户在 eBay/Etsy 浏览的是日本进口 sashiko 专业套件（Olympus、DARUMA），但 L4 匹配到的是 Hobby Lobby（大众手工连锁店），seller 档次从专业渠道降级到入门级渠道。

### Copilot vs SLM Seller 档次对比

| 品类 | Copilot Seller 档次 | SLM Seller 档次 | 差距 |
|------|---------------------|-----------------|------|
| 刺绣 | Etsy 手工专业卖家（KINKAME、Treenway Silks）→ **专业级** | Hobby Lobby、Amazon → **入门级** | ⚠️ 显著降级 |
| 女装上衣 | Splendid、Levi's、Free People → **中高端休闲** | Nordstrom Rack、belk → **中档折扣** | 轻微降级 |
| 裤装 | Ann Taylor、Lee → **中档** | Macy's、Banana Republic Factory → **中档** | 基本一致 |
| 婴儿 | 无对比数据 | 全部 none | — |
| 鞋类 | 无独立 journey | Kohl's、Nordstrom Rack、Macy's → **中档偏低** | — |

### 根因判断

1. **婴儿背带 journey 的 seller 虚构是 L3 层 bug（P0）**：与品牌虚构同源，模型在生成婴儿背带 tid 时，seller 字段也使用了虚构或不相关的值。用户实际浏览的 Pottery Barn Kids 完全缺失。**根因：模型对婴儿背带品类的 seller 知识库为空**。
2. **Nordstrom → Nordstrom Rack 降级是产品目录问题（L4）**：Bing Shopping 产品目录中 Nordstrom Rack 的 offer 数量可能远多于 Nordstrom 正价，导致 L4 匹配时优先命中 Rack。
3. **Kohl's 渗透是 L3/L4 层的 seller 选择偏差**：Kohl's 在鞋类品类中 offer 数量大，导致 tid 的 seller 和 L4 匹配均偏向 Kohl's，与用户实际购物渠道偏好不符。
4. **刺绣品类 seller 降级是品类理解不足（L2→L3）**：SLM 将 sashiko 专业需求泛化为通用刺绣入门，seller 也相应从 Etsy/eBay 专业渠道降级为 Hobby Lobby/Amazon。根因在 L2 的品类理解，seller 降级是其连锁效应。

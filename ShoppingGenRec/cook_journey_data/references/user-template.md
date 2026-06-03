# 用户分析模板

**分析视角：对称对比。** P1 和 P2 用<strong>同一套维度</strong>评估，每段分析都列出两端各自的表现与差距，由数据决定胜负。

**语言规范：** 分析论述用中文；源数据（journey title、product name、query、reason、seller 等）保留英文原文并附中文含义/翻译，方便理解。示例：
- Journey: "Your next run starts with the right pair"（你的下一次跑步从对的鞋开始）
- Product: "Brooks Ghost 16"（Brooks Ghost 16 跑鞋）
- Query: "women's midi belted shirtdress L'Agence"（L'Agence 品牌女款系腰带中长衬衫裙）
- Reason: "You browsed multiple running shoes from Nike and HOKA"（你浏览了多款 Nike 和 HOKA 跑鞋）

每个 Deep 用户生成的 md 文件遵循此结构：

```markdown
# 用户 [ID] 对比分析

> **分析视角：** P1 vs P2 对称对比，列出两端各自的优势与差距

## 零、数据校验（分析前必须完成）

> **校验结果（与 paired_data.json 预计算值比对）：**
> - P1 Journey 数：X 个（paired_data.p1.journey_count: X ✅ / 不一致 ❌ → 停止分析）
> - P2 Journey 数：Y 个（paired_data.p2.journey_count: Y ✅ / 不一致 ❌ → 停止分析）
> - P2 浏览记录数：Z 条（**直接引用 paired_data.p2.recent_events_count，禁止自行计算**）
> - P1 / P2 Profile retailerPreferences：[列表] / 空

**校验方法：**
1. 从 paired_data.json 读取该用户的 `p1.journey_count`、`p2.journey_count`、`p2.recent_events_count` 预计算值
2. 自行计算 `len(p1.journeys)` 和 `len(p2.journeys)` 并与预计算值比对
3. **浏览记录数直接引用 `p2.recent_events_count` 预计算值，不要自行解析字符串计数**（该值由 Phase 1 Python 脚本精确计算，LLM 对长字符串计行极易出错）
4. 如任一数值不一致，**立即停止分析并报告数据问题**，不要继续后续章节

## 一、用户历史对比
概括性总结两端历史信息的覆盖范围和差异（不逐条列出）。强调"同一用户的两个 history 切片"的联合视角。

## 二、Journey 语义匹配结果
- HIGH = 同一具体产品品类；MEDIUM = 同子品类不同角度；NO MATCH = 不同品类
- 匹配矩阵 + 各端独有 journey 及原因（P1 独有 / P2 独有 各列一栏）

## 三、相似 Journey 深入对比（仅 HIGH/MEDIUM）
- Query/TID 生成准确性对比（两端各自评估）
- 产品对比：相关性、popularity、seller/brand authority、价格区间
- 匹配质量统计（Schema C: exact/fuzzy/none 分布；Schema B: 召回相关性人工判定）
- **核心分析对象**：Journey title/description/reason + 产品 query/name 质量 + 召回产品的相关性、品牌、seller、价格
- **可选辅助字段（仅当数据中存在且非空时使用）**：Schema B 的 `stats.reasoning` / `ANNScore` / `IsFiltered` / `FilterReason` 若有则作为补充证据

## 四、差异归因（L1-L4）
**对称归因**：P1 和 P2 各自在哪个 layer 出现什么问题，分别点出。
- 每个差异点需要 **L1→L3→L4 证据链**：具体数据流转
- 不要预设单边视角；如果某 layer 是 P1 占优，明说

## 五、关键发现
分两个对称的子节：
### 5.1 P1 优势 / P2 差距
列出 P2 在哪些维度落后于 P1，附具体案例
### 5.2 P2 优势 / P1 差距
列出 P1 在哪些维度落后于 P2，附具体案例

> 命名上保持对称：不要用"反向优势"这种暗示某一端是基线的措辞。

## 六、改进建议
对两端对称提建议：P1 可优化什么、P2 可优化什么。

## 七、Seller Authority 根因分析
- 用户历史中的 Seller（L1）
  - 从 P1 / P2 各自的 history 中提取浏览过的 seller
  - 从 Profile.retailerPreferences 提取偏好 seller
- Seller 流转追踪（L1 → L3 → L4）：per-journey 表格，<strong>P1 和 P2 各占一列</strong>
  - Schema A: L3 隐含在 queries[]，L4 = products[].seller
  - Schema B: L3 = query 中可能含 seller / 品牌关键词，L4 = matched_products[].Seller
  - Schema C: L3 = TID[5] (seller slot)，L4 = matched_products[].Seller
- 降级或升级模式分析（全价↔折扣、垂直↔大众、官网↔第三方）
- 根因判断（L1/L3/L4 哪一层导致差异，分别对 P1 和 P2 判断）
```

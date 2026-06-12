## Role
Expert Product Summarizer.

## Objective
Validate whether the input describes a real, specific product, then generate a concise 7-slot summary that captures the product's key searchable attributes. The task proceeds in two sequential steps: validation first, then summarization.

## Step 1 — Product Validation (Gate — must pass before summarization)

Carefully analyze the PRODUCT INFORMATION below and determine whether it describes a real, specific product. A real, specific product is a concrete item, service, software product, tool, food item, appliance, accessory, etc. that a customer could reasonably buy or use.

**NOT a real, specific product** includes:
- A general marketing slogan or brand campaign phrase
- A vague browsing or search phrase
- A category name without a concrete product
- A store page title or collection name
- A brand name alone
- Text too vague to identify an actual product

If the input does NOT describe a real, specific product, output ONLY:
`<Output>[]</Output>`

## Step 2 — Summarization (7 Slots)

Generate exactly **7 slots**: 5 product-description slots + Brand + Seller.

### Slot Format

- Prefer **single descriptive words** or short hyphenated words.
- Multi-word phrases are allowed ONLY for model names, brand names, seller names, and fixed proper nouns.
- Each phrase counts as ONE slot.

### Slots 1-5 — Product Description (broad to specific)

Use 5 slots to describe the product from broad to specific. Think of them as the 5 search terms a shopper would type to find this exact product. Test: if someone reads only these 5 words (without brand or seller), could they distinguish this product from other similar products? If not, replace the least distinctive word with a more specific one.

**Slot 1** is always the product category (the broadest useful term).

**Slots 2-5** are flexible — pick the 4 most informative, distinguishing attributes from the product information. Consider all of the following attribute types and include whichever are most important for this specific product:

- **Model name or product line** (e.g., iPhone 17 Pro Max, Technic BMW M4 GT3 EVO, Saros 20, WH-1000XM6, Nanit Pro, Caseta, Kindle Paperwhite). Keep model names compact: drop brand prefix and non-essential suffixes like color/storage/SKU. One phrase slot, never split across slots.
- **Form, subtype, or subject** — what form it takes or what it represents (e.g., over-ear, midi, race-car, cordless, platform, 3-in-1, pendant, castle)
- **Style or technique** — the defining style or method (e.g., button-down, hand-blown, beaded, double-breasted, noise-canceling)
- **Material or fabric** (e.g., cotton, linen, leather, ceramic, stainless, glass, titanium, denim, satin)
- **Color or pattern** (e.g., deep-blue, cosmic-orange, striped, navy, floral, plaid)
- **Size, capacity, or key spec** (e.g., 256gb, 1tb, 65-inch, 747-piece, 6-person, 28l, 140w)
- **Gender or target audience** (e.g., women, men, kids, baby, pet, professional, petite). Exclude generic values like "unisex" or "adult".
- **Occasion or use scenario** (e.g., evening, outdoor, backpacking, workwear, gaming, desk, kitchen)
- **Key function or feature** (e.g., unlocked, rechargeable, self-emptying, dimmable, waterproof, foldable, smart)

**Filling principles:**
- Maximize information density — every slot should carry a concrete, searchable word that helps identify or distinguish the product.
- Spread across different attribute types for maximum coverage. Do not pick 4 attributes of the same type.
- Do NOT repeat the category in different words (chandelier and "illuminating", vacuum and "cleaning" are redundant).
- No vague fillers: never use flagship, premium, classic, modern, basic, general, other, none.
- Only include attributes that are explicitly stated or clearly supported by the product information. Do not guess.

### Slot 6 — Brand

MUST include if explicitly provided. Keep original form — do not translate or rewrite.
If the brand is "Other", "Unknown", or blank, use a remaining useful product attribute instead.

### Slot 7 — Seller / Retailer

MUST include if explicitly provided and different from brand. Keep original form.
If brand and seller are the same, do NOT repeat — use a remaining useful product attribute instead.

## Rules

### Word Style
Prefer single words. Use hyphens when natural (noise-canceling, deep-blue, self-emptying). Phrase slots only for model names, brand names, seller names, and proper nouns.

### Language
Use the product's primary language for descriptive slots. Brand names, seller names, model names, and proper nouns must remain in their original form and untranslated.

### English Word Form
Use base/dictionary form when natural (singular nouns, no unnecessary -ing/-ed). Do NOT alter proper nouns or model names.

### No Duplicates
All 7 slots must be different. No hypernym+hyponym pairs (do not output both "coat" and "outerwear", or both "shoes" and "footwear").

### No Guessing
Do not invent missing facts. Only include attributes explicitly stated or clearly supported by the product information.

### Similar Items
Use the TOP 5 SIMILAR PRODUCTS only as reference for terminology consistency. Prefer accurate description of the current product over forced consistency.

## Output Examples

`<Output>[smartphone, iPhone 17 Pro Max, unlocked, 256gb, deep-blue, Apple, Best Buy]</Output>`
`<Output>[toy, Technic BMW M4 GT3 EVO, race-car, 747-piece, kids, LEGO, Scheels]</Output>`
`<Output>[vacuum, Saros 20, robotic, self-emptying, pet, Roborock, Best Buy]</Output>`
`<Output>[shirtdress, button-down, midi, cotton, women, Evi Grintela, Neiman Marcus]</Output>`
`<Output>[chandelier, hand-blown, pendant, glass, art-deco, Rejuvenation, Frontgate]</Output>`
`<Output>[sneaker, platform, low-top, cushioned, women, Converse, Nordstrom Rack]</Output>`

Non-product: `<Output>[]</Output>`

## Final Check Before Output

1. Is this a real, specific product? If not, output `[]`
2. Are there exactly 7 slots?
3. Do Slots 1-5 together distinguish this product from similar ones? If someone reads only these 5 words, can they tell which specific product it is?
4. All 7 slots distinct? No vague fillers?
5. Brand in Slot 6, Seller in Slot 7 (if available and different)?
6. Proper nouns intact and untranslated?
7. No extra text outside `<Output></Output>`?

## Output Format

Output ONLY the result. No explanations. No extra text.

If NOT a product:
```
<Output>[]</Output>
```

If a product:
```
<Output>[slot1, slot2, slot3, slot4, slot5, slot6, slot7]</Output>
```

## Inputs

PRODUCT INFORMATION:
{product_info_text}

TOP 5 SIMILAR PRODUCTS (for reference only):
{similar_items_text}

Output:

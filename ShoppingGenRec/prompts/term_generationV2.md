## Role
Expert Product Summarizer.

## Objective
Validate whether the input describes a real, specific product, then generate a concise 7-slot summary capturing the product's key attributes. The task proceeds in two sequential steps: validation first, then summarization.

## Step 1 — Product Validation (Gate — must pass before summarization)

Carefully analyze the PRODUCT INFORMATION below and determine whether it describes a real, specific product. A real, specific product is a concrete item, service, software product, tool, food item, appliance, accessory, etc. that a customer could reasonably buy or use.

**NOT a real, specific product** includes:
- A general marketing slogan
- A brand slogan or campaign phrase
- A vague browsing or search phrase
- A category name without a concrete product
- A store page title or collection name
- A brand name alone
- Text too vague to identify an actual product

If the input does NOT describe a real, specific product, output ONLY:
`<Output>[]</Output>`

## Step 2 — Summarization (7 Slots)

If Step 1 confirms the input is a real, specific product, generate exactly **7 summary slots** to summarize it.

### Slot Format

- Each slot should normally be **ONE WORD**. Prefer **single descriptive words** whenever possible.
- Only use a **multi-word phrase** when it cannot be split safely, such as:
  - Brand names
  - Retailer names
  - Fixed proper nouns
  - Platform names
- Example phrase slots (each counts as ONE slot):
  - "home depot"
  - "joss & main"
  - "楽天市場"

### Priority Order for the 7 Slots

Each slot has a designated role. Fill all 7 slots in order:

**Slot 1 — Product Category**
Main product category or type. Use the most specific single-word category that accurately describes the product. Be consistent: use the same category word for similar products (e.g., always "shirtdress" or always "dress" for shirt dresses — do not mix).
Examples: headphone, sneaker, cream, laptop, shirtdress, television, tent

**Slot 2 — Product Identity: Model Name OR Primary Style/Function**
This slot answers “what specifically is it?” beyond the category.
- If the product has a specific model name or product line, it MUST be here as a phrase slot. E.g., iPhone 17 Pro Max, WH-1000XM6, Technic BMW M4 GT3 EVO, Saros 20, V15 Detect, Kindle Paperwhite, Roomba j9+, Instant Pot Duo, Caseta, Mad House 6.
- If not, use the defining sub-style, technique, or type that a shopper would search for. E.g., hand-blown (chandelier), button-down (shirtdress), robotic (vacuum), beaded (clutch), kitten (heel), double-breasted (blazer), cabin (tent), noise-canceling (headphone).
- Do NOT repeat the category in different words (chandelier→"illuminating", vacuum→"cleaning", clutch→"carrying"). These add no information.

**Slot 3 — Structural Form / Subtype**
This slot answers "what form, subtype, or subject?" — the physical shape, subtype, or what the product represents. Pick the term a shopper would use to filter or search. When both "how it's made" and "what it is/represents" are options, prefer what it represents.
- Electronics: wireless, cordless, robotic, usb-c, over-ear, in-ear, 3-in-1, all-in-one
- Clothing: midi, maxi, ankle, sleeveless, short-sleeve, wide-leg, a-line, bootcut
- Shoes: low-top, platform, slip-on, lace-up, open-toe, block-heel
- Home: countertop, clip-on, wall-mounted, bedside, freestanding, pendant
- Toys/Art/Decor: race-car, castle, spaceship, floral, landscape, abstract
- Accessories: hoop, crossbody, sling, trapezoid, envelope
- Outdoor: 2-person, 6-person, 28l, ultralight

**Slot 4 — Distinguishing Attribute**
The attribute that most differentiates this product from similar items within the same subtype. Color, size, storage, material, pattern, or key spec all belong here.
Examples: 256gb, 65-inch, titanium, striped, lightweight, waterproof, stainless, hepa, linen, self-emptying, dimmable, ceramic

**Slot 5 — Audience / Context / Additional Attribute**
Fill this slot using the following priority (first match wins):
1. **Target audience** if explicitly stated: kids, women, men, professional, baby, pet. (Exclude generic "unisex" or "adult".)
2. **Use scenario or occasion** if clear: outdoor, gaming, office, evening, backpacking, kitchen, desk.
3. **Any remaining useful product attribute**: color, material, function, pattern, or spec not yet used in Slots 3–4. A concrete attribute (e.g., deep-blue, cotton, noise-canceling, 1TB) is always better than a vague filler.

Do NOT use vague quality tiers (flagship, premium, classic, modern, basic) — they add no useful information.

**Slot 6 — Brand**
Brand or platform ecosystem.
If brand is explicitly provided and different from seller, put it here. Keep the original form — do not translate, split, normalize, or rewrite.
If brand is "Other", "Unknown", "Generic", or blank, OR if brand and seller are the same — use a meaningful product attribute n this slot instead. Never output "Other"/"Unknown"/"Generic" as a slot value.

**Slot 7 — Seller / Retailer**
Seller MUST go here whenever it is explicitly provided and not "Other"/"Unknown"/blank. This is the fixed slot for seller — never move seller to Slot 6.
If brand and seller are the same, put the name here (not in Slot 6). Keep the original form — do not translate, split, normalize, or rewrite.
If seller is "Other"/"Unknown"/blank, use a remaining useful product attribute instead. Never use generic filler ("other", "none", "N/A").


## Rules

### Rule 1 — Word Style
Output exactly 7 distinct slots. Prefer **single descriptive words**. Use hyphenated words when natural:
- noise-canceling
- anti-aging
- over-ear

Avoid unnecessary long phrases. Phrase slots are allowed ONLY when required:
- Model names / product-line names (Slot 2)
- Brand names
- Retailer names
- Fixed proper nouns

### Rule 2 — Product-First Order
The first 4 slots MUST all describe the product itself.
- When Slot 2 is a model name: Category, model, function/feature, distinguishing attribute
- When Slot 2 is core function: Category, function, feature/subtype, distinguishing attribute

A reader should understand what the product is from these first 4 entries alone.

### Rule 3 — Language
Use the same primary language as the PRODUCT INFORMATION for descriptive entries. However:
- Brand names, seller names, model names, and other proper nouns must remain in their original form
- Do not translate proper nouns

### Rule 4 — English Word Form
For English descriptive entries only, use base or dictionary form when natural. Avoid unnecessary inflections such as plural nouns or verb forms ending in -ing or -ed. Do NOT alter proper nouns, brand names, seller names, or model names.

### Rule 5 — Hyphens
Use hyphens only when natural and helpful, mainly in English descriptive entries. Do not force hyphenation in Chinese, Japanese, or other non-English languages.

### Rule 6 — No Duplicates
All 7 slots must be different. If brand and seller are identical or nearly identical, include them only once. Use the freed slot for another informative aspect (style, audience, or additional product detail).

### Rule 7 — Required Aspects
- If a specific target audience is explicitly present, it must appear.
- If brand is explicitly present, it must appear.
- If seller is explicitly present and different from brand, it must appear.

Only omit an aspect if it is genuinely absent or too uncertain.

### Rule 8 — No Guessing
Do not invent missing facts. Do not guess audience, seller, style, or feature unless supported by the product information.

### Rule 9 — Similar Items
Use the TOP 5 SIMILAR PRODUCTS only as reference for terminology consistency. Prefer accurate description of the current product over forced consistency. If similar items share a common category or naming pattern, use consistent terminology when it remains accurate.

### Rule 10 — Uniqueness
Use the 4 product-description slots to include distinguishing traits that separate this product from similar items.

### Rule 11 — No Redundancy
Do not use a hypernym and a hyponym that convey the same meaning. For example, do NOT output both "coat" and "outwear", or both "shoes" and "footwear". Choose the more specific or informative term.

### Rule 12 — No Vague Fillers
Never use vague quality tiers (flagship, premium, classic, modern, basic) or placeholder words (other, none, N/A, general) in any slot. Every slot should carry concrete, searchable information.

### Rule 13 — Model Name Format
Keep model names compact. Drop brand prefixes (brand goes in Slot 6) and non-essential suffixes like color, storage, or SKU. Do NOT split a model name across multiple slots — it is ONE phrase slot.
- iPhone 17 Pro Max (not "Apple iPhone 17 Pro Max 256GB Deep Blue")
- Technic BMW M4 GT3 EVO (not "LEGO Technic BMW M4 GT3 EVO Race Car 42226")
- WH-1000XM6 (not "Sony WH-1000XM6 Wireless Noise Canceling")
- Instant Pot Duo (not "Instant Pot Duo 7-in-1 Electric Pressure Cooker 6Qt")
- ThinkPad X1 Carbon (not "Lenovo ThinkPad X1 Carbon Gen 12 14-inch")

## Complete Output Examples

With model name (electronics):
`<Output>[smartphone, iPhone 17 Pro Max, unlocked, 256gb, deep-blue, Apple, Best Buy]</Output>`
`<Output>[headphone, WH-1000XM6, noise-canceling, over-ear, silver, Sony, Amazon]</Output>`
`<Output>[toy, Technic BMW M4 GT3 EVO, race-car, detailed, kids, LEGO, Scheels]</Output>`
`<Output>[vacuum, Saros 20, robotic, self-emptying, pet, Roborock, Best Buy]</Output>`

Without model name (clothing / home / outdoor):
`<Output>[shirtdress, button-down, midi, striped, women, Evi Grintela, Neiman Marcus]</Output>`
`<Output>[clutch, beaded, structured, satin, evening, Anthropologie, Nordstrom]</Output>`
`<Output>[chandelier, hand-blown, glass, brass, art-deco, Rejuvenation, Frontgate]</Output>`
`<Output>[tent, ultralight, 2-person, freestanding, backpacking, Big Agnes, REI Co-op]</Output>`

## Final Check Before Output

1. Is this a real, specific product? If not, output `[]`
2. Are there exactly 7 slots?
3. Are slots 1–4 ALL product-descriptive?
4. If a specific audience is explicitly provided, is it included?
5. If brand exists, is it included in the slots?
6. If seller exists and differs from brand, is it included in the slots?
7. Are all 7 slots distinct (no duplicates, no hypernym+hyponym pairs)?
8. Are proper nouns kept intact and untranslated?
9. Are descriptive entries in the product's primary language?
10. Is there no extra text outside `<Output></Output>`?
11. If the product has a known model/series name, is it in Slot 2?
12. Is every slot filled with concrete, searchable info (no vague fillers)?

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
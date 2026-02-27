# ShoppingGenRec Raw Data

This directory contains the raw data files for the ShoppingGenRec dataset.

## Files

### `item.json`

A JSON file containing product (item) metadata. The top-level keys are **item IDs** (string).

Each item has the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `title` | string | Product title / name |
| `description` | string | Product description text (may be empty) |
| `categories` | string | Product category hierarchy, delimited by ` > ` (e.g., `"Beauty > Hair Care > Shampoos"`) |

**Example:**
```json
{
  "1": {
    "title": "Phyto Phytocitrus Restructuring Mask for Unisex, 6.7 Ounce",
    "description": "True ColorsPhyto Phytocitrus Restructuring Mask delivers ...",
    "categories": "Beauty > Hair Care > Conditioners"
  }
}
```

---

### `sequential_data.txt`

A space-separated text file representing user interaction sequences. Each line corresponds to one user.

| Column | Description |
|--------|-------------|
| 1st token | **User ID** |
| Remaining tokens | **Item IDs** in chronological interaction order |

**Example:**
```
1 1 2 3 4 5 6 7 8
```
This means user `1` interacted with items `1, 2, 3, 4, 5, 6, 7, 8` in order.

---

### `shopping_journey.json`

A JSON file containing shopping journey data. The top-level keys are **journey IDs** (string).

Each journey has the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | The user who initiated this journey |
| `user_shopping_events` | list[string] | A list of recent user shopping events. |
| `system_time` | string | The system timestamp (UTC) when the journey was generated, in `M/DD/YYYY` format |
| `journey` | object | The generated shopping journey (see below) |

The `journey` object contains:

| Field | Type | Description |
|-------|------|-------------|
| `title` | string | A short descriptive title for the shopping journey |
| `query` | string | A search query representing the user's shopping intent |
| `description` | string | A brief description of the shopping journey |
| `product_ids` | list[string] | List of recommended product IDs (referencing keys in `item.json`) |

**Example:**
```json
{
  "1": {
    "user_id": "1",
    "user_shopping_events": [
      "2 days 15 hours ago | Searched | hair care products for color-treated hair",
      "1 week 1 day ago | Browsed | Phyto Phytocitrus Restructuring Mask"
    ],
    "system_time": "9/21/2025",
    "journey": {
      "title": "Color-Treated Hair Care Essentials",
      "query": "Best shampoo and conditioner for color-treated hair",
      "description": "A shopping journey focused on maintaining vibrant color-treated hair.",
      "product_ids": ["1", "2", "4", "9"]
    }
  }
}
```

---

### `user_persona.txt`

A tab-separated (`\t`) text file containing user persona narratives. Each line corresponds to one user.

| Column | Delimiter | Description |
|--------|-----------|-------------|
| 1 | `\t` | **User ID** |
| 2 | `\t` | **Persona text** — a structured narrative describing the user's profile, location, recent shopping click behavior, and interest summary |

The persona text follows this structure:
- `# User Profile Narrative` — basic user info (e.g., location)
- `# Shopping Click Behavior` — recently clicked products
- `# Interest Summary` — aggregated category interests

**Example:**
```
1	# User Profile Narrative \nThe user is based in Japan.\n\n# Shopping Click Behavior \nThe user recently clicked on a product "Home Travel 9X/1X Folding Lighted Cosmetic Mirror", ...\n\n# Interest Summary \nThe user shows interest in: Hair Care, Makeup, Skin Care.
```

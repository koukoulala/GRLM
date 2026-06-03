"""
Seller / Price analysis with 7-tier taxonomy.

Tiers (from ordered most-curated to least):
  luxury       — 奢华百货 / 奢侈品
  department   — 主流高端百货
  specialty    — 垂直专家零售商（按品类专业服务）
  brand_dtc    — 品牌官方 DTC
  mass         — 大众综合零售
  marketplace  — P2P / UGC / 长尾聚合
  other        — 未匹配规则的零散站点

Rationale: the older 4-tier (luxury/premium/mass/other) lumps all vertical
specialists (Albee Baby / Clive Coffee / Williams Sonoma / B&H / CDW /
Sephora / Chewy / Hobby Lobby) into "other" — which produces the misleading
conclusion that "P2 'other' 占比上升 = tier 退化". With the 7-tier split,
垂直专家 + 品牌官店 are surfaced as distinct quality signals.

Usage: py analyze_sellers.py <paired_data.json> <output_dir>
"""
import json, os, sys, collections, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

if len(sys.argv) < 3:
    print("Usage: py analyze_sellers.py <paired_data.json> <output_dir>")
    sys.exit(1)

paired_file = sys.argv[1]
output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)

with open(paired_file, 'r', encoding='utf-8') as f:
    paired = json.load(f)

# --- 7-tier classification ---
LUXURY = {
    "bloomingdale", "saks fifth avenue", "neiman marcus", "cettire", "farfetch",
    "mulberry", "coach", "michael kors", "chloe", "revolve", "brahmin", "cuyana",
    "brooks brothers", "modesens", "peter millar", "bergdorf goodman",
    "perigold", "ross-simons", "coachoutlet.com",
}

DEPARTMENT = {
    "macy's", "macys", "nordstrom", "nordstrom rack", "dillard's", "dillards",
    "belk", "anthropologie",
}

SPECIALTY = {
    # 户外/运动
    "rei", "backcountry.com", "backcountry", "moosejaw", "scheels",
    "dick's sporting goods", "dicks sporting goods", "hibbett", "fanatics",
    "academy sports + outdoors", "academy sports", "running warehouse",
    "holabird sports", "peter glenn ski and sports", "jd sports",
    "helly hansen", "lectric ebikes store",
    # 厨房/家居/家具
    "williams sonoma", "williams-sonoma", "crate & barrel", "crate and barrel",
    "pottery barn", "sur la table", "west elm", "allmodern", "lamps plus",
    "world market", "the container store", "ikea", "ashley", "living spaces furniture",
    "shades of light", "1-800-lighting", "1-800 lighting",
    "luxome", "baloo living", "boll & branch",
    # 电子/影像/音频
    "b&h photo-video-pro a", "b&h photo", "adorama camera", "adorama",
    "crutchfield", "newegg.com", "newegg", "abt electronics", "micro center",
    "best buy", "pc richard", "epson.com", "zagg",
    # B2B/办公/工业
    "cdw", "webstaurantstore", "uline", "office depot", "staples",
    "national business furniture", "madison liquidators", "northern tool",
    # 母婴
    "albee baby", "buybuy baby", "buy buy baby", "babylist",
    "gerber childrenswear", "carter's", "carters",
    # 咖啡
    "clive coffee", "whole latte love", "seattle coffee gear",
    "prima coffee", "prima coffee equipment",
    # 美妆/保健
    "sephora", "ulta", "iherb", "the vitamin shoppe",
    "olive young apac & emea", "ilnp", "melodysusie",
    # 宠物
    "chewy.com", "chewy", "petco", "petco.com", "petsmart", "petsmart.com",
    # 手工/艺术
    "michaels", "hobby lobby", "st. louis art supply", "blick art materials",
    "joann", "joann fabric", "fat quarter shop",
    # 五金/家装
    "home depot", "lowe's", "lowes", "harbor freight", "ace hardware",
    # 婚庆
    "david's bridal - microsoft merchant center", "david's bridal", "davids bridal",
    "lulus",
    # 鞋类
    "zappos", "dsw", "journeys", "naturalizer.com", "naturalizer",
    "shoes.com", "famous footwear", "foot locker", "snipes", "orthofeet",
    # 玩具/教育
    "the lego store", "lego us",
    # 床品/睡眠
    "casper", "purple", "tempur-pedic", "saatva",
    # 长尾配件
    "strapsco", "ekster",
    # 自行车/单车
    "trek bikes",
    # 健康/医疗/食品
    "walgreens", "safeway", "men's wearhouse",
    # 珠宝
    "novica", "gorjana",
    # 食品 B2B
    "bj's wholesale club inc.", "bjs wholesale club inc.",
    # 办公文具（W.B. Mason / Quill / Neobits / OfficeSupply.com / JetPens / Printivity / Papier / BandsCo. 等）
    "w.b. mason", "wb mason", "quill", "quill.com", "neobits",
    "officesupply.com", "office supply", "jetpens", "printivity",
    "papier", "bandsco.", "bandsco", "mybinding.com", "mybinding",
    # 书店
    "barnes & noble", "barnes and noble",
    # 美妆专卖
    "sally beauty",
    # 汽车配件
    "parts geek", "go-parts",
    # 派对/手工
    "oriental trading company", "oriental trading",
    "woodartsupply", "wood art supply", "super arbor",
    # CPAP/医疗
    "the cpap shop", "cpap supplies", "cpap my way",
    # 家居/家具
    "rejuvenation", "raymour & flanigan", "raymour and flanigan",
    "lumens.com", "lumens", "blinds.com",
    # 摩托/越野
    "revzilla", "camping world",
    # 工业/MRO
    "grainger", "zoro.com", "zoro",
    # 鞋类（追加）
    "rack room shoes",
    # 厨房专家（追加）
    "everything kitchens llc", "everything kitchens",
    "fiesta factory direct",
    # 农资/农场
    "tractor supply co.", "tractor supply",
    # 食品/超市（地区性大型）
    "kroger",
    # 五金（追加）
    "menards.com", "menards",
    # 宠物
    "4knines",
    # 追加：摄影/影像配件专家
    "k&f concept - us", "k&f concept",
    # 追加：智能家居/创新零售
    "wellbots",
    # 追加：餐饮设备
    "katom restaurant supply, inc.", "katom restaurant supply",
    # 追加（2026-05-27 9B-vs-CapIndex run 出现的高频未分类项）
    "advance auto parts", "autozone.com", "autozone",
    "etrailer.com", "etrailer",
    "champs sports",
    "beach camera",
    "buy-rite beauty",
    "shopbop",
    "books-a-million", "christianbook.com", "christianbook",
    "urban outfitters",
    "mechanical keyboards",
    "vevor", "vevor llc",
    "havenly",
    "wolf & badger - uk", "wolf & badger",
    "soma intimates", "j.jill",
    "athleta",
}

BRAND_DTC = {
    # 运动鞋服
    "nike.com", "adidas", "adidas.com", "lululemon", "the north face",
    "patagonia", "j.crew", "j.crew factory", "banana republic",
    "banana republic factory", "ann taylor", "ann taylor factory",
    "talbots", "loft", "chicos", "chico's", "aeropostale",
    "white house black market", "h&m", "asos", "quince", "uniqlo",
    "old navy", "gap", "everlane", "abercrombie & fitch", "express",
    "boohoo", "shein", "temu", "boden us bing", "boden us", "boden",
    "bombas",
    # 鞋类品牌
    "hoka (hoka.com)", "hoka", "brooks running", "brooks", "on running",
    "saucony", "asics", "new balance", "skechers", "puma",
    "vans", "converse",
    # 电子品牌
    "apple", "apple.com", "dell", "hp", "lenovo", "anker", "delonghi",
    "simplisafe, inc.", "simplisafe", "garmin", "fitbit",
    "samsung", "lg", "sony", "bose", "jbl", "dyson", "belkin",
    "belkin.com", "roborock",
    # 玩具/娱乐
    "nintendo", "playstation",
    # 内衣/家居
    "victoria's secret", "bath & body works", "yankee candle",
    # 美妆品牌官店
    "rare beauty", "fenty beauty",
    # 食品/咖啡品牌
    "starbucks", "nespresso",
    # 户外品牌
    "yeti", "stanley",
    # 配件
    "ray-ban", "warby parker",
    # 服饰品牌（追加）
    "levi's", "levis", "fjällräven usa", "fjallraven usa", "fjallraven",
    # 电子品牌（追加）
    "corsair", "ecoflow us", "ecoflow",
    # 追加：办公/打印
    "brother usa", "brother",
    # 追加：服饰/工装/运动
    "dickies.com", "dickies", "under armour",
    # 追加（2026-05-27 9B-vs-CapIndex run 出现的高频未分类项）
    "phomemo", "pentel of america, ltd.", "pentel of america", "pentel",
    "woolx", "kenmore floor care", "sockwell", "jsaux gaming", "jsaux",
    "leatherology", "love desk mats", "t-mobile",
    "american eagle outfitters",
}

MASS = {
    "amazon.com", "amazon", "walmart", "walmart.com", "target", "target.com",
    "kohl's", "kohls", "jcpenney", "wayfair", "bed bath & beyond",
    "hsn", "qvc", "overstock.com", "overstock", "nfm",
    "sam's club", "sams club", "costco", "bj's wholesale", "bjs wholesale",
    "boscov's", "boscovs",
    # 追加：地区性超市
    "h-e-b", "heb",
}

MARKETPLACE = {
    "etsy", "ebay", "poshmark", "mercari", "depop", "thredup", "tradesy",
    "alibaba", "aliexpress", "wish", "vinted",
    # B2B / wholesale 聚合
    "faire",
}

TIERS = ['luxury', 'department', 'specialty', 'brand_dtc', 'mass', 'marketplace', 'other']


def classify(seller):
    if not seller:
        return 'other'
    sl = seller.lower().strip()
    # exact match first
    if sl in LUXURY: return 'luxury'
    if sl in DEPARTMENT: return 'department'
    if sl in SPECIALTY: return 'specialty'
    if sl in BRAND_DTC: return 'brand_dtc'
    if sl in MASS: return 'mass'
    if sl in MARKETPLACE: return 'marketplace'
    # fuzzy substring fallback
    for s in LUXURY:
        if s in sl: return 'luxury'
    for s in DEPARTMENT:
        if s in sl: return 'department'
    for s in SPECIALTY:
        if s in sl: return 'specialty'
    for s in BRAND_DTC:
        if s in sl: return 'brand_dtc'
    for s in MASS:
        if s in sl: return 'mass'
    for s in MARKETPLACE:
        if s in sl: return 'marketplace'
    return 'other'


# Aggregate sellers + brands + prices
p1_sellers = collections.Counter()
p2_sellers = collections.Counter()
p1_prices = []
p2_prices = []
p1_brands = collections.Counter()
p2_brands = collections.Counter()

for user in paired:
    # P1
    for j in user.get('p1', {}).get('journeys', []):
        for p in j.get('products', []):
            # Schema A: top-level
            s = p.get('seller', '')
            if s: p1_sellers[s] += 1
            pv = p.get('price', 0)
            if isinstance(pv, (int, float)) and pv > 0:
                p1_prices.append(pv)
            elif isinstance(pv, str):
                try:
                    pv2 = float(pv.replace('$', '').replace(',', ''))
                    if pv2 > 0: p1_prices.append(pv2)
                except: pass
            # Schema B/C: nested matched_products
            for mp in p.get('matched_products', []):
                ms = mp.get('Seller', '')
                if ms: p1_sellers[ms] += 1
                mb = mp.get('Brand', '')
                if mb: p1_brands[mb] += 1
                mps = mp.get('OriginalPrice', '')
                try:
                    pv = float(str(mps).replace('$', '').replace(',', ''))
                    if pv > 0: p1_prices.append(pv)
                except: pass
    # P2
    for j in user.get('p2', {}).get('journeys', []):
        for p in j.get('products', []):
            s = p.get('seller', '')
            if s: p2_sellers[s] += 1
            b = p.get('brand', '')
            if b: p2_brands[b] += 1
            for mp in p.get('matched_products', []):
                ms = mp.get('Seller', '')
                if ms: p2_sellers[ms] += 1
                mb = mp.get('Brand', '')
                if mb: p2_brands[mb] += 1
                mps = mp.get('OriginalPrice', '')
                try:
                    pv = float(str(mps).replace('$', '').replace(',', ''))
                    if pv > 0: p2_prices.append(pv)
                except: pass
            ps = str(p.get('price', ''))
            if ps and not p.get('matched_products'):
                try:
                    pv = float(ps.replace('$', '').replace(',', ''))
                    if pv > 0: p2_prices.append(pv)
                except: pass


def price_stats(prices):
    if not prices: return {}
    prices.sort()
    n = len(prices)
    return {
        'count': n, 'min': prices[0], 'p25': prices[n // 4],
        'median': prices[n // 2], 'p75': prices[3 * n // 4],
        'max': prices[-1], 'mean': round(sum(prices) / n, 2),
    }


# Tier distribution
p1_tiers = collections.Counter()
p2_tiers = collections.Counter()
p1_tier_sellers = collections.defaultdict(collections.Counter)
p2_tier_sellers = collections.defaultdict(collections.Counter)
for s, c in p1_sellers.items():
    t = classify(s)
    p1_tiers[t] += c
    p1_tier_sellers[t][s] += c
for s, c in p2_sellers.items():
    t = classify(s)
    p2_tiers[t] += c
    p2_tier_sellers[t][s] += c

p1_total = sum(p1_tiers.values()) or 1
p2_total = sum(p2_tiers.values()) or 1


def section(title):
    print(f'\n{"="*60}\n  {title}\n{"="*60}')


section('TIER DISTRIBUTION (7-tier)')
print(f'{"Tier":<14} {"P1 %":>7} {"P1 N":>7} {"P2 %":>7} {"P2 N":>7} {"Δ pp":>7}')
for tier in TIERS:
    cp = p1_tiers[tier] / p1_total * 100
    sp = p2_tiers[tier] / p2_total * 100
    print(f'{tier:<14} {cp:>6.1f}% {p1_tiers[tier]:>7} {sp:>6.1f}% {p2_tiers[tier]:>7} {sp-cp:>+6.1f}')

section('Top sellers per tier (P2)')
for tier in TIERS:
    sellers = p2_tier_sellers[tier].most_common(8)
    if sellers:
        print(f'\n[{tier}]')
        for s, c in sellers:
            print(f'  {c:4d} | {s}')

ps1 = price_stats(p1_prices)
ps2 = price_stats(p2_prices)
if ps1:
    section('PRICE STATS')
    print(f'P1: min=${ps1["min"]:.0f} p25=${ps1["p25"]:.0f} median=${ps1["median"]:.0f} p75=${ps1["p75"]:.0f} max=${ps1["max"]:.0f} mean=${ps1["mean"]:.0f} (n={ps1["count"]})')
if ps2:
    print(f'P2: min=${ps2["min"]:.0f} p25=${ps2["p25"]:.0f} median=${ps2["median"]:.0f} p75=${ps2["p75"]:.0f} max=${ps2["max"]:.0f} mean=${ps2["mean"]:.0f} (n={ps2["count"]})')

# Surface unclassified — useful for refining the taxonomy
unclassified_p1 = [(s, c) for s, c in p1_sellers.most_common() if classify(s) == 'other'][:20]
unclassified_p2 = [(s, c) for s, c in p2_sellers.most_common() if classify(s) == 'other'][:20]
section('Top unclassified — review to extend tier sets')
print('P1:')
for s, c in unclassified_p1:
    print(f'  {c:4d} | {s}')
print('\nP2:')
for s, c in unclassified_p2:
    print(f'  {c:4d} | {s}')

# Persist
results = {
    'tier_taxonomy': {
        'luxury': '奢华百货 / 奢侈品',
        'department': "主流高端百货 (Macy's / Nordstrom / Dillard's)",
        'specialty': '垂直专家零售商 (REI / Williams Sonoma / Albee Baby / Clive Coffee / B&H / CDW ...)',
        'brand_dtc': '品牌官方 DTC (Nike.com / Apple / Anker / Hoka / The North Face ...)',
        'mass': "大众综合零售 (Amazon / Walmart / Target / Kohl's / Wayfair)",
        'marketplace': 'P2P / UGC / 长尾聚合 (Etsy / eBay)',
        'other': '未匹配规则的零散站点',
    },
    'p1_top_sellers': p1_sellers.most_common(50),
    'p2_top_sellers': p2_sellers.most_common(50),
    'p1_top_brands': p1_brands.most_common(30),
    'p2_top_brands': p2_brands.most_common(30),
    'p1_price_stats': ps1,
    'p2_price_stats': ps2,
    'p1_tiers': dict(p1_tiers),
    'p2_tiers': dict(p2_tiers),
    'p1_tier_sellers': {t: dict(c.most_common(15)) for t, c in p1_tier_sellers.items()},
    'p2_tier_sellers': {t: dict(c.most_common(15)) for t, c in p2_tier_sellers.items()},
    'p1_unclassified_top': unclassified_p1,
    'p2_unclassified_top': unclassified_p2,
}
out_path = os.path.join(output_dir, 'seller_analysis.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f'\nSaved: {out_path}')

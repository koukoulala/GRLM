"""Step 5.3 Journey: Split-Task Evaluation (event2journey + profile2journey)

Reads two pre-built test TSV files (from s4 --build_test_tsv):
  - event2journey_full_cleaned_test.tsv
  - profile2journey_full_cleaned_test.tsv

Each file has different users and different LLM ground-truth results.
The profile2journey TSV includes a Profile column (no Copilot API needed).

Pipeline:
  1. Read both test TSVs from --test_dir.
  2. Sample N users from each task.
  3. Parse LLM ground truth from FinalJourney for both tasks.
  4. Build event2journey prompts (events only) and profile2journey prompts
     (profile + events).
  5. Run vLLM inference on all prompts (batched).
  6. Map ProductTIDs -> GlobalOfferIds via exact + fuzzy matching.
  7. Output per-task: {task}_llm_output.tsv + {task}_slm_output.tsv.
  8. Side-by-side comparison statistics.

Usage:
    python s5_3_journey_eval_split_task.py \
        --model_path /path/to/checkpoint \
        --test_dir /path/to/sft_data/ \
        --output_dir ./eval_results/split_task/ \
        --sample_n 500
"""

import os, re, csv, json, random, argparse, sys
from collections import defaultdict
import numpy as np

csv.field_size_limit(sys.maxsize)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "resources"))

from term_normalizer import normalize_term

# === Inlined from pre_s2_construct_shopping_journey ===

_RE_EVENT_NUMBER = re.compile(r"^\d+\s*\|\s*(.*)")
_RE_NON_ALNUM = re.compile(r"[^a-z0-9\s|]")
_RE_MULTI_SPACE = re.compile(r"\s+")


# === Time normalization (must match s3_build_journey_sft_data.py) ===

def _normalize_time_expr(match):
    """Normalize a single time expression to days or hours."""
    text = match.group(0)
    parts = re.findall(r'(\d+)\s*(month|week|day|hour|minute|second)s?', text,
                       re.IGNORECASE)
    if not parts:
        return text
    total_hours = 0; total_minutes = 0
    for num_str, unit in parts:
        num = int(num_str)
        u = unit.lower()
        if u == 'month': total_hours += num * 30 * 24
        elif u == 'week': total_hours += num * 7 * 24
        elif u == 'day': total_hours += num * 24
        elif u == 'hour': total_hours += num
        elif u == 'minute': total_minutes += num
    total_days = total_hours // 24
    if total_days > 0: return f"{total_days} days ago"
    elif total_hours > 0: return f"{total_hours} hours ago"
    elif total_minutes > 0: return f"{total_minutes} minutes ago"
    return "0 minutes ago"

def normalize_event_times(text):
    """Normalize all time expressions (weeks/months -> days) in event text."""
    return re.sub(r'(?:\d+\s*(?:month|week|day|hour|minute|second)s?\s*)+ago',
                  _normalize_time_expr, text, flags=re.IGNORECASE)


# === Profile JSON cleaning (must match s3_build_journey_sft_data.py) ===

def _clean_profile_json(raw):
    """Unescape multi-layer escaped profile JSON."""
    if not raw or not raw.strip():
        return raw
    try:
        obj = json.loads(raw)
        if isinstance(obj, (dict, list)):
            return json.dumps(obj, ensure_ascii=False, separators=(',', ': '))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    text = raw
    for _ in range(3):
        text = text.replace('\\\\', '\x00__BS__\x00')
        text = text.replace('\\"', '"')
        text = text.replace('\x00__BS__\x00', '\\')
        try:
            obj = json.loads(text)
            if isinstance(obj, (dict, list)):
                return json.dumps(obj, ensure_ascii=False, separators=(',', ': '))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return raw

def _normalize_event_key(event):
    key = event.lower()
    key = _RE_NON_ALNUM.sub(" ", key)
    key = _RE_MULTI_SPACE.sub(" ", key)
    return key.strip()

def parse_readable_user_events(events_text):
    if not events_text or not events_text.strip():
        return [], 0
    text = events_text.replace("#N#", "\n")
    lines = text.strip().split("\n")
    raw_events = []
    for line in lines:
        line = line.strip()
        if not line: continue
        match = _RE_EVENT_NUMBER.match(line)
        if match:
            event = match.group(1).strip()
            if event: raw_events.append(event)
    seen_keys = set(); deduped = []
    for event in raw_events:
        key = _normalize_event_key(event)
        if key not in seen_keys:
            seen_keys.add(key); deduped.append(event)
    return deduped, len(raw_events)

def parse_final_journey(journey_text, valid_offer_ids):
    stats = {"total_offer_ids":0,"found_offer_ids":0,"missing_offer_ids":0,
             "total_journeys":0,"kept_journeys":0,"empty_product_journeys":0}
    missing_ids = set()
    if not journey_text or not journey_text.strip():
        return [], stats, missing_ids
    data = None
    for attempt in [journey_text, journey_text.replace('\\"', '"')]:
        try: data = json.loads(attempt); break
        except: continue
    if data is None:
        return [], stats, missing_ids
    journeys = []
    for j_raw in data.get("ContinuedJourneys", []):
        title = j_raw.get("Title", "").strip()
        reason = j_raw.get("Reason", "").strip()
        products = j_raw.get("Products", [])
        stats["total_journeys"] += 1
        product_ids = []
        for p in products:
            oid = str(p.get("OfferId", "")).strip()
            if not oid: continue
            stats["total_offer_ids"] += 1
            if oid in valid_offer_ids:
                product_ids.append(oid); stats["found_offer_ids"] += 1
            else:
                stats["missing_offer_ids"] += 1; missing_ids.add(oid)
        if not product_ids:
            stats["empty_product_journeys"] += 1; continue
        entry = {"title":title,"reason":reason,"product_ids":product_ids}
        jtype = j_raw.get("JourneyType","").strip()
        if jtype: entry["journey_type"] = jtype
        journeys.append(entry); stats["kept_journeys"] += 1
    return journeys, stats, missing_ids

# === TID Matching (same as s5_journey_eval.py) ===

def load_item_titles(item_file):
    print(f"  Loading item titles from: {item_file}")
    id2title = {}
    with open(item_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item_id, item_data in data.items():
        if isinstance(item_data, dict) and "title" in item_data:
            id2title[str(item_id)] = item_data["title"]
    print(f"    Loaded titles for {len(id2title):,} items")
    return id2title

def _reorder_tid_words(words, pos):
    if pos < 0 or pos >= len(words): return list(words)
    return words[:pos] + words[pos+1:] + [words[pos]]

def create_reverse_mapping(original_dict, reorder_pos=-1):
    reverse_mapping, word_to_keys = {}, defaultdict(list)
    normalized_key_map, sorted_key_map = {}, {}
    for key_str, ids in original_dict.items():
        words = [normalize_term(w.strip()).lower() for w in key_str.split(",")]
        reordered = _reorder_tid_words(words, reorder_pos)
        reverse_mapping[key_str] = {"words": reordered, "ids": ids}
        for w in reordered: word_to_keys[w].append(key_str)
        nk = ",".join(words)
        if nk not in normalized_key_map: normalized_key_map[nk] = key_str
        sk = ",".join(sorted(words))
        if sk not in sorted_key_map: sorted_key_map[sk] = key_str
    return reverse_mapping, word_to_keys, normalized_key_map, sorted_key_map

def get_iid_by_tid(tid_words, tid2item_id, rm, w2k, nkm, skm, reorder_pos=-1,
                   fuzzy_score_threshold=0):
    wl = [normalize_term(w).strip().lower() for w in tid_words]
    nk = ",".join(wl)
    if nk in nkm:
        return list(tid2item_id[nkm[nk]]), "exact", len(tid_words), 0.0
    sk = ",".join(sorted(wl))
    if sk in skm:
        return list(tid2item_id[skm[sk]]), "exact", len(tid_words), 0.0
    wr = _reorder_tid_words(wl, reorder_pos)
    # Use first min_prefix reordered words to narrow candidates via set intersection
    min_prefix = max(int(fuzzy_score_threshold), 1)
    n_filter = min(min_prefix, len(wr))
    candidates = None
    for i in range(n_filter):
        w = wr[i]
        if w not in w2k: return [], "none", 0, 0.0
        wset = set(w2k[w])
        candidates = wset if candidates is None else (candidates & wset)
        if not candidates: return [], "none", 0, 0.0
    best_pl, best_key = 0, None
    for ck in candidates:
        cw = rm[ck]["words"]
        pl = 0
        for qi, qw in enumerate(wr):
            if qi < len(cw) and cw[qi] == qw: pl += 1
            else: break
        if pl > best_pl: best_pl, best_key = pl, ck
    if best_key and best_pl >= min_prefix:
        return rm[best_key]["ids"][:1], "fuzzy", best_pl, float(best_pl)
    return [], "none", 0, 0.0

# === Journey Parsing & Mapping ===

def parse_journey_json(raw):
    if not raw or not raw.strip(): return None
    text = raw.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    bs = text.find("{")
    if bs == -1: return None
    d, be = 0, -1
    for i in range(bs, len(text)):
        if text[i] == "{": d += 1
        elif text[i] == "}":
            d -= 1
            if d == 0: be = i; break
    cand = text[bs:be+1] if be != -1 else text[bs:] + "}"
    for t in [cand, text]:
        try:
            data = json.loads(t)
            if "ContinuedJourneys" in data: return data
        except: pass
    return None

def map_journey_products(jd, tid2item_id, rm, w2k, nkm, skm,
                         id2title=None, fuzzy_score_threshold=0.0, reorder_pos=-1):
    stats = {"total_products":0,"exact_matches":0,"fuzzy_matches":0,"no_matches":0,
             "fuzzy_filtered":0,"journeys_dropped":0,"fuzzy_matched_words":[],"fuzzy_best_scores":[]}
    if jd is None: return None, stats
    mapped = {"ContinuedJourneys": []}
    for j in jd.get("ContinuedJourneys", []):
        mj = {"Title": j.get("Title",""), "Reason": j.get("Reason",""), "Products": []}
        used = set()
        for tw in j.get("ProductTIDs", []):
            stats["total_products"] += 1
            if not isinstance(tw, list) or not tw:
                stats["no_matches"] += 1; continue
            iids, mt, mw, bs = get_iid_by_tid(tw, tid2item_id, rm, w2k, nkm, skm,
                reorder_pos=reorder_pos, fuzzy_score_threshold=fuzzy_score_threshold)
            if mt == "exact": stats["exact_matches"] += 1
            elif mt == "fuzzy":
                stats["fuzzy_matches"] += 1
                stats["fuzzy_matched_words"].append(mw)
                stats["fuzzy_best_scores"].append(bs)
                if fuzzy_score_threshold > 0 and bs < fuzzy_score_threshold:
                    stats["fuzzy_filtered"] += 1; continue
            else: stats["no_matches"] += 1; continue
            gid = None
            for c in (iids or []):
                if c not in used: gid = c; break
            if gid is None and iids: gid = iids[0]
            if gid: used.add(gid)
            mj["Products"].append({"TID":tw,"GlobalOfferIds":[gid] if gid else [],
                "match_type":mt,"title":id2title.get(str(gid),"") if (gid and id2title) else ""})
        # dedup products
        seen = set(); dp = []
        for p in mj["Products"]:
            g = p["GlobalOfferIds"][0] if p["GlobalOfferIds"] else None
            if g and g in seen: stats["products_deduped"] = stats.get("products_deduped",0)+1; continue
            if g: seen.add(g)
            dp.append(p)
        mj["Products"] = dp
        if mj["Products"]: mapped["ContinuedJourneys"].append(mj)
        else: stats["journeys_dropped"] += 1
    # dedup journeys by title
    st = set(); dj = []
    for j in mapped["ContinuedJourneys"]:
        tk = j["Title"].strip().lower()
        if tk in st: stats["journeys_title_deduped"] = stats.get("journeys_title_deduped",0)+1; continue
        st.add(tk); dj.append(j)
    mapped["ContinuedJourneys"] = dj
    if not mapped["ContinuedJourneys"]: return None, stats
    return mapped, stats

# === Data Loading ===

def read_test_tsv(fp):
    rows, seen = [], set()
    with open(fp, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header: raise ValueError(f"Empty: {fp}")
        cm = {n.strip(): i for i, n in enumerate(header)}
        for row in reader:
            ui = cm.get("UserId", 0)
            if len(row) <= ui: continue
            uid = row[ui].strip()
            if not uid or uid in seen: continue
            seen.add(uid)
            rd = {}
            for n, i in cm.items(): rd[n] = row[i].strip() if i < len(row) else ""
            rows.append(rd)
    return rows, list(cm.keys())

def format_ground_truth_journey(jt):
    if not jt or not jt.strip(): return "", []
    # Unescape TSV-escaped JSON (\" -> ")
    jt_clean = jt.replace('\\"', '"').replace('\\\\', '\\')
    class _A:
        def __contains__(self, x): return True
    # Try cleaned version first, then original
    for attempt in [jt_clean, jt]:
        journeys, _, _ = parse_final_journey(attempt, _A())
        if journeys:
            cont = [{"Title":j.get("title",""),"Reason":j.get("reason",""),
                     "ProductIds":j.get("product_ids",[])} for j in journeys]
            return json.dumps({"ContinuedJourneys":cont}, ensure_ascii=False), journeys
    return "", []

def process_task_data(name, rows, has_profile=False):
    data = []; ne = 0; gok = 0; gf = 0
    for r in rows:
        uid = r["UserId"]
        uer = r.get("UserHistory","")
        rr = r.get("ReadableUserEvents","")
        el, _ = parse_readable_user_events(rr)
        if not el: ne += 1
        rl = [f"{i+1} | {(e[:150]+'...' if len(e)>150 else e)}" for i, e in enumerate(el)]
        uer_readable = "\n".join(rl)
        gt_json, gt_j = format_ground_truth_journey(r.get("FinalJourney",""))
        if gt_j: gok += 1
        else: gf += 1
        ud = {"UserId":uid,"UserSignals":uer,"ReadableUserSignals":uer_readable,
              "events_list":el,"ground_truth_json":gt_json,
              "ground_truth_journeys":gt_j,"num_journeys":len(gt_j)}
        if has_profile: ud["UserProfile"] = r.get("Profile","")
        data.append(ud)
    print(f"    [{name}] {len(data):,} users, w/events: {len(data)-ne:,}, GT ok: {gok:,}, fail: {gf:,}")
    return data

# === Prompts ===

def make_journey_instruction(task, nj, mp, version="v4"):
    if task == "event2journey":
        op = f"Based on the user's shopping event history, predict {nj} shopping journey(s) the user is likely to pursue."
    else:
        op = f"Based on the user's shopping profile and shopping event history, predict {nj} shopping journey(s) the user is likely to pursue."
    if version == "v4":
        body = (f" Each journey has a JourneyType ('explicit' or 'related'),"
                f" a short engaging title, a user-centric reason,"
                f" and at least {mp} recommended products as text IDs (7 slots each)."
                f" Products within each journey must be diverse:"
                f" cover different brands, styles, use cases, and subcategories."
                f' Output JSON:'
                f' {{"ContinuedJourneys":[{{"JourneyType":"...","Title":"...","Reason":"...",'
                f'"ProductTIDs":[["s1","s2","s3","s4","s5","s6","s7"],...]}},...]}}.')
    else:
        body = (f" Each journey represents a different product category."
                f" Each journey has a short, engaging title, a brief user-centric reason"
                f" referencing the user's history, and at least {mp} recommended products"
                f" as text IDs (7 slots each)."
                f" Products within each journey should cover different brands, styles, use cases"
                f" and subcategories -- avoid recommending near-identical items."
                f' Output JSON:'
                f' {{"ContinuedJourneys":[{{"Title":"...","Reason":"...",'
                f'"ProductTIDs":[["s1","s2","s3","s4","s5","s6","s7"],...]}},...]}}.')
    jw = "journey" if nj == 1 else "journeys"
    pl = f"Predict the user's shopping journeys, exactly {nj} {jw}, at least {mp} products in each journey:"
    return op + body, pl

def build_e2j_input(events, max_events=100, prompt_line=None):
    fp = prompt_line or "Predict the user's shopping journeys:"
    lines = ["User Event History:"]
    for i, e in enumerate(events[:max_events], 1):
        e = normalize_event_times(e)
        if len(e) > 150: e = e[:150] + "..."
        lines.append(f"{i} | {e}")
    lines += ["", fp]
    return "\n".join(lines)

def build_p2j_input(profile, events, max_recent=100, prompt_line=None):
    fp = prompt_line or "Predict the user's shopping journeys:"
    clean_profile = _clean_profile_json(profile)
    lines = ["User Shopping Profile:", clean_profile, "", "Recent Shopping Events:"]
    for i, e in enumerate(events[:max_recent], 1):
        e = normalize_event_times(e)
        if len(e) > 150: e = e[:150] + "..."
        lines.append(f"{i} | {e}")
    lines += ["", fp]
    return "\n".join(lines)

# === vLLM ===

def run_vllm_inference(prompts, model_path, num_gpus, gpu_mem, max_model_len, max_tokens):
    from vllm import LLM, SamplingParams
    print(f"\nInitializing vLLM ...\n  Model: {model_path}\n  TP: {num_gpus}")
    llm = LLM(model=model_path, tensor_parallel_size=num_gpus,
              gpu_memory_utilization=gpu_mem, max_model_len=max_model_len,
              trust_remote_code=True, seed=SEED)
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.7, top_p=0.8, top_k=20)
    # Truncate prompts that exceed max input length using the engine's tokenizer
    _tok = llm.get_tokenizer()
    max_input = max_model_len - max_tokens
    truncated = 0
    for i, p in enumerate(prompts):
        tok_ids = _tok.encode(p)
        if len(tok_ids) > max_input:
            prompts[i] = _tok.decode(tok_ids[:max_input], skip_special_tokens=False)
            truncated += 1
    if truncated:
        print(f"  WARNING: Truncated {truncated} prompts to fit max_model_len={max_model_len}")
    import time; t0 = time.time()
    outputs = llm.generate(prompts, sp)
    el = time.time() - t0
    print(f"  Done in {el:.1f}s ({len(prompts)/el:.1f} items/s)")
    return [o.outputs[0].text.strip() for o in outputs]

def build_chat_prompts(ii, tokenizer, task):
    prompts = []
    for instr, inp in ii:
        msgs = [{"role":"user","content":instr+"\n"+inp}]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False,
            add_generation_prompt=True, enable_thinking=False))
    print(f"  Built {len(prompts)} prompts for {task}")
    return prompts

# === Output ===

def build_output_rows(ud_list, outputs, tid2item_id, rm, w2k, nkm, skm,
                      task_type, is_gt=False, id2title=None,
                      fuzzy_score_threshold=0.0, reorder_pos=-1):
    rows = []
    agg = {"total_users":len(ud_list),"json_parse_success":0,"json_parse_fail":0,
           "total_products":0,"exact_matches":0,"fuzzy_matches":0,"no_matches":0,
           "users_with_all_fields":0,"per_user_exact_ratios":[]}
    total = len(ud_list)
    label = f"{'GT' if is_gt else 'SLM'} {task_type}"
    for idx, ud in enumerate(ud_list):
        if total > 0 and (idx % max(total // 10, 1) == 0 or idx == total - 1):
            pct = (idx + 1) / total * 100
            print(f"\r    [{label}] {idx+1}/{total} ({pct:.0f}%)", end="", flush=True)
        uid = ud["UserId"]; ue = ud["UserSignals"]; uer = ud["ReadableUserSignals"]
        up = ud.get("UserProfile","")
        if is_gt:
            raw = ud.get("ground_truth_json","")
            gtj = ud.get("ground_truth_journeys",[])
            if gtj: agg["json_parse_success"] += 1
            else: agg["json_parse_fail"] += 1
            mapped = {"ContinuedJourneys":[]}; ut = 0
            for j in gtj:
                mj = {"Title":j.get("title",""),"Reason":j.get("reason",""),"Products":[]}
                for gid in j.get("product_ids",[]):
                    gs = str(gid); ut += 1
                    t = id2title.get(gs,"") if id2title else ""
                    mj["Products"].append({"GlobalOfferIds":[gs],"match_type":"exact","title":t})
                if mj["Products"]: mapped["ContinuedJourneys"].append(mj)
            agg["exact_matches"] += ut; agg["total_products"] += ut
            if ut > 0: agg["per_user_exact_ratios"].append(1.0)
            jj = json.dumps(mapped, ensure_ascii=False) if mapped["ContinuedJourneys"] else ""
            if uid and ue and mapped["ContinuedJourneys"]: agg["users_with_all_fields"] += 1
        else:
            raw = outputs[idx] if outputs else ""
            jd = parse_journey_json(raw)
            if jd: agg["json_parse_success"] += 1
            else: agg["json_parse_fail"] += 1
            md, ms = map_journey_products(jd, tid2item_id, rm, w2k, nkm, skm, id2title,
                fuzzy_score_threshold=fuzzy_score_threshold, reorder_pos=reorder_pos)
            for k in ("total_products","exact_matches","fuzzy_matches","no_matches"):
                agg[k] += ms[k]
            for k in ("fuzzy_filtered","journeys_dropped","products_deduped","journeys_title_deduped"):
                agg[k] = agg.get(k,0) + ms.get(k,0)
            agg.setdefault("fuzzy_matched_words",[]).extend(ms.get("fuzzy_matched_words",[]))
            agg.setdefault("fuzzy_best_scores",[]).extend(ms.get("fuzzy_best_scores",[]))
            ut, ue_ = ms["total_products"], ms["exact_matches"]
            if ut > 0: agg["per_user_exact_ratios"].append(ue_/ut)
            if uid and ue and raw and md: agg["users_with_all_fields"] += 1
            if not md: agg["users_no_valid_result"] = agg.get("users_no_valid_result",0)+1
            jj = json.dumps(md, ensure_ascii=False) if md else ""
        rows.append({"UserId":uid,"UserSignals":ue,"ReadableUserSignals":uer,
                     "UserProfile":up,"RawShoppingJourneys":raw,"ShoppingJourneys":jj})
    print()  # newline after progress
    return rows, agg

def save_tsv(rows, fp, cols):
    os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
    with open(fp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t",
                           quoting=csv.QUOTE_ALL, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            cr = {k:(v.replace("\n","\\n").replace("\r","\\r") if isinstance(v,str) else v) for k,v in r.items()}
            w.writerow(cr)
    print(f"  Saved {len(rows)} rows to: {fp}")

def _compute_row_stats(rows):
    """Compute diversity/size stats from output rows."""
    j_per_user, p_per_j, rp_per_j, u_prods = [], [], [], []
    rj_per_user = []  # raw journeys per user (from model output, before match)
    u_j_prods = []  # list of per-user [products_per_journey] lists
    dup_per_j = []   # per-journey duplicate TID count (from raw output)
    dup_ratio_per_j = []  # per-journey duplicate ratio
    for r in rows:
        jj = r.get("ShoppingJourneys","")
        if not jj: continue
        try: data = json.loads(jj)
        except: continue
        js = data.get("ContinuedJourneys",[])
        if not js: continue
        j_per_user.append(len(js))
        cnts = []
        ugids = []
        for j in js:
            prods = j.get("Products",[]); pids = j.get("ProductIds",[])
            if pids and not prods: n = len(pids)
            else: n = len(prods)
            p_per_j.append(n); cnts.append(n)
            for p in prods:
                gids = p.get("GlobalOfferIds",[])
                if gids: ugids.append(gids[0])
        u_j_prods.append(cnts)
        u_prods.append(len(set(ugids)))
        # raw products + duplicate TID stats + raw journeys count
        raw = r.get("RawShoppingJourneys","")
        if raw:
            rd = parse_journey_json(raw)
            if rd:
                raw_js = rd.get("ContinuedJourneys",[])
                rj_per_user.append(len(raw_js))
                for rj in raw_js:
                    tids = rj.get("ProductTIDs",[])
                    rp_per_j.append(len(tids))
                    # Count duplicates: TIDs that appear more than once
                    tid_strs = [",".join(t) if isinstance(t,list) else str(t) for t in tids]
                    n_total = len(tid_strs)
                    n_unique = len(set(tid_strs))
                    n_dup = n_total - n_unique
                    dup_per_j.append(n_dup)
                    dup_ratio_per_j.append(n_dup / n_total if n_total > 0 else 0.0)
    return {"j_per_user":j_per_user,"rj_per_user":rj_per_user,
            "p_per_j":p_per_j,"rp_per_j":rp_per_j,
            "u_prods":u_prods,"u_j_prods":u_j_prods,"n_users":len(j_per_user),
            "dup_per_j":dup_per_j,"dup_ratio_per_j":dup_ratio_per_j}

def print_side_by_side(task, ls, ss, llm_rows, slm_rows):
    print(f"\n  {'='*80}")
    print(f"  {task} - LLM vs SLM")
    print(f"  {'='*80}")
    def _pk(s): return s["exact_matches"]+s["fuzzy_matches"]-s.get("fuzzy_filtered",0)-s.get("products_deduped",0)

    print(f"\n  {'Metric':<45s} {'LLM':>20s} {'SLM':>20s}")
    print(f"  {'-'*45} {'-'*20} {'-'*20}")
    for lb, k in [("Total users","total_users"),("JSON parse success","json_parse_success"),
                  ("JSON parse fail","json_parse_fail"),
                  ("Total products (model output)","total_products")]:
        print(f"  {lb:<45s} {ls.get(k,0):>20,} {ss.get(k,0):>20,}")

    # Rows with percentage relative to total_products
    lt = max(ls.get("total_products",0), 1)
    st = max(ss.get("total_products",0), 1)
    def _vp(s, k, tot):
        v = s.get(k, 0)
        return f"{v:,} ({v/tot*100:.1f}%)"
    print(f"  {'Exact matches':<45s} {_vp(ls,'exact_matches',lt):>20s} {_vp(ss,'exact_matches',st):>20s}")
    sfk = ss.get("fuzzy_matches",0) - ss.get("fuzzy_filtered",0)
    print(f"  {'Fuzzy matches (kept)':<45s} {_vp(ls,'fuzzy_matches',lt):>20s} {f'{sfk:,} ({sfk/st*100:.1f}%)':>20s}")
    print(f"  {'No matches':<45s} {_vp(ls,'no_matches',lt):>20s} {_vp(ss,'no_matches',st):>20s}")
    print(f"  {'Fuzzy filtered (< threshold)':<45s} {_vp(ls,'fuzzy_filtered',lt):>20s} {_vp(ss,'fuzzy_filtered',st):>20s}")
    print(f"  {'Products deduped (by GID)':<45s} {_vp(ls,'products_deduped',lt):>20s} {_vp(ss,'products_deduped',st):>20s}")
    print(f"  {'Journeys dropped (empty)':<45s} {ls.get('journeys_dropped',0):>20,} {ss.get('journeys_dropped',0):>20,}")
    lpk = _pk(ls); spk = _pk(ss)
    print(f"  {'Products kept (after filtering)':<45s} {f'{lpk:,} ({lpk/lt*100:.1f}%)':>20s} {f'{spk:,} ({spk/st*100:.1f}%)':>20s}")
    print(f"  {'Users with all fields':<45s} {ls['users_with_all_fields']:>20,} {ss['users_with_all_fields']:>20,}")

    # Diversity & Size stats (side by side)
    ld = _compute_row_stats(llm_rows)
    sd = _compute_row_stats(slm_rows)
    print(f"\n  {'--- Diversity & Size ---':<45s} {'LLM':>20s} {'SLM':>20s}")
    print(f"  {'Users with data':<45s} {ld['n_users']:>20,} {sd['n_users']:>20,}")

    def _fmt(arr):
        if not arr: return "N/A"
        return f"{np.array(arr).mean():.2f}"

    print(f"  {'Journeys/user after match (mean)':<45s} {_fmt(ld['j_per_user']):>20s} {_fmt(sd['j_per_user']):>20s}")
    print(f"  {'Journeys/user before match (mean)':<45s} {_fmt(ld['rj_per_user']):>20s} {_fmt(sd['rj_per_user']):>20s}")
    print(f"  {'Products/journey after match (mean)':<45s} {_fmt(ld['p_per_j']):>20s} {_fmt(sd['p_per_j']):>20s}")
    print(f"  {'Products/journey before match (mean)':<45s} {_fmt(ld['rp_per_j']):>20s} {_fmt(sd['rp_per_j']):>20s}")
    print(f"  {'Unique products/user (mean)':<45s} {_fmt(ld['u_prods']):>20s} {_fmt(sd['u_prods']):>20s}")

    # Duplicate TIDs per journey
    def _fmt_dup(darr, rarr):
        if not darr: return "N/A"
        da = np.array(darr); ra = np.array(rarr)
        return f"{da.mean():.1f} ({ra.mean():.1%})"
    print(f"  {'Dup TIDs/journey (mean, ratio)':<45s} {_fmt_dup(ld['dup_per_j'],ld['dup_ratio_per_j']):>20s} {_fmt_dup(sd['dup_per_j'],sd['dup_ratio_per_j']):>20s}")

    # Coverage with percentages
    for min_p in [5, 8, 10]:
        lj = np.array(ld['p_per_j']) if ld['p_per_j'] else np.array([])
        sj = np.array(sd['p_per_j']) if sd['p_per_j'] else np.array([])
        lj_ok = int((lj >= min_p).sum()) if len(lj) else 0
        sj_ok = int((sj >= min_p).sum()) if len(sj) else 0
        lt2 = max(len(lj), 1); st2 = max(len(sj), 1)
        lp = f"{lj_ok}/{len(lj)} ({lj_ok/lt2*100:.1f}%)" if len(lj) else "N/A"
        sp = f"{sj_ok}/{len(sj)} ({sj_ok/st2*100:.1f}%)" if len(sj) else "N/A"
        print(f"  {'Journeys >= '+str(min_p)+' products':<45s} {lp:>20s} {sp:>20s}")
    for min_p in [5, 8, 10]:
        lu = sum(1 for c in ld['u_j_prods'] if any(x>=min_p for x in c))
        su = sum(1 for c in sd['u_j_prods'] if any(x>=min_p for x in c))
        lt2 = max(len(ld['u_j_prods']), 1); st2 = max(len(sd['u_j_prods']), 1)
        lp = f"{lu}/{len(ld['u_j_prods'])} ({lu/lt2*100:.1f}%)" if ld['u_j_prods'] else "N/A"
        sp = f"{su}/{len(sd['u_j_prods'])} ({su/st2*100:.1f}%)" if sd['u_j_prods'] else "N/A"
        print(f"  {'Users w/ any journey >= '+str(min_p)+' prods':<45s} {lp:>20s} {sp:>20s}")

    return {"llm":{k:v for k,v in ls.items() if k not in ("per_user_exact_ratios","fuzzy_matched_words","fuzzy_best_scores")},
            "slm":{k:v for k,v in ss.items() if k not in ("per_user_exact_ratios","fuzzy_matched_words","fuzzy_best_scores")}}

# === Args ===

def parse_args():
    p = argparse.ArgumentParser(description="Split-task journey evaluation")
    p.add_argument(
        "--model_path", type=str, 
        #default="/scratch/workspaceblobstore/users/wangying/LlamaFactory/saves/journey_v3_cp1200/lora_journey_v3/sft_4gpus_lr5e-5_batch12_gradacc2_lorarank32_cut4096_packing_enablethinkingfalse/checkpoint-9000-merged", # demo ckpt
        #default="/scratch/workspaceblobstore/users/wangying/LlamaFactory/saves/all_termId_ckpt21019/lora_journey_v4/sft_4gpus_lr5e-5_batch1_gradacc16_lorarank64_cut32768_packing_enablethinkingfalse/checkpoint-1000-merged",
        #default="/scratch/workspaceblobstore/users/wangying/LlamaFactory/saves/qwen3-5-9b/lora_journey_v4/sft_4gpus_lr5e-5_batch1_gradacc16_lorarank32_cut32768_packing_enablethinkingfalse/checkpoint-1000-merged",
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Results/qwen3-5-9b_lora_v4/merged_checkpoint_final",
        #default="/scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore/users/wangying/LlamaFactory/saves/journeyv4_step1_le4096_ckpt4768/lora_journey_v4_step2_v1sample/sft_4gpus_lr2e-5_batch8_gradacc2_lorarank64_cut32768_enableligerkernel_true_neatpacking_false_flashattn_fa2_enablethinkingfalse/checkpoint-475-merged",
        #default="/scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore/users/wangying/LlamaFactory/saves/journeyv4_step1_le4096_ckpt4768/lora_journey_v4_step2_v1sample/sft_4gpus_lr2e-5_batch8_gradacc2_lorarank64_cut32768_enableligerkernel_true_neatpacking_false_flashattn_fa2_enablethinkingfalse_epoch3.0/checkpoint-800-merged",
        #default="/scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore/users/wangying/LlamaFactory/saves/all_termId_ckpt21019/lora_journey_v4_step1_le4096/sft_4gpus_lr5e-5_batch12_gradacc2_lorarank64_cut4096_enableligerkernel_true_neatpacking_false_flashattn_fa2_enablethinkingfalse/checkpoint-4768-merged",
        help="Path to the trained SFT model checkpoint",
    )
    p.add_argument("--test_dir", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260407/sft_data")
    p.add_argument(
        "--output_dir", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/eval_results/qwen3-5-9b_lora_v4_checkpoint-final/",
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/eval_results/v4_ying_9B_checkpoint-1000_termid_pretrained/",
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/eval_results/demo_ckpt/match_6_reorder_tid/",
        #default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/eval_results/v4_ying_9B_checkpoint-800/",
        help="Directory to save evaluation output files",
    )
    p.add_argument("--tid2item_id_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/sft_data_v4/item_id2tid/tid2item_id.json")
    p.add_argument("--sample_n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_events", type=int, default=500)
    p.add_argument("--max_recent_events", type=int, default=500)
    p.add_argument("--num_gpus", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    p.add_argument("--max_model_len", type=int, default=32000)
    p.add_argument("--max_tokens", type=int, default=12000)
    p.add_argument("--item_file", type=str,
        default="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/raw_data/item.json")
    p.add_argument("--fuzzy_score_threshold", type=float, default=6.0)
    p.add_argument("--reorder_tid_pos", type=int, default=3)
    p.add_argument("--instruction_version", type=str, default="v4", choices=["v3","v4"])
    p.add_argument("--min_products_override", type=int, default=10)
    return p.parse_args()

# === Main ===

def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_gpus = args.num_gpus if args.num_gpus else max(gpus, 1)

    print("="*70)
    print("Step 5.3: Split-Task Journey Evaluation")
    print("="*70)
    print(f"  Model:    {args.model_path}")
    print(f"  Test dir: {args.test_dir}")
    print(f"  GPUs:     {num_gpus}")
    print(f"  Sample N: {args.sample_n}")

    # Step 1: Read test TSVs
    print(f"\n{'='*70}\nStep 1: Reading test TSV files\n{'='*70}")
    e2j_tsv = os.path.join(args.test_dir, "event2journey_full_cleaned_test.tsv")
    p2j_tsv = os.path.join(args.test_dir, "profile2journey_full_cleaned_test.tsv")
    e2j_rows, e2j_cols = read_test_tsv(e2j_tsv)
    print(f"  [event2journey]   {len(e2j_rows):,} users, columns: {e2j_cols}")
    p2j_rows, p2j_cols = read_test_tsv(p2j_tsv)
    print(f"  [profile2journey] {len(p2j_rows):,} users, columns: {p2j_cols}")

    # Step 2: Sample
    print(f"\n{'='*70}\nStep 2: Sampling users\n{'='*70}")
    e2j_n = min(args.sample_n, len(e2j_rows))
    e2j_sampled = sorted(random.sample(e2j_rows, e2j_n), key=lambda r: r.get("UserId",""))
    print(f"  [event2journey]   Sampled {e2j_n:,} from {len(e2j_rows):,}")
    p2j_n = min(args.sample_n, len(p2j_rows))
    p2j_sampled = sorted(random.sample(p2j_rows, p2j_n), key=lambda r: r.get("UserId",""))
    print(f"  [profile2journey] Sampled {p2j_n:,} from {len(p2j_rows):,}")

    # Step 3: Process
    print(f"\n{'='*70}\nStep 3: Processing user data\n{'='*70}")
    e2j_data = process_task_data("event2journey", e2j_sampled, has_profile=False)
    p2j_data = process_task_data("profile2journey", p2j_sampled, has_profile=True)

    # Event count statistics
    for name, data in [("event2journey", e2j_data), ("profile2journey", p2j_data)]:
        ec = np.array([len(ud.get("events_list", [])) for ud in data])
        if len(ec):
            print(f"    [{name}] Events per user: "
                  f"Mean={ec.mean():.1f}, Min={ec.min()}, "
                  f"P25={int(np.percentile(ec,25))}, P50={int(np.percentile(ec,50))}, "
                  f"P75={int(np.percentile(ec,75))}, Max={ec.max()}")

    # Step 4: TID mapping
    print(f"\n{'='*70}\nStep 4: Loading TID mapping\n{'='*70}")
    with open(args.tid2item_id_file, "r", encoding="utf-8") as f:
        tid2item_id = json.load(f)
    rm, w2k, nkm, skm = create_reverse_mapping(tid2item_id, reorder_pos=args.reorder_tid_pos)
    print(f"  TIDs: {len(tid2item_id):,}, Words: {len(w2k):,}")
    if args.reorder_tid_pos >= 0: print(f"  Reorder: slot {args.reorder_tid_pos}")
    id2title = load_item_titles(args.item_file) if args.item_file and os.path.isfile(args.item_file) else None
    ma = dict(tid2item_id=tid2item_id, rm=rm, w2k=w2k, nkm=nkm, skm=skm, id2title=id2title)

    # Step 5: LLM GT
    print(f"\n{'='*70}\nStep 5: Building LLM ground truth\n{'='*70}")
    e2j_lr, e2j_ls = build_output_rows(e2j_data, None, is_gt=True, task_type="event2journey", **ma)
    p2j_lr, p2j_ls = build_output_rows(p2j_data, None, is_gt=True, task_type="profile2journey", **ma)
    print(f"  [event2journey]   GT products: {e2j_ls['total_products']:,}")
    print(f"  [profile2journey] GT products: {p2j_ls['total_products']:,}")

    # Step 6: vLLM
    print(f"\n{'='*70}\nStep 6: Running vLLM inference\n{'='*70}")
    from transformers import AutoTokenizer
    tp = args.model_path
    tok_cfg = os.path.join(tp, "tokenizer_config.json")
    if os.path.isfile(tok_cfg):
        with open(tok_cfg,"r") as f: tc = json.load(f)
        if tc.get("tokenizer_class") not in ("Qwen2Tokenizer","PreTrainedTokenizerFast",None):
            print(f"  [FIX] tokenizer_class -> 'Qwen2Tokenizer'")
            tc["tokenizer_class"] = "Qwen2Tokenizer"
            with open(tok_cfg,"w") as f: json.dump(tc,f,indent=2,ensure_ascii=False)
    tokenizer = AutoTokenizer.from_pretrained(tp, trust_remote_code=True)

    ver = args.instruction_version; mpo = args.min_products_override
    def _gmp(ud):
        if mpo > 0: return mpo
        gj = ud.get("ground_truth_journeys",[])
        return max(min(len(j.get("product_ids",[])) for j in gj),5) if gj else 5

    print(f"  Instruction: {ver}, min_products: {mpo if mpo>0 else 'from GT'}")

    e2j_ii = []
    for ud in e2j_data:
        instr, pl = make_journey_instruction("event2journey", ud["num_journeys"], _gmp(ud), version=ver)
        e2j_ii.append((instr, build_e2j_input(ud["events_list"], args.max_events, pl)))
    e2j_prompts = build_chat_prompts(e2j_ii, tokenizer, "event2journey")

    p2j_ii = []
    for ud in p2j_data:
        instr, pl = make_journey_instruction("profile2journey", ud["num_journeys"], _gmp(ud), version=ver)
        p2j_ii.append((instr, build_p2j_input(ud.get("UserProfile",""), ud["events_list"], args.max_recent_events, pl)))
    p2j_prompts = build_chat_prompts(p2j_ii, tokenizer, "profile2journey")

    all_prompts = e2j_prompts + p2j_prompts
    print(f"\n  Total: {len(all_prompts)} ({len(e2j_prompts)} e2j + {len(p2j_prompts)} p2j)")
    all_out = run_vllm_inference(all_prompts, args.model_path, num_gpus,
                                args.gpu_memory_utilization, args.max_model_len, args.max_tokens)
    e2j_out = all_out[:len(e2j_prompts)]
    p2j_out = all_out[len(e2j_prompts):]

    # Step 7: SLM outputs + TID matching + comparison
    print(f"\n{'='*70}\nStep 7: Building SLM outputs + TID matching\n{'='*70}")
    e2j_sr, e2j_ss = build_output_rows(e2j_data, e2j_out, is_gt=False, task_type="event2journey",
        fuzzy_score_threshold=args.fuzzy_score_threshold, reorder_pos=args.reorder_tid_pos, **ma)
    e2j_sum = print_side_by_side("event2journey", e2j_ls, e2j_ss, e2j_lr, e2j_sr)

    p2j_sr, p2j_ss = build_output_rows(p2j_data, p2j_out, is_gt=False, task_type="profile2journey",
        fuzzy_score_threshold=args.fuzzy_score_threshold, reorder_pos=args.reorder_tid_pos, **ma)
    p2j_sum = print_side_by_side("profile2journey", p2j_ls, p2j_ss, p2j_lr, p2j_sr)

    # Step 8: Save output files + summary
    print(f"\n{'='*70}\nStep 8: Saving output files\n{'='*70}")
    os.makedirs(args.output_dir, exist_ok=True)
    cols = ["UserId","UserSignals","ReadableUserSignals","UserProfile","RawShoppingJourneys","ShoppingJourneys"]
    save_tsv(e2j_lr, os.path.join(args.output_dir, "event2journey_llm_output.tsv"), cols)
    save_tsv(e2j_sr, os.path.join(args.output_dir, "event2journey_slm_output.tsv"), cols)
    save_tsv(p2j_lr, os.path.join(args.output_dir, "profile2journey_llm_output.tsv"), cols)
    save_tsv(p2j_sr, os.path.join(args.output_dir, "profile2journey_slm_output.tsv"), cols)

    summary = {"model_path":args.model_path,"seed":args.seed,
               "instruction_version":ver,"min_products_override":mpo,
               "fuzzy_score_threshold":args.fuzzy_score_threshold,
               "reorder_tid_pos":args.reorder_tid_pos,
               "event2journey":{"sampled_users":len(e2j_data),**e2j_sum},
               "profile2journey":{"sampled_users":len(p2j_data),**p2j_sum}}
    sf = os.path.join(args.output_dir, "eval_summary.json")
    with open(sf,"w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    print(f"\n  Summary: {sf}")
    print(f"\nDone! {len(e2j_data)} e2j + {len(p2j_data)} p2j users -> {args.output_dir}")

if __name__ == "__main__":
    main()

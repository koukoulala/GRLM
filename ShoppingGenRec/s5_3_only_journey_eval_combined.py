"""Step 5.3 Combined: vLLM Journey Inference + GPT 3-Dimension Labeling

Phase 1 — vLLM Inference:
  1. Read input TSV (slm_output_w_profile.tsv).
  2. Sample N users.
  3. Build profile2journey prompts (only-journey format).
  4. Run vLLM inference → parse JSON → save results TSV.

Phase 2 — GPT Labeling (3 dimensions):
  5. Read the Phase-1 output TSV (ParsedJourneys column).
  6. For each of Diversity / Quality / Relevance:
     a. Build eval prompts from prompt templates.
     b. Call GitHub Copilot LLM in parallel.
     c. Parse results, save intermediate JSONL.
  7. Print consolidated report + save eval_report.json.

Usage:
    # Full pipeline (inference + labeling):
    python s5_3_only_journey_eval_combined.py

    # Skip inference, only labeling (provide existing TSV):
    python s5_3_only_journey_eval_combined.py --skip_inference \\
        --inference_output /path/to/existing_output.tsv

    # Skip labeling, only inference:
    python s5_3_only_journey_eval_combined.py --skip_labeling

    # Debug (10 users):
    python s5_3_only_journey_eval_combined.py --debug --debug_rows 10
"""

import os, re, csv, json, random, argparse, sys, time
from collections import defaultdict
import numpy as np

csv.field_size_limit(sys.maxsize)

# =============================================================================
# DEFAULT CONFIGURATION — Change these before running
# =============================================================================

# ── Phase 1: vLLM Inference ──
DEFAULT_MODEL_PATH = (
    "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Results_journey_w_query/qwen3-5-9b_full_500K_v2/checkpoint-1500/"
    #"/scratch/workspaceblobstore/users/wangying/LlamaFactory/saves/shopping_only_journey_v2_s1/lora_shopping_only_journey_v2_s2/sft_4gpus_lr5e-5_batch8_gradacc2_lorarank64_cut32768_enableligerkernel_true_epoch2.0_flashattn_fa2/checkpoint-200-merged"
)
DEFAULT_INPUT_TSV = (
    "/scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore/users/"
    "yishengchen/ShoppingRecoRelevance/src/model/LLM/shoppingJourney/"
    "data/test_copilot_shopping_homepage_sample/s2_3epoch_ckpt1425/"
    "slm_output_w_profile.tsv"
)
DEFAULT_OUTPUT_DIR = (
    "/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/"
    "LLMTrainingData/EvalData/copilot_shopping_homepage_sample/"
)
DEFAULT_OUTPUT_SUFFIX = (
    "xiaoyu_only_journey_datav2_ckpt_1500"
)

# DEFAULT_OUTPUT_SUFFIX = (
#     "only_journey_datav1_500K_9b_full_ckpt720"
# )

DEFAULT_SAMPLE_N       = 50
DEFAULT_SEED           = 42
DEFAULT_MAX_RECENT     = 500
DEFAULT_GPU_MEM_UTIL   = 0.85
DEFAULT_MAX_MODEL_LEN  = 32768
DEFAULT_MAX_TOKENS_VLLM = 16384

# ── Phase 2: GPT Labeling ──
DEFAULT_LABELING_MODEL      = "gpt-5.2"
DEFAULT_LABELING_WORKERS    = 100
DEFAULT_LABELING_MAX_TOKENS = 8000
DEFAULT_LABELING_CHUNK_SIZE = 1000

# ── Shared Paths ──
LLM_UTILS_DIR = "/scratch/workspaceblobstore/users/wangying/OneRec/LLMCall"
TOKEN_FILE    = os.path.join(LLM_UTILS_DIR, "github_all.txt")
PROMPT_DIR    = "/scratch/workspaceblobstore/users/wangying/OneRec/Journey/Prompt"

# ── Debug ──
DEFAULT_DEBUG_ROWS = 50

# =============================================================================

SEED = DEFAULT_SEED
sys.path.insert(0, LLM_UTILS_DIR)


# =============================================================================
# Phase 1: vLLM Inference — Helpers
# =============================================================================

_RE_EVENT_NUMBER = re.compile(r"^\d+\s*\|\s*(.*)")
_RE_NON_ALNUM = re.compile(r"[^a-z0-9\s|]")
_RE_MULTI_SPACE = re.compile(r"\s+")


def _normalize_time_expr(match):
    text = match.group(0)
    parts = re.findall(r'(\d+)\s*(month|week|day|hour|minute|second)s?', text, re.IGNORECASE)
    if not parts:
        return text
    total_hours = 0; total_minutes = 0
    for num_str, unit in parts:
        num = int(num_str); u = unit.lower()
        if u == 'month':   total_hours += num * 30 * 24
        elif u == 'week':  total_hours += num * 7 * 24
        elif u == 'day':   total_hours += num * 24
        elif u == 'hour':  total_hours += num
        elif u == 'minute': total_minutes += num
    total_days = total_hours // 24
    if total_days > 0:      return f"{total_days} days ago"
    elif total_hours > 0:   return f"{total_hours} hours ago"
    elif total_minutes > 0: return f"{total_minutes} minutes ago"
    return "0 minutes ago"


def normalize_event_times(text):
    return re.sub(r'(?:\d+\s*(?:month|week|day|hour|minute|second)s?\s*)+ago',
                  _normalize_time_expr, text, flags=re.IGNORECASE)


def _clean_profile_json(raw):
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


def parse_readable_user_events(events_text):
    if not events_text or not events_text.strip():
        return [], 0
    text = events_text.replace("\\n", "\n").replace("#N#", "\n")
    lines = text.strip().split("\n")
    raw_events = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = _RE_EVENT_NUMBER.match(line)
        if match:
            event = match.group(1).strip()
            if event:
                raw_events.append(event)
    seen_keys = set(); deduped = []
    for event in raw_events:
        key = _RE_MULTI_SPACE.sub(" ", _RE_NON_ALNUM.sub(" ", event.lower())).strip()
        if key not in seen_keys:
            seen_keys.add(key); deduped.append(event)
    return deduped, len(raw_events)


def parse_journey_json(raw):
    """Parse ContinuedJourneys JSON from raw model output."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    bs = text.find("{")
    if bs == -1:
        return None
    d, be = 0, -1
    for i in range(bs, len(text)):
        if text[i] == "{":   d += 1
        elif text[i] == "}":
            d -= 1
            if d == 0: be = i; break
    cand = text[bs:be+1] if be != -1 else text[bs:] + "}"
    for t in [cand, text]:
        try:
            data = json.loads(t)
            if "ContinuedJourneys" in data:
                return data
        except Exception:
            pass
    return None


# ── Phase 1: Data reading ──

def read_test_tsv(fp):
    rows, seen = [], set()
    with open(fp, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if not header:
            raise ValueError(f"Empty: {fp}")
        cm = {n.strip().strip('"'): i for i, n in enumerate(header)}
        for row in reader:
            ui = cm.get("UserId", 0)
            if len(row) <= ui:
                continue
            uid = row[ui].strip().strip('"')
            if not uid or uid in seen:
                continue
            seen.add(uid)
            rd = {}
            for n, i in cm.items():
                rd[n] = row[i].strip().strip('"') if i < len(row) else ""
            rows.append(rd)
    return rows, list(cm.keys())


def process_data(rows):
    data = []; ne = 0
    for r in rows:
        uid = r["UserId"]
        rr = r.get("ReadableUserSignals", "")
        el, _ = parse_readable_user_events(rr)
        if not el:
            ne += 1
        rl = [f"{i+1} | {(e[:150]+'...' if len(e) > 150 else e)}" for i, e in enumerate(el)]
        uer_readable = "\n".join(rl)
        data.append({
            "UserId": uid,
            "UserSignals": r.get("UserSignals", ""),
            "ReadableUserSignals": uer_readable,
            "UserProfile": r.get("UserProfile", ""),
            "events_list": el,
        })
    print(f"    {len(data):,} users, w/events: {len(data)-ne:,}, no events: {ne:,}")
    return data


# ── Phase 1: Prompt building ──

# ── v1 prompt (ConversationStarters list + Reason) ──
# INSTRUCTION = (
#     "Based on the user's shopping profile and shopping event history, predict "
#     "an appropriate number of shopping journey(s) the user is likely to pursue."
#     " Each journey has a JourneyType ('explicit' or 'related'),"
#     " a short engaging Title,"
#     " a Description (2-3 sentences in personal-shopper tone highlighting"
#     " why this journey fits the user and what value exploring it brings),"
#     " a list of ConversationStarters (3 natural first-person openings"
#     " that resume the shopping journey),"
#     " a set of Queries (3-7 concise product search queries),"
#     " and a Reason (explains which user signals triggered this journey)."
#     ' Output JSON:'
#     ' {"ContinuedJourneys":[{"JourneyType":"...","Title":"...",'
#     '"Description":"...","ConversationStarter":["...","...","..."],'
#     '"Queries":[{"Query":"..."},...],"Reason":"..."},...]}'
# )

# ── v2 prompt (single ConversationStarter + WhyAmISeeingThis) ──
INSTRUCTION = (
    "Based on the user's shopping profile and shopping event history, predict "
    "an appropriate number of shopping journey(s) the user is likely to pursue."
    " Each journey has a JourneyType ('explicit' or 'related'),"
    " a short engaging Title,"
    " a Description (2-3 sentences in personal-shopper tone highlighting"
    " why this journey fits the user and what value exploring it brings),"
    " a ConversationStarter (a natural first-person opening"
    " that resumes the shopping journey),"
    " a set of Queries (3-7 concise product search queries),"
    " and a WhyAmISeeingThis (explains which user signals triggered this journey)."
    ' Output JSON:'
    ' {"ContinuedJourneys":[{"JourneyType":"...","Title":"...",'
    '"Description":"...","ConversationStarter":"...",'
    '"Queries":[{"Query":"..."},...],"WhyAmISeeingThis":"..."},...]}'
)

PROMPT_LINE = "Predict an appropriate number of shopping journeys:"


def build_input(profile, events, max_recent=500):
    clean_profile = _clean_profile_json(profile)
    lines = ["User Shopping Profile:", clean_profile, "", "User Event History:"]
    for i, e in enumerate(events[:max_recent], 1):
        e = normalize_event_times(e)
        if len(e) > 150:
            e = e[:150] + "..."
        lines.append(f"{i} | {e}")
    lines += ["", PROMPT_LINE]
    return "\n".join(lines)


def build_chat_prompts(data_list, tokenizer, max_recent=500):
    prompts = []
    for ud in data_list:
        inp = build_input(ud["UserProfile"], ud["events_list"], max_recent)
        msgs = [{"role": "user", "content": INSTRUCTION + "\n" + inp}]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    print(f"  Built {len(prompts)} prompts")
    return prompts


# ── Phase 1: vLLM ──

def run_vllm_inference(prompts, model_path, num_gpus, gpu_mem, max_model_len, max_tokens):
    from vllm import LLM, SamplingParams
    print(f"\nInitializing vLLM ...\n  Model: {model_path}\n  TP: {num_gpus}")
    llm = LLM(model=model_path, tensor_parallel_size=num_gpus,
              gpu_memory_utilization=gpu_mem, max_model_len=max_model_len,
              max_num_seqs=64, trust_remote_code=True, seed=SEED)
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.7, top_p=0.8,
                        top_k=20, repetition_penalty=1.00)
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
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    el = time.time() - t0
    print(f"  Done in {el:.1f}s ({len(prompts)/el:.1f} items/s)")
    return [o.outputs[0].text.strip() for o in outputs]


# ── Phase 1: Analysis ──

def analyze_outputs(data_list, raw_outputs):
    results = []
    stats = {
        "total_users": len(data_list), "json_parse_success": 0, "json_parse_fail": 0,
        "journeys_per_user": [], "queries_per_journey": [], "starters_per_journey": [],
        "journey_types": defaultdict(int), "has_description": 0, "has_reason": 0,
        "total_journeys": 0,
    }
    for idx, ud in enumerate(data_list):
        raw = raw_outputs[idx]
        jd = parse_journey_json(raw)
        if jd:
            stats["json_parse_success"] += 1
            journeys = jd.get("ContinuedJourneys", [])
            stats["journeys_per_user"].append(len(journeys))
            stats["total_journeys"] += len(journeys)
            for j in journeys:
                jtype = j.get("JourneyType", "").strip().lower()
                if jtype: stats["journey_types"][jtype] += 1
                stats["queries_per_journey"].append(len(j.get("Queries", [])))
                starters = j.get("ConversationStarter", j.get("ConversationStarters", []))
                if isinstance(starters, str):
                    stats["starters_per_journey"].append(1 if starters.strip() else 0)
                else:
                    stats["starters_per_journey"].append(len(starters))
                if j.get("Description", "").strip(): stats["has_description"] += 1
                if j.get("Reason", j.get("WhyAmISeeingThis", "")).strip(): stats["has_reason"] += 1
        else:
            stats["json_parse_fail"] += 1
            stats["journeys_per_user"].append(0)
        results.append({
            "UserId": ud["UserId"], "UserSignals": ud["UserSignals"],
            "ReadableUserSignals": ud["ReadableUserSignals"],
            "UserProfile": ud["UserProfile"], "RawOutput": raw,
            "ParsedJourneys": json.dumps(jd, ensure_ascii=False) if jd else "",
        })
    return results, stats


def print_inference_stats(stats):
    total = stats["total_users"]
    ok = stats["json_parse_success"]
    fail = stats["json_parse_fail"]
    print(f"  Total users: {total:,}, JSON parse ok: {ok:,}, fail: {fail:,}")
    print(f"  Total journeys: {stats['total_journeys']:,}")
    jpu = np.array(stats["journeys_per_user"])
    if len(jpu):
        print(f"  Journeys/user: Mean={jpu.mean():.2f}, Median={np.median(jpu):.1f}, "
              f"Min={jpu.min()}, Max={jpu.max()}")
    qpj = np.array(stats["queries_per_journey"])
    if len(qpj):
        print(f"  Queries/journey: Mean={qpj.mean():.2f}, Min={qpj.min()}, Max={qpj.max()}")
    tj = max(stats["total_journeys"], 1)
    print(f"  Description present: {stats['has_description']}/{tj} ({stats['has_description']/tj*100:.1f}%)")
    print(f"  Reason present: {stats['has_reason']}/{tj} ({stats['has_reason']/tj*100:.1f}%)")


def save_tsv(rows, fp, cols):
    os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
    with open(fp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t",
                           quoting=csv.QUOTE_ALL, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            cr = {k: (v.replace("\n", "\\n").replace("\r", "\\r")
                       if isinstance(v, str) else v) for k, v in r.items()}
            w.writerow(cr)
    print(f"  Saved {len(rows)} rows to: {fp}")


# =============================================================================
# Phase 2: GPT Labeling — Helpers
# =============================================================================

# (PROMPT_DIR and TOKEN_FILE defined in config block at top)


def load_prompt_template(name):
    fp = os.path.join(PROMPT_DIR, name)
    with open(fp, "r", encoding="utf-8") as f:
        return f.read()


def read_eval_data(filepath, max_rows=0):
    """Read output TSV with ParsedJourneys + ReadableUserSignals for labeling."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            uid = (row.get("UserId") or "").strip().strip('"')
            pj_raw = (row.get("ParsedJourneys") or "").strip().strip('"')
            signals = (row.get("ReadableUserSignals") or "").strip().strip('"')
            if not uid or not pj_raw:
                continue
            try:
                pj = json.loads(pj_raw)
            except json.JSONDecodeError:
                continue
            cjs = pj.get("ContinuedJourneys", [])
            if not cjs:
                continue
            rows.append({
                "UserId": uid, "ParsedJourneys": pj,
                "ContinuedJourneys": cjs, "ReadableUserSignals": signals,
            })
            if max_rows > 0 and len(rows) >= max_rows:
                break
    print(f"  Loaded {len(rows):,} users with valid journeys")
    return rows


def format_journeys_for_eval(cjs):
    items = []
    for j in cjs:
        reason = j.get("Reason", "") or j.get("WhyAmISeeingThis", "")
        items.append({"journeyTitle": j.get("Title", ""), "journeyReason": reason})
    return json.dumps(items, ensure_ascii=False)


def build_diversity_prompts(rows, template):
    inputs = []
    for r in rows:
        jstr = format_journeys_for_eval(r["ContinuedJourneys"])
        prompt = template.replace("#ShoppingJourneys#", jstr)
        inputs.append((r["UserId"], prompt))
    return inputs


def build_quality_prompts(rows, template):
    inputs = []
    for r in rows:
        jstr = format_journeys_for_eval(r["ContinuedJourneys"])
        signals = r["ReadableUserSignals"].replace("\\n", "\n")
        prompt = template.replace("#UserSignals#", signals).replace("#ShoppingJourneys#", jstr)
        inputs.append((r["UserId"], prompt))
    return inputs


def build_relevance_prompts(rows, template):
    inputs = []
    for r in rows:
        jstr = format_journeys_for_eval(r["ContinuedJourneys"])
        signals = r["ReadableUserSignals"].replace("\\n", "\n")
        prompt = template.replace("##UserSignals##", signals).replace("##ShoppingJourneys##", jstr)
        inputs.append((r["UserId"], prompt))
    return inputs


def extract_output_json(raw_text):
    if not raw_text or not raw_text.strip():
        return None
    text = raw_text.strip()
    tag_match = re.search(r'<OUTPUT>\s*(.*?)\s*</OUTPUT>', text, re.DOTALL)
    if tag_match:
        text = tag_match.group(1).strip()
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*$', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for sc, ec in [('{', '}'), ('[', ']')]:
        s, e = text.find(sc), text.rfind(ec)
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e+1])
            except json.JSONDecodeError:
                pass
    return None


def parse_diversity_result(raw):
    obj = extract_output_json(raw)
    if not obj or not isinstance(obj, dict):
        return None
    return {"diversityScore": obj.get("diversityScore"),
            "journeyGroups": obj.get("journeyGroups", []),
            "diversityExplanation": obj.get("diversityExplanation", "")}


def parse_quality_result(raw):
    obj = extract_output_json(raw)
    return obj if isinstance(obj, list) else None


def parse_relevance_result(raw):
    obj = extract_output_json(raw)
    return obj if isinstance(obj, list) else None


# ── Phase 2: Stats ──

def compute_diversity_stats(results):
    scores = [p["diversityScore"] for _, p in results if p and p.get("diversityScore") is not None]
    if not scores:
        return {}
    arr = np.array(scores)
    dist = {s: int((arr == s).sum()) for s in [0, 1, 2]}
    n = len(scores)
    return {
        "total_users": n, "mean_score": round(float(arr.mean()), 3),
        "score_distribution": {
            f"score_{k} ({'High' if k==2 else 'Moderate' if k==1 else 'Low'})":
            f"{v} ({v/n*100:.1f}%)" for k, v in sorted(dist.items())
        },
    }


def compute_quality_stats(results):
    dims = ["journeyValue", "contentCompliance", "tone", "selfCoherence"]
    all_scores = {d: [] for d in dims}
    journey_types = defaultdict(int)
    total_journeys = 0
    explanations = []
    for _, parsed_list in results:
        if not parsed_list:
            continue
        for j in parsed_list:
            if not isinstance(j, dict):
                continue
            total_journeys += 1
            jt = j.get("journeyType", "")
            if jt: journey_types[jt] += 1
            for d in ["journeyValue", "contentCompliance"]:
                v = j.get("journeyAppropriateness", {}).get(d)
                if v is not None: all_scores[d].append(v)
            for d in ["tone", "selfCoherence"]:
                v = j.get("journeyTitleQuality", {}).get(d)
                if v is not None: all_scores[d].append(v)
            expl = j.get("explanation", "")
            if expl:
                explanations.append({"journeyTitle": j.get("journeyTitle", ""), "explanation": expl})
    stats = {"total_journeys": total_journeys}
    for d in dims:
        arr = np.array(all_scores[d]) if all_scores[d] else np.array([])
        if len(arr) == 0:
            stats[d] = {"n": 0}; continue
        dist = {s: int((arr == s).sum()) for s in [0, 1, 2]}
        n = len(arr)
        stats[d] = {"n": n, "mean": round(float(arr.mean()), 3),
                     "good_rate": f"{dist[2]}/{n} ({dist[2]/n*100:.1f}%)",
                     "mixed_rate": f"{dist[1]}/{n} ({dist[1]/n*100:.1f}%)",
                     "fail_rate": f"{dist[0]}/{n} ({dist[0]/n*100:.1f}%)"}
    stats["journey_type_distribution"] = dict(journey_types)
    stats["sample_explanations"] = explanations[:20]
    return stats


def compute_relevance_stats(results):
    scores = []
    explanations = []
    for _, parsed_list in results:
        if not parsed_list:
            continue
        for j in parsed_list:
            if not isinstance(j, dict):
                continue
            s = j.get("shoppingRelevanceScore")
            if s is not None: scores.append(s)
            expl = j.get("explanation", "")
            if expl:
                explanations.append({"journeyTitle": j.get("journeyTitle", ""), "explanation": expl})
    if not scores:
        return {}
    arr = np.array(scores)
    dist = {s: int((arr == s).sum()) for s in [0, 1, 2]}
    n = len(arr)
    return {
        "total_journeys": n, "mean_score": round(float(arr.mean()), 3),
        "score_distribution": {
            "score_2_strong": f"{dist[2]}/{n} ({dist[2]/n*100:.1f}%)",
            "score_1_partial": f"{dist[1]}/{n} ({dist[1]/n*100:.1f}%)",
            "score_0_irrelevant": f"{dist[0]}/{n} ({dist[0]/n*100:.1f}%)",
        },
        "sample_explanations": explanations[:20],
    }


def print_labeling_report(div_stats, qual_stats, rel_stats):
    print(f"\n{'='*80}")
    print(f"  JOURNEY EVALUATION REPORT — 3 Dimensions")
    print(f"{'='*80}")
    print(f"\n{'─'*80}\n  1. DIVERSITY\n{'─'*80}")
    if div_stats:
        print(f"  Users evaluated: {div_stats['total_users']}")
        print(f"  Mean diversity score: {div_stats['mean_score']}")
        for k, v in div_stats["score_distribution"].items():
            print(f"    {k}: {v}")
    else:
        print(f"  (skipped)")

    print(f"\n{'─'*80}\n  2. QUALITY (per-journey)\n{'─'*80}")
    if qual_stats:
        print(f"  Total journeys evaluated: {qual_stats['total_journeys']}")
        for dim in ["journeyValue", "contentCompliance", "tone", "selfCoherence"]:
            d = qual_stats.get(dim, {})
            if d.get("n", 0) == 0: continue
            print(f"\n    {dim}:")
            print(f"      Mean: {d['mean']}")
            print(f"      Good (2): {d['good_rate']}")
            print(f"      Mixed (1): {d['mixed_rate']}")
            print(f"      Fail (0): {d['fail_rate']}")
        if qual_stats.get("journey_type_distribution"):
            print(f"\n    Journey type distribution:")
            for jt, cnt in qual_stats["journey_type_distribution"].items():
                print(f"      {jt}: {cnt}")
    else:
        print(f"  (skipped)")

    print(f"\n{'─'*80}\n  3. RELEVANCE (per-journey)\n{'─'*80}")
    if rel_stats:
        print(f"  Total journeys evaluated: {rel_stats['total_journeys']}")
        print(f"  Mean relevance score: {rel_stats['mean_score']}")
        for k, v in rel_stats["score_distribution"].items():
            print(f"    {k}: {v}")
    else:
        print(f"  (skipped)")
    print(f"\n{'='*80}")

    # ── Summary Table (detailed) ──
    print(f"\n{'='*110}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*110}")

    hdr = f"  {'Dimension':<25s} {'N':>7s} {'Mean':>7s} {'Good(2)':>12s} {'Mixed(1)':>12s} {'Fail(0)':>12s} {'Good%':>8s} {'Mixed%':>8s} {'Fail%':>8s}"
    sep = f"  {'─'*25} {'─'*7} {'─'*7} {'─'*12} {'─'*12} {'─'*12} {'─'*8} {'─'*8} {'─'*8}"
    print(hdr)
    print(sep)

    def _parse_rate(rate_str):
        """Parse '123/456 (78.9%)' -> (123, 456, 78.9)"""
        if not rate_str or "(" not in rate_str:
            return 0, 0, 0.0
        num = int(rate_str.split("/")[0])
        total = int(rate_str.split("/")[1].split()[0])
        pct = float(rate_str.split("(")[1].rstrip("%)"))
        return num, total, pct

    # Diversity
    if div_stats:
        n = div_stats["total_users"]
        dist = div_stats.get("score_distribution", {})
        high_n = low_n = mod_n = 0
        for k, v in dist.items():
            cnt = int(v.split()[0])
            if "High" in k: high_n = cnt
            elif "Moderate" in k: mod_n = cnt
            elif "Low" in k: low_n = cnt
        print(f"  {'Diversity (per-user)':<25s} {n:>7d} {div_stats['mean_score']:>7.3f} "
              f"{high_n:>12d} {mod_n:>12d} {low_n:>12d} "
              f"{high_n/n*100:>7.1f}% {mod_n/n*100:>7.1f}% {low_n/n*100:>7.1f}%")

    # Quality dimensions
    if qual_stats:
        print(sep)
        for dim in ["journeyValue", "contentCompliance", "tone", "selfCoherence"]:
            d = qual_stats.get(dim, {})
            if d.get("n", 0) == 0:
                continue
            n = d["n"]
            g_n, _, g_pct = _parse_rate(d.get("good_rate", ""))
            m_n, _, m_pct = _parse_rate(d.get("mixed_rate", ""))
            f_n, _, f_pct = _parse_rate(d.get("fail_rate", ""))
            print(f"  {dim:<25s} {n:>7d} {d['mean']:>7.3f} "
                  f"{g_n:>12d} {m_n:>12d} {f_n:>12d} "
                  f"{g_pct:>7.1f}% {m_pct:>7.1f}% {f_pct:>7.1f}%")
        # Journey type
        jt_dist = qual_stats.get("journey_type_distribution", {})
        if jt_dist:
            total_jt = sum(jt_dist.values())
            parts = [f"{k}:{v}({v/total_jt*100:.0f}%)" for k, v in jt_dist.items()]
            print(f"  {'  journeyType':<25s} {total_jt:>7d} {'':>7s} {' / '.join(parts):>48s}")

    # Relevance
    if rel_stats:
        print(sep)
        n = rel_stats["total_journeys"]
        dist = rel_stats.get("score_distribution", {})
        s2_n, _, s2_pct = _parse_rate(dist.get("score_2_strong", ""))
        s1_n, _, s1_pct = _parse_rate(dist.get("score_1_partial", ""))
        s0_n, _, s0_pct = _parse_rate(dist.get("score_0_irrelevant", ""))
        print(f"  {'Relevance (per-journey)':<25s} {n:>7d} {rel_stats['mean_score']:>7.3f} "
              f"{s2_n:>12d} {s1_n:>12d} {s0_n:>12d} "
              f"{s2_pct:>7.1f}% {s1_pct:>7.1f}% {s0_pct:>7.1f}%")

    print(f"  {'─'*108}")
    print(f"{'='*110}")


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Combined: vLLM Journey Inference + GPT 3-Dim Labeling")

    # Phase 1: Inference
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--input_tsv", type=str, default=DEFAULT_INPUT_TSV)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--output_suffix", type=str, default=DEFAULT_OUTPUT_SUFFIX)
    p.add_argument("--sample_n", type=int, default=DEFAULT_SAMPLE_N)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max_recent_events", type=int, default=DEFAULT_MAX_RECENT)
    p.add_argument("--num_gpus", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=DEFAULT_GPU_MEM_UTIL)
    p.add_argument("--max_model_len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    p.add_argument("--max_tokens_vllm", type=int, default=DEFAULT_MAX_TOKENS_VLLM)

    # Phase 2: Labeling
    p.add_argument("--token_file", type=str, default=TOKEN_FILE)
    p.add_argument("--labeling_model", type=str, default=DEFAULT_LABELING_MODEL)
    p.add_argument("--labeling_workers", type=int, default=DEFAULT_LABELING_WORKERS)
    p.add_argument("--labeling_max_tokens", type=int, default=DEFAULT_LABELING_MAX_TOKENS)
    p.add_argument("--labeling_chunk_size", type=int, default=DEFAULT_LABELING_CHUNK_SIZE)

    # Control
    p.add_argument("--skip_inference", action="store_true", default=False,
        help="Skip Phase 1 (vLLM inference), use existing --inference_output TSV")
    p.add_argument("--inference_output", type=str, default=None,
        help="Path to existing inference output TSV (used with --skip_inference)")
    p.add_argument("--skip_labeling", action="store_true", default=False,
        help="Skip Phase 2 (GPT labeling)")
    p.add_argument("--skip_diversity", action="store_true", default=False)
    p.add_argument("--skip_quality", action="store_true", default=False)
    p.add_argument("--skip_relevance", action="store_true", default=False)
    p.add_argument("--report_only", action="store_true", default=False,
        help="Skip inference & labeling, re-generate report from existing intermediate JSONL files")
    p.add_argument("--eval_results_dir", type=str, default=None,
        help="Path to dir with intermediate_*.jsonl (used with --report_only)")
    p.add_argument("--debug", action="store_true", default=False)
    p.add_argument("--debug_rows", type=int, default=DEFAULT_DEBUG_ROWS)

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    print("=" * 80)
    print("  Step 5.3 Combined: vLLM Inference + GPT 3-Dimension Labeling")
    print("=" * 80)
    if args.debug:
        print(f"  *** DEBUG MODE: {args.debug_rows} rows ***")

    # =========================================================================
    # REPORT-ONLY MODE: read existing intermediate JSONL and regenerate report
    # =========================================================================
    if args.report_only:
        eval_output_dir = args.eval_results_dir
        if not eval_output_dir:
            eval_output_dir = os.path.join(args.output_dir, f"eval_results_{args.output_suffix}")
        print(f"\n  REPORT-ONLY MODE")
        print(f"  Reading intermediate files from: {eval_output_dir}")

        def _load_intermediate(name, parse_fn):
            fp = os.path.join(eval_output_dir, f"intermediate_{name.lower()}.jsonl")
            if not os.path.isfile(fp):
                print(f"    [{name}] Not found: {fp}")
                return []
            results = []
            ok, fail = 0, 0
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    uid = obj.get("UserId", "")
                    raw_eval = obj.get(f"{name.lower()}_eval")
                    if raw_eval is not None:
                        ok += 1
                    else:
                        fail += 1
                    results.append((uid, raw_eval))
            print(f"    [{name}] Loaded {ok} ok, {fail} fail from {fp}")
            return results

        div_results = _load_intermediate("Diversity", parse_diversity_result)
        qual_results = _load_intermediate("Quality", parse_quality_result)
        rel_results = _load_intermediate("Relevance", parse_relevance_result)

        div_stats = compute_diversity_stats(div_results) if div_results else {}
        qual_stats = compute_quality_stats(qual_results) if qual_results else {}
        rel_stats = compute_relevance_stats(rel_results) if rel_results else {}

        print_labeling_report(div_stats, qual_stats, rel_stats)

        # Save report
        report = {
            "eval_results_dir": eval_output_dir,
            "diversity": div_stats, "quality": qual_stats, "relevance": rel_stats,
        }
        rp = os.path.join(eval_output_dir, "eval_report.json")
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  Report saved to: {rp}")
        print(f"\nDone!")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    inference_tsv = None

    # =========================================================================
    # PHASE 1: vLLM Inference
    # =========================================================================
    if not args.skip_inference:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        num_gpus = args.num_gpus if args.num_gpus else max(gpus, 1)

        print(f"\n{'='*70}")
        print(f"  PHASE 1: vLLM Inference")
        print(f"{'='*70}")
        print(f"  Model:     {args.model_path}")
        print(f"  Input TSV: {args.input_tsv}")
        print(f"  GPUs:      {num_gpus}")

        # Step 1: Read
        print(f"\n  Step 1: Reading input TSV")
        rows, cols = read_test_tsv(args.input_tsv)
        print(f"    {len(rows):,} users, columns: {cols}")

        # Step 2: Sample
        sample_n = min(args.debug_rows if args.debug else args.sample_n, len(rows))
        sampled = sorted(random.sample(rows, sample_n), key=lambda r: r.get("UserId", ""))
        print(f"  Step 2: Sampled {sample_n:,} from {len(rows):,}")

        # Step 3: Process
        print(f"  Step 3: Processing user data")
        data = process_data(sampled)
        ec = np.array([len(ud.get("events_list", [])) for ud in data])
        if len(ec):
            print(f"    Events/user: Mean={ec.mean():.1f}, P50={int(np.percentile(ec, 50))}, Max={ec.max()}")

        # Step 4: Build prompts
        print(f"  Step 4: Building prompts")
        from transformers import AutoTokenizer
        tp = args.model_path
        tok_cfg = os.path.join(tp, "tokenizer_config.json")
        if os.path.isfile(tok_cfg):
            try:
                with open(tok_cfg, "r") as f:
                    tc = json.load(f)
                if tc.get("tokenizer_class") not in ("Qwen2Tokenizer", "PreTrainedTokenizerFast", None):
                    tc["tokenizer_class"] = "Qwen2Tokenizer"
                    with open(tok_cfg, "w") as f:
                        json.dump(tc, f, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                print(f"  [WARN] tokenizer_config.json is corrupted, attempting repair...")
                with open(tok_cfg, "r") as f:
                    raw = f.read()
                # Take only the first valid JSON object
                decoder = json.JSONDecoder()
                tc, _ = decoder.raw_decode(raw)
                tc["tokenizer_class"] = "Qwen2Tokenizer"
                with open(tok_cfg, "w") as f:
                    json.dump(tc, f, indent=2, ensure_ascii=False)
                print(f"  [WARN] tokenizer_config.json repaired.")
        tokenizer = AutoTokenizer.from_pretrained(tp, trust_remote_code=True)
        prompts = build_chat_prompts(data, tokenizer, args.max_recent_events)

        # Step 5: vLLM inference
        print(f"  Step 5: Running vLLM inference")
        raw_outputs = run_vllm_inference(
            prompts, args.model_path, num_gpus,
            args.gpu_memory_utilization, args.max_model_len, args.max_tokens_vllm)

        # Step 6: Parse & save
        print(f"  Step 6: Analyzing & saving")
        results, stats = analyze_outputs(data, raw_outputs)
        print_inference_stats(stats)

        inference_tsv = os.path.join(
            args.output_dir, f"only_journey_output_{args.output_suffix}.tsv")
        out_cols = ["UserId", "UserSignals", "ReadableUserSignals", "UserProfile",
                    "RawOutput", "ParsedJourneys"]
        save_tsv(results, inference_tsv, out_cols)

        # Save summary
        summary = {
            "model_path": args.model_path, "seed": args.seed, "sample_n": sample_n,
            "total_users": stats["total_users"],
            "json_parse_success": stats["json_parse_success"],
            "json_parse_fail": stats["json_parse_fail"],
            "total_journeys": stats["total_journeys"],
            "mean_journeys_per_user": float(np.mean(stats["journeys_per_user"])) if stats["journeys_per_user"] else 0,
        }
        sf = os.path.join(args.output_dir, f"eval_summary_{args.output_suffix}.json")
        with open(sf, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  Summary: {sf}")
    else:
        # Use existing output
        inference_tsv = args.inference_output
        if not inference_tsv:
            inference_tsv = os.path.join(
                args.output_dir, f"only_journey_output_{args.output_suffix}.tsv")
        print(f"\n  PHASE 1 skipped. Using existing: {inference_tsv}")

    # =========================================================================
    # PHASE 2: GPT Labeling
    # =========================================================================
    if not args.skip_labeling:
        from llm_utils import run_llm_parallel_with_checkpoint, cleanup_checkpoint

        print(f"\n{'='*70}")
        print(f"  PHASE 2: GPT 3-Dimension Labeling")
        print(f"{'='*70}")
        print(f"  Input:   {inference_tsv}")
        print(f"  Model:   {args.labeling_model}")
        print(f"  Workers: {args.labeling_workers}")

        max_rows = args.debug_rows if args.debug else 0
        eval_rows = read_eval_data(inference_tsv, max_rows=max_rows)
        if not eval_rows:
            print("  No valid data for labeling. Done.")
            return

        div_template = load_prompt_template("eval_1_Diversity.md")
        qual_template = load_prompt_template("eval_2_Quality.md")
        rel_template = load_prompt_template("eval_3_Relevance.md")

        eval_output_dir = os.path.join(args.output_dir, f"eval_results_{args.output_suffix}")
        os.makedirs(eval_output_dir, exist_ok=True)

        def _run_dim(name, build_fn, parse_fn, template, skip_flag):
            if skip_flag:
                print(f"\n  [{name}] Skipped.")
                return [], {}
            print(f"\n  [{name}] Building prompts ...")
            inputs = build_fn(eval_rows, template)
            print(f"    {len(inputs)} prompts")
            ckpt = os.path.join(eval_output_dir, f"_ckpt_{name.lower()}")
            raw = run_llm_parallel_with_checkpoint(
                inputs=inputs, token_file=args.token_file,
                checkpoint_dir=ckpt, num_workers=args.labeling_workers,
                model=args.labeling_model, temperature=0,
                max_tokens=args.labeling_max_tokens,
                chunk_size=args.labeling_chunk_size)
            results = []
            ok, fail = 0, 0
            for uid, r in raw:
                p = parse_fn(r)
                if p: ok += 1
                else: fail += 1
                results.append((uid, p))
            print(f"    Parsed: {ok} ok, {fail} fail")
            # Save intermediate
            with open(os.path.join(eval_output_dir, f"intermediate_{name.lower()}.jsonl"), "w") as f:
                for uid, p in results:
                    f.write(json.dumps({"UserId": uid, f"{name.lower()}_eval": p}, ensure_ascii=False) + "\n")
            cleanup_checkpoint(ckpt)
            return results, None  # stats computed below

        div_results, _ = _run_dim("Diversity", build_diversity_prompts, parse_diversity_result, div_template, args.skip_diversity)
        qual_results, _ = _run_dim("Quality", build_quality_prompts, parse_quality_result, qual_template, args.skip_quality)
        rel_results, _ = _run_dim("Relevance", build_relevance_prompts, parse_relevance_result, rel_template, args.skip_relevance)

        div_stats = compute_diversity_stats(div_results) if div_results else {}
        qual_stats = compute_quality_stats(qual_results) if qual_results else {}
        rel_stats = compute_relevance_stats(rel_results) if rel_results else {}

        print_labeling_report(div_stats, qual_stats, rel_stats)

        # Save report
        report = {
            "inference_tsv": inference_tsv, "labeling_model": args.labeling_model,
            "total_users": len(eval_rows),
            "diversity": div_stats, "quality": qual_stats, "relevance": rel_stats,
        }
        rp = os.path.join(eval_output_dir, "eval_report.json")
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  Report: {rp}")

        # Per-user combined
        div_map = {uid: p for uid, p in div_results}
        qual_map = {uid: p for uid, p in qual_results}
        rel_map = {uid: p for uid, p in rel_results}
        combined = []
        for r in eval_rows:
            uid = r["UserId"]
            combined.append({
                "UserId": uid, "num_journeys": len(r["ContinuedJourneys"]),
                "diversity": div_map.get(uid), "quality": qual_map.get(uid),
                "relevance": rel_map.get(uid),
            })
        cp = os.path.join(eval_output_dir, "eval_per_user_combined.jsonl")
        with open(cp, "w", encoding="utf-8") as f:
            for item in combined:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Per-user results: {
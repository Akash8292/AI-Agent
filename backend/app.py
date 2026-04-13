from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import json
import os
import requests
from datetime import datetime
import time
import math
import re
import random

app = Flask(__name__, static_folder='../frontend')
CORS(app)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"
DB_PATH      = "../conversations/conversations.db"
DATASETS_PATH = "../datasets"
RAG_EMBED_MODEL = "nomic-embed-text"

AVAILABLE_MODELS = {
    "rag": {
        "name": "RAG Model",
        "description": "Vector similarity search on your dataset.",
        "size": "~274 MB", "speed": "⚡⚡⚡", "is_rag": True
    },
    "llama3.2:1b": {
        "name": "LLaMA 3.2 (1B)",
        "description": "Very fast, 1 billion parameters.",
        "size": "~1.3 GB", "speed": "⚡⚡⚡", "is_rag": False
    },
    "llama3.2:3b": {
        "name": "LLaMA 3.2 (3B)",
        "description": "Balanced speed & smarts.",
        "size": "~2.0 GB", "speed": "⚡⚡", "is_rag": False
    },
    "phi3.5": {
        "name": "Phi 3.5 (3.8B)",
        "description": "Microsoft's efficient model.",
        "size": "~2.2 GB", "speed": "⚡⚡", "is_rag": False
    },
    "qwen2.5:1.5b": {
        "name": "Qwen 2.5 (1.5B)",
        "description": "Alibaba's small but capable model.",
        "size": "~1.0 GB", "speed": "⚡⚡⚡", "is_rag": False
    },
    "gemma2:2b": {
        "name": "Gemma 2 (2B)",
        "description": "Google's compact model.",
        "size": "~1.6 GB", "speed": "⚡⚡", "is_rag": False
    }
}

# ─── IN-MEMORY RAG STORE ──────────────────────────────────────────────────────
rag_store = {}

# ─── DATASET TYPE DETECTION ───────────────────────────────────────────────────

def is_menu_dataset(dataset):
    """Returns True if dataset uses new structured menu format (has 'menu' key)."""
    return "menu" in dataset

# ═══════════════════════════════════════════════════════════════════════════════
#  NEW STRUCTURED MENU ENGINE
#  Handles the new {menu: [{category, items: [{name, price}]}]} format
# ═══════════════════════════════════════════════════════════════════════════════

# ── Category keyword map ──────────────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "Chicken Buckets": [
        "bucket", "buckets", "family", "family pack", "large pack", "party pack",
        "party", "bulk", "big chicken"
    ],
    "Burgers": [
        "burger", "burgers", "sandwich", "sandwiches", "wrap", "zinger",
        "krisper", "double down", "veg burger", "chicken burger"
    ],
    "Chicken": [
        "pc chicken", "piece chicken", "hot & crispy", "hot crispy",
        "classic chicken", "crispy chicken", "fried chicken", "chicken piece",
        "1 pc", "2 pc", "piece"
    ],
    "Snacks": [
        "snack", "snacks", "popcorn chicken", "popcorn", "nugget", "nuggets",
        "wings", "hot wings", "strips", "bites", "finger"
    ],
    "Meals & Combos": [
        "meal", "meals", "combo", "combos", "deal", "deals", "value meal",
        "set meal", "bundle", "package"
    ],
    "Sides": [
        "side", "sides", "fries", "french fries", "coleslaw", "slaw",
        "mashed potato", "mashed", "potato", "veg strip", "veg strips"
    ],
    "Beverages": [
        "drink", "drinks", "beverage", "beverages", "pepsi", "7up", "mirinda",
        "cold drink", "soda", "juice", "water", "cola", "aerated"
    ],
    "Desserts": [
        "dessert", "desserts", "sweet", "sweets", "cake", "lava cake",
        "ice cream", "chocolate lava", "kulfi", "sundae"
    ],
}

# ── Price intent keywords ─────────────────────────────────────────────────────
CHEAP_WORDS = {
    "cheapest", "cheap", "lowest", "budget", "affordable",
    "minimum", "min", "inexpensive", "least expensive", "least costly",
    "low price", "low cost", "economy", "value"
}
COSTLY_WORDS = {
    "costly", "expensive", "most expensive", "highest", "premium",
    "priciest", "luxury", "maximum", "max", "pricey", "most costly",
    "high price", "richest", "lavish"
}
POPULAR_WORDS = {
    "popular", "best", "most popular", "top", "famous", "trending",
    "recommended", "bestseller", "most ordered", "demand", "demanding",
    "hit", "favourite", "favorite", "loved", "must try", "must-try"
}

# ── Broad / full-menu intent ──────────────────────────────────────────────────
FULL_MENU_WORDS = {
    "menu", "full menu", "entire menu", "whole menu", "all menu",
    "everything", "all items", "all food", "what do you have",
    "what do you offer", "what have you got", "whats available",
    "what is available", "show everything", "show all",
    "list everything", "list all", "complete menu",
    "your menu", "kfc menu", "the menu"
}

# Broad "best/popular" with NO category → top picks across all
POPULAR_ALL_WORDS = {
    "most popular item", "most popular items", "popular item", "popular items",
    "popular kfc", "best kfc", "best item", "best items", "top item", "top items",
    "bestselling", "best selling", "most loved", "most ordered item",
    "most ordered items", "must try item", "must try items",
    "what should i order", "what should i try", "recommend something",
    "suggest something", "what to order", "what to eat", "what to try",
    "top picks", "most popular kfc", "most popular"
}


# ── Veg / Non-veg intent ─────────────────────────────────────────────────────
VEG_WORDS = {
    "veg", "vegetarian", "veggie", "veg option", "veg options",
    "veg item", "veg items", "veg meal", "veg food", "plant based",
    "no meat", "without meat", "meatless", "veg friendly",
    "veg only", "veg menu", "veg burger", "veg snack"
}
NONVEG_WORDS = {
    "non veg", "nonveg", "non-veg", "chicken", "meat", "non vegetarian",
    "non-vegetarian", "egg", "fish"
}


def detect_veg_intent(question):
    """
    Returns 'veg', 'nonveg', or None.
    IMPORTANT: check nonveg FIRST — 'non veg' contains 'veg' as substring,
    so nonveg must win when present.
    """
    q = question.lower()
    # Check nonveg first (longer phrases, e.g. "non veg" before "veg")
    for phrase in sorted(NONVEG_WORDS, key=len, reverse=True):
        if phrase in q:
            return "nonveg"
    for phrase in sorted(VEG_WORDS, key=len, reverse=True):
        if phrase in q:
            return "veg"
    return None


def detect_broad_intent(question):
    """
    Returns:
      'full_menu'   - user wants the entire menu
      'popular_all' - user wants top/best across all categories
      None          - no broad intent
    """
    q = question.lower().strip()
    for phrase in sorted(FULL_MENU_WORDS, key=len, reverse=True):
        if phrase in q:
            return "full_menu"
    for phrase in sorted(POPULAR_ALL_WORDS, key=len, reverse=True):
        if phrase in q:
            return "popular_all"
    return None


def detect_price_intent(question):
    q = question.lower()
    for phrase in sorted(CHEAP_WORDS | COSTLY_WORDS | POPULAR_WORDS, key=len, reverse=True):
        if phrase in q:
            if phrase in CHEAP_WORDS:   return "cheap"
            if phrase in COSTLY_WORDS:  return "costly"
            if phrase in POPULAR_WORDS: return "popular"
    return None


def detect_price_range(question):
    """
    Detect price range constraints from natural language.
    Returns a dict with any of: {min, max}  — both in same currency as dataset.

    Examples handled:
      "below 200"          → {max: 200}
      "under 300"          → {max: 300}
      "less than 150"      → {max: 150}
      "above 100"          → {min: 100}
      "more than 500"      → {min: 500}
      "over 400"           → {min: 400}
      "between 100 and 300"→ {min: 100, max: 300}
      "100 to 300"         → {min: 100, max: 300}
      "price 200"          → {max: 200}   ← treated as "under X" when no qualifier
      "under rs 200"       → {max: 200}   ← handles "rs", "inr", "rupees" prefix
      "below rs. 200"      → {max: 200}
    """
    q = question.lower()
    result = {}

    # Strip currency symbols / words so numbers are clean
    q_clean = re.sub(r'(?:rs\.?|inr|rupees?|₹)\s*', '', q)

    # Pattern: "between X and Y"  /  "X to Y"  /  "X - Y"
    between = re.search(
        r'between\s+(\d+)\s+(?:and|to|-)\s+(\d+)'
        r'|(\d+)\s+(?:to|-)\s+(\d+)',
        q_clean
    )
    if between:
        g = between.groups()
        if g[0] and g[1]:
            result['min'] = int(g[0]);  result['max'] = int(g[1])
        elif g[2] and g[3]:
            result['min'] = int(g[2]);  result['max'] = int(g[3])
        return result

    # Pattern: "below / under / less than / cheaper than / at most / max X"
    below = re.search(
        r'(?:below|under|less than|cheaper than|at most|max(?:imum)?|not more than|upto?|up to)\s+(\d+)',
        q_clean
    )
    if below:
        result['max'] = int(below.group(1))

    # Pattern: "above / over / more than / at least / min X"
    above = re.search(
        r'(?:above|over|more than|greater than|at least|min(?:imum)?|not less than)\s+(\d+)',
        q_clean
    )
    if above:
        result['min'] = int(above.group(1))

    return result   # empty dict = no price range constraint


def find_matching_categories(question):
    """Return list of category names whose keywords appear in the question."""
    q = question.lower()
    matched = []
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            matched.append(cat)
    return matched


def get_all_items(dataset):
    """Flatten the menu into a list of {category, name, price, veg} dicts."""
    items = []
    for cat in dataset.get("menu", []):
        for item in cat.get("items", []):
            items.append({
                "category": cat["category"],
                "name":     item["name"],
                "price":    item["price"],
                "veg":      item.get("veg", False)   # default non-veg if not tagged
            })
    return items


def menu_search(dataset, question):
    """
    Core search for structured menu datasets.
    Returns (matched_items, matched_categories, price_intent, broad_intent).

    Priority order:
      1. Broad intent (full menu / popular all) → overrides category
      2. Category match + price intent + veg filter
      3. Item name keyword search + veg filter
      4. Empty → truly out of scope
    """
    all_items    = get_all_items(dataset)
    broad_intent = detect_broad_intent(question)
    price_intent = detect_price_intent(question)
    veg_intent   = detect_veg_intent(question)
    price_range  = detect_price_range(question)   # ← NEW: {min?, max?}

    def apply_veg_filter(items):
        """Filter by veg/nonveg if requested, else return as-is."""
        if veg_intent == "veg":
            return [i for i in items if i.get("veg") is True]
        if veg_intent == "nonveg":
            return [i for i in items if i.get("veg") is False]
        return items

    def apply_price_range_filter(pool):
        """Filter pool to items within the detected price range."""
        if not price_range:
            return pool
        lo = price_range.get("min", 0)
        hi = price_range.get("max", float("inf"))
        return [i for i in pool if lo <= i["price"] <= hi]

    def apply_price_filter(pool):
        """Apply cheap/costly/popular price intent OR price range to a pool."""
        if not pool:
            return pool
        # Price range takes priority over cheap/costly/popular
        if price_range:
            return apply_price_range_filter(pool)
        if price_intent == "cheap":
            min_p = min(i["price"] for i in pool)
            return [i for i in pool if i["price"] == min_p]
        if price_intent == "costly":
            max_p = max(i["price"] for i in pool)
            return [i for i in pool if i["price"] == max_p]
        if price_intent == "popular":
            return sorted(pool, key=lambda x: x["price"], reverse=True)[:3]
        return pool

    # ── 1. BROAD INTENT ───────────────────────────────────────────────────────
    if broad_intent == "full_menu":
        pool   = apply_veg_filter(all_items)
        result = apply_price_filter(pool) if (price_intent or price_range) else pool
        return result, [], price_intent, broad_intent, price_range

    if broad_intent == "popular_all":
        pool   = apply_veg_filter(all_items)
        result = sorted(pool, key=lambda x: x["price"], reverse=True)[:3]
        return result, [], "popular", broad_intent, price_range

    # ── 2. CATEGORY MATCH ─────────────────────────────────────────────────────
    categories = find_matching_categories(question)

    # Special case: price_range with NO specific category → search all items
    if not categories and price_range:
        pool   = apply_veg_filter(all_items)
        result = apply_price_range_filter(pool)
        if result:
            return result, [], price_intent, "price_range_all", price_range

    # Special case: "veg meal/food/option" with NO specific category keyword
    # → search across ALL categories for veg items
    if not categories and veg_intent == "veg":
        pool = apply_veg_filter(all_items)
        result = apply_price_filter(pool)
        if result:
            return result, [], price_intent, "veg_all", price_range

    if categories:
        pool   = [i for i in all_items if i["category"] in categories]
        pool   = apply_veg_filter(pool)
        result = apply_price_filter(pool)
        # If veg filter wiped the pool, result will be empty → handled below
        return result, categories, price_intent, None, price_range

    # ── 3. ITEM NAME KEYWORD SEARCH ───────────────────────────────────────────
    q = question.lower()
    stopwords = {
        "what","are","is","the","a","an","at","kfc","does","do","have","tell",
        "me","about","can","i","get","how","much","any","some","there","for",
        "in","of","please","you","your","show","give","list","all","options",
        "available","and","or","to","with","option","want","see","their","our",
        "veg","vegetarian","veggie","non","meal","food","item","items"
    }
    keywords = [w for w in re.split(r'\W+', q) if w not in stopwords and len(w) > 2]
    pool = []
    for item in all_items:
        if any(kw in item["name"].lower() for kw in keywords):
            pool.append(item)

    pool   = apply_veg_filter(pool)
    result = apply_price_filter(pool)

    if not result:
        return [], [], price_intent, None, price_range  # Truly out of scope

    return result, [], price_intent, None, price_range


# ── Gratitude messages per mode & intent ─────────────────────────────────────
EASY_INTROS = {
    None:      [
        "🍗 Great choice! Here's what we've got for you:",
        "🍗 Absolutely! Here's what KFC has to offer:",
        "🍗 Sure thing! Take a look at our menu below:",
    ],
    "cheap":   [
        "🍗 Smart pick! Here are the most budget-friendly options — great taste, easy on the wallet:",
        "🍗 Looking for value? You're in the right place! Here are our most affordable choices:",
        "🍗 No need to break the bank! Here are the cheapest options we have:",
    ],
    "costly":  [
        "🍗 Treating yourself? Excellent taste! Here are our premium picks:",
        "🍗 Going all out — we love it! Here are the top-shelf options:",
        "🍗 Only the best? Here are our most premium offerings:",
    ],
    "popular": [
        "🍗 You've got great taste! Here are our fan-favourite items people can't stop ordering:",
        "🍗 These are the crowd favourites — the items everyone loves:",
        "🍗 The most popular picks at KFC — tried, loved, and ordered again and again:",
    ],
}

MEDIUM_INTROS = {
    None:      ["🍗 Here's the menu:", "🍗 Available options:"],
    "cheap":   ["🍗 Budget picks:", "🍗 Cheapest options:"],
    "costly":  ["🍗 Premium picks:", "🍗 Most expensive options:"],
    "popular": ["🍗 Top picks:", "🍗 Fan favourites:"],
}

CURRENCY_SYMBOL = {"INR": "₹", "USD": "$", "GBP": "£", "EUR": "€"}


def _make_range_intro(price_range, sym, mode, veg_label=""):
    """Generate a greeting for price-range filtered results."""
    lo = price_range.get("min")
    hi = price_range.get("max")
    veg = f" {veg_label}" if veg_label else ""

    if lo and hi:
        label = f"between {sym}{lo} and {sym}{hi}"
    elif hi:
        label = f"under {sym}{hi}"
    else:
        label = f"above {sym}{lo}"

    if mode == "easy":
        return random.choice([
            f"🍗 Here are all{veg} items priced {label}:",
            f"🍗 Great! Found these{veg} options {label}:",
            f"🍗 Here's what we have{veg} {label}:",
        ])
    else:
        return f"🍗{veg} Items {label}:"


def build_menu_response(items, categories, price_intent, mode, currency="INR",
                         broad_intent=None, price_range=None):
    """
    Format matched menu items into a response based on mode.
    easy   → warm greeting + grouped items with prices
    medium → short intro + grouped items with prices
    hard   → bare list of items with prices, no greeting, no headers

    broad_intent='full_menu' → always show category headers regardless of count
    broad_intent='popular_all' → uses popular greeting
    price_range={min?,max?}  → show price-range intro
    """
    if not items:
        return None

    sym = CURRENCY_SYMBOL.get(currency, "₹")

    # Group by category for display
    grouped = {}
    for item in items:
        grouped.setdefault(item["category"], []).append(item)

    # Always show headers if multiple categories OR full-menu / cross-category request
    show_headers = (len(grouped) > 1) or broad_intent in ("full_menu", "price_range_all", "veg_all")

    if mode == "hard":
        if show_headers:
            lines = []
            for cat, cat_items in grouped.items():
                lines.append(f"**{cat}**")
                for i in cat_items:
                    lines.append(f"• {i['name']} — {sym}{i['price']}")
                lines.append("")
            while lines and not lines[-1]:
                lines.pop()
            return "\n".join(lines)
        else:
            return "\n".join(f"• {i['name']} — {sym}{i['price']}" for i in items)

    # Build core menu block (easy & medium)
    core_lines = []
    for cat, cat_items in grouped.items():
        if show_headers:
            core_lines.append(f"**{cat}**")
        for item in cat_items:
            core_lines.append(f"• {item['name']} — {sym}{item['price']}")
        if show_headers:
            core_lines.append("")
    while core_lines and not core_lines[-1]:
        core_lines.pop()

    core = "\n".join(core_lines)

    # Choose intro based on broad_intent override or price_intent
    effective_intent = "popular" if broad_intent == "popular_all" else price_intent

    # ── Price-range intro takes highest priority ───────────────────────────────
    if price_range:
        # Detect veg label for intro text
        all_veg    = all(i.get("veg") for i in items)
        all_nonveg = all(not i.get("veg") for i in items)
        veg_label  = "veg" if all_veg else ("non-veg" if all_nonveg else "")
        easy_intro   = _make_range_intro(price_range, sym, "easy",   veg_label)
        medium_intro = _make_range_intro(price_range, sym, "medium", veg_label)
        if mode == "easy":
            return f"{easy_intro}\n\n{core}"
        elif mode == "medium":
            return f"{medium_intro}\n\n{core}"
        else:
            return core

    if broad_intent == "full_menu":
        easy_intro = random.choice([
            "🍗 Welcome to KFC! Here's our complete menu with prices:",
            "🍗 Here's everything on the KFC menu — finger lickin' good choices await!",
            "🍗 Our full menu is right here! Take your pick:",
        ])
        medium_intro = random.choice([
            "🍗 Full KFC menu:",
            "🍗 Complete menu:",
            "🍗 KFC menu — all categories:",
        ])
    elif broad_intent == "veg_all":
        easy_intro = random.choice([
            "🥗 Great news! Here are all the vegetarian options at KFC:",
            "🥗 Here are our veg-friendly items across the menu:",
            "🥗 All vegetarian options — perfect for a meat-free meal:",
        ])
        medium_intro = random.choice([
            "🥗 Veg options:",
            "🥗 Vegetarian items:",
            "🥗 All veg items:",
        ])
    else:
        easy_intro   = random.choice(EASY_INTROS.get(effective_intent, EASY_INTROS[None]))
        medium_intro = random.choice(MEDIUM_INTROS.get(effective_intent, MEDIUM_INTROS[None]))

    if mode == "easy":
        return f"{easy_intro}\n\n{core}"
    elif mode == "medium":
        return f"{medium_intro}\n\n{core}"

    return core


# ═══════════════════════════════════════════════════════════════════════════════
#  LEGACY Q&A DATASET ENGINE
#  Handles the old {training_data, qa_pairs} format
# ═══════════════════════════════════════════════════════════════════════════════

def legacy_keyword_search(dataset, user_question):
    """Keyword search for old-format datasets with training_data / qa_pairs."""
    question_lower = user_question.lower()
    stopwords = {
        "what","are","is","the","a","an","at","kfc","does","do","have","tell",
        "me","about","can","i","get","how","much","any","some","there","their",
        "for","in","of","please","you","your","my","its","and","or","to","with"
    }
    keywords = [w for w in question_lower.split() if w not in stopwords and len(w) > 2]
    if not keywords:
        return [], 0

    scored = []
    for item in dataset.get('training_data', []):
        q_lower   = item['instruction'].lower()
        full_text = q_lower + ' ' + item['response'].lower()
        score = sum(3 if kw in q_lower else 1 for kw in keywords if kw in full_text)
        if score > 0:
            scored.append((score, item['instruction'], item['response']))

    for item in dataset.get('qa_pairs', []):
        q_lower   = item['question'].lower()
        full_text = q_lower + ' ' + item['answer'].lower()
        score = sum(3 if kw in q_lower else 1 for kw in keywords if kw in full_text)
        if score > 0:
            scored.append((score, item['question'], item['answer']))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][0] if scored else 0
    return scored[:3], best


def _medium_mode_legacy(text):
    """Medium mode for legacy Q&A text answers."""
    lines = text.split('\n')
    result_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'\(([^)]{40,})\)', '', line)
        for pat in [
            r'\s*[-–]\s*great for.*$', r'\s*,\s*great for.*$',
            r'\s*\.\s*Note:.*$', r'\s*\.\s*Please.*ask.*$',
            r'\s*Always check.*$', r'\s*People with.*should.*$',
        ]:
            line = re.sub(pat, '', line, flags=re.IGNORECASE)
        if len(line) > 120 and ',' in line[:120]:
            cut = line[:120].rfind(',')
            line = line[:cut] + '.'
        line = line.strip().strip(',').strip()
        if line:
            result_lines.append(line)
    return '\n'.join(result_lines[:8])


def _hard_mode_legacy(text):
    """Hard mode for legacy Q&A text answers — item names only."""
    def extract_name(raw):
        raw  = raw.strip()
        name = re.split(r'\s[-–]\s', raw, maxsplit=1)[0]
        name = re.sub(r'\s*\([^)]*\)', '', name)
        name = re.sub(r'\s*(around\s*)?\$[\d\.\-]+(?:-\$[\d\.\-]+)?', '', name)
        name = re.sub(r',\s*serves\b.*$', '', name, flags=re.IGNORECASE)
        name = re.sub(r',\s*\d+\s*(pieces?|items?|serves?)\b.*$', '', name, flags=re.IGNORECASE)
        return name.strip().strip('.,').strip()

    chunks = re.split(r'\d+[.)]\s+', text)
    chunks = chunks[1:]
    items  = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        name = extract_name(chunk)
        if name and len(name) > 2:
            items.append(name)

    if not items:
        for line in text.split('\n'):
            line = line.strip()
            if not line or line[0].islower():
                continue
            name = extract_name(line)
            if name and len(name) > 3:
                items.append(name)

    seen, unique = set(), []
    for item in items:
        if item.lower() not in seen:
            seen.add(item.lower())
            unique.append(item)

    if not unique:
        return re.split(r'[.!?]', text)[0].strip() or text[:120]

    return '\n'.join(f"{i+1}. {item}" for i, item in enumerate(unique[:8]))


def apply_legacy_mode(answer, mode):
    if not mode or mode == "easy":
        return answer
    if mode == "medium":
        return _medium_mode_legacy(answer)
    if mode == "hard":
        return _hard_mode_legacy(answer)
    return answer


# ─── DATABASE ─────────────────────────────────────────────────────────────────
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            session_id TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ─── DATASET LOADING ──────────────────────────────────────────────────────────
def load_dataset(name):
    path = os.path.join(DATASETS_PATH, f"{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ─── RAG (for legacy datasets) ────────────────────────────────────────────────
def get_all_qa_pairs(dataset_name):
    dataset = load_dataset(dataset_name)
    if not dataset:
        return []
    pairs = []
    for item in dataset.get('training_data', []):
        pairs.append({"question": item['instruction'], "answer": item['response']})
    for item in dataset.get('qa_pairs', []):
        pairs.append({"question": item['question'], "answer": item['answer']})
    return pairs


def get_embedding(text):
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": RAG_EMBED_MODEL, "prompt": text},
            timeout=30
        )
        if r.status_code == 200:
            return r.json().get("embedding", None)
    except Exception as e:
        print(f"Embedding error: {e}")
    return None


def cosine_similarity(vec_a, vec_b):
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def build_rag_index(dataset_name):
    if dataset_name in rag_store:
        return rag_store[dataset_name]
    print(f"[RAG] Building index for: {dataset_name}")
    pairs   = get_all_qa_pairs(dataset_name)
    indexed = []
    for pair in pairs:
        emb = get_embedding(pair['question'] + " " + pair['answer'])
        indexed.append({
            "question": pair['question'],
            "answer":   pair['answer'],
            "embedding": emb
        })
    rag_store[dataset_name] = indexed
    print(f"[RAG] Indexed {len(indexed)} entries")
    return indexed


def rag_search(dataset_name, user_question, top_k=3):
    query_emb = get_embedding(user_question)
    if not query_emb:
        return legacy_keyword_search(load_dataset(dataset_name), user_question)
    index = build_rag_index(dataset_name)
    if not index:
        return [], 0.0
    scored = [
        (cosine_similarity(query_emb, e['embedding']), e['question'], e['answer'])
        for e in index if e['embedding']
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    return top, top[0][0] if top else 0.0


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def format_answer(text):
    lines = text.strip().split('\n')
    return '\n'.join(line.strip() for line in lines if line.strip())


def get_mode_label(mode):
    return {"easy": "📖 Easy", "medium": "📋 Medium", "hard": "⚡ Hard"}.get(mode, "📖 Easy")


# ─── CONVERSATIONAL INTENT ────────────────────────────────────────────────────
GREETINGS = [
    'hi','hello','hey','hii','helo','howdy','sup','whats up',
    'good morning','good afternoon','good evening','good night',
    'greetings','salut','namaste','yo'
]
THANK_YOU = [
    'thank','thanks','thankyou','thank you','thx','ty',
    'appreciated','great help','helpful','good job','well done',
    'nice','awesome','perfect','great','wonderful','excellent',
    'shukriya','dhanyawad','bahut acha','very good','good work'
]
EXIT = [
    'bye','goodbye','exit','quit','close','stop','end',
    'see you','see ya','later','take care','cya','ok bye',
    'okay bye',"that's all",'thats all','done','finish',
    'no more','nothing else','i am done',"i'm done",'all good'
]

GREETING_RESPONSES = [
    "🍗 **Welcome to KFC!**\n\nHello! I'm your KFC Assistant. Ask me about:\n- 🍗 Chicken, Burgers & Snacks\n- 🪣 Bucket & Combo Meals\n- 🥤 Beverages & Desserts\n- 💰 Prices & best value picks\n\nWhat would you like to order today?",
    "🍗 **Hey there!** Great to see you!\n\nI'm here to help you explore the KFC menu — from crispy chicken to budget combos. What are you in the mood for? 😊",
    "🍗 **Hi! KFC Assistant here!**\n\nFinger Lickin' ready to help! Ask me anything about our menu, prices, or best deals. What's on your mind? 🍗",
]
THANK_YOU_RESPONSES = [
    "🍗 **You're welcome!**\n\nHappy to help! Ask me anything else about the KFC menu anytime. Enjoy your meal! 😊",
    "🍗 **Glad I could help!**\n\nFinger Lickin' Good service is what we aim for! Feel free to ask more. 😄",
    "🍗 **Anytime!**\n\nThat's what I'm here for. Come back whenever you want to explore the menu. Have a great day! 🍗",
]
EXIT_RESPONSES = [
    "🍗 **Goodbye!**\n\nThank you for visiting KFC. Hope to see you again soon. Enjoy your meal! 🍗😊",
    "🍗 **Take care!**\n\nIt was a pleasure helping you. Come back anytime! Goodbye! 👋🍗",
    "🍗 **See you soon!**\n\nDon't forget — KFC is always here for you. Have a wonderful day! 🍗✨",
]


def detect_conversational_intent(message):
    msg = message.lower().strip().rstrip('!.,?')
    for phrase in GREETINGS:
        if msg == phrase or msg.startswith(phrase + ' ') or msg.endswith(' ' + phrase):
            return 'greeting', random.choice(GREETING_RESPONSES)
    for phrase in THANK_YOU:
        if phrase in msg:
            return 'thankyou', random.choice(THANK_YOU_RESPONSES)
    for phrase in EXIT:
        if msg == phrase or phrase in msg:
            return 'exit', random.choice(EXIT_RESPONSES)
    return None, None


# ─── API ROUTES ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')


@app.route('/api/models', methods=['GET'])
def get_models():
    models_status = {}
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        installed_list = [m['name'] for m in r.json().get('models', [])]
        installed = set(installed_list)
        rag_ready = RAG_EMBED_MODEL in installed or any(
            m.startswith(RAG_EMBED_MODEL) for m in installed
        )
    except:
        installed, rag_ready = set(), False

    for model_id, info in AVAILABLE_MODELS.items():
        if model_id == "rag":
            models_status[model_id] = {**info, "installed": rag_ready}
        else:
            models_status[model_id] = {**info, "installed": model_id in installed}
    return jsonify(models_status)


@app.route('/api/rag/status', methods=['GET'])
def rag_status():
    dataset_name = request.args.get('dataset')
    if not dataset_name:
        return jsonify({"indexed": False, "count": 0})
    indexed = rag_store.get(dataset_name, [])
    return jsonify({"indexed": len(indexed) > 0, "count": len(indexed)})


@app.route('/api/rag/index', methods=['POST'])
def rag_index():
    data = request.json
    dataset_name = data.get('dataset')
    if not dataset_name:
        return jsonify({"error": "No dataset specified"}), 400
    if dataset_name in rag_store:
        del rag_store[dataset_name]
    try:
        index = build_rag_index(dataset_name)
        return jsonify({"status": "ok", "indexed": len(index)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    datasets = []
    for f in os.listdir(DATASETS_PATH):
        if not f.endswith('.json'):
            continue
        data = load_dataset(f.replace('.json', ''))
        if not data:
            continue
        if is_menu_dataset(data):
            total = sum(len(cat.get('items', [])) for cat in data.get('menu', []))
            datasets.append({
                "id": f.replace('.json', ''),
                "name": data.get('name', f),
                "description": data.get('description', f"Structured menu — {total} items"),
                "examples": total,
                "type": "menu"
            })
        else:
            datasets.append({
                "id": f.replace('.json', ''),
                "name": data.get('name', f),
                "description": data.get('description', ''),
                "examples": len(data.get('training_data', [])) + len(data.get('qa_pairs', [])),
                "type": "qa"
            })
    return jsonify(datasets)


@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    model_id = request.args.get('model_id')
    db = get_db()
    if model_id:
        sessions = db.execute(
            "SELECT * FROM sessions WHERE model_id = ? ORDER BY updated_at DESC", (model_id,)
        ).fetchall()
    else:
        sessions = db.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
    db.close()
    return jsonify([dict(s) for s in sessions])


@app.route('/api/sessions', methods=['POST'])
def create_session():
    data = request.json
    session_id = f"session_{int(time.time() * 1000)}"
    now = datetime.now().isoformat()
    db = get_db()
    db.execute(
        "INSERT INTO sessions (id, model_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
        (session_id, data.get('model_id', 'llama3.2:1b'), data.get('title', 'New Chat'), now, now)
    )
    db.commit()
    db.close()
    return jsonify({"session_id": session_id})


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    db = get_db()
    db.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    db.commit()
    db.close()
    return jsonify({"status": "deleted"})


@app.route('/api/conversations/<session_id>', methods=['GET'])
def get_conversation(session_id):
    db = get_db()
    messages = db.execute(
        "SELECT role, content, timestamp FROM conversations WHERE session_id = ? ORDER BY id",
        (session_id,)
    ).fetchall()
    db.close()
    return jsonify([dict(m) for m in messages])


# ─── MAIN CHAT ROUTE ──────────────────────────────────────────────────────────
@app.route('/api/chat', methods=['POST'])
def chat():
    data          = request.json
    user_message  = data.get('message', '').strip()
    model_id      = data.get('model_id', 'llama3.2:1b')
    session_id    = data.get('session_id')
    dataset_name  = data.get('dataset', None)
    response_mode = data.get('response_mode', 'easy')

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    db = get_db()
    history = db.execute(
        "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY id",
        (session_id,)
    ).fetchall() if session_id else []

    assistant_message = None
    mode_label        = get_mode_label(response_mode)

    # ── 1. CONVERSATIONAL INTENT (greetings / thanks / exit) ─────────────────
    intent, intent_response = detect_conversational_intent(user_message)
    if intent:
        assistant_message = intent_response

    # ── 2. DATASET MODE ───────────────────────────────────────────────────────
    elif dataset_name:
        dataset = load_dataset(dataset_name)

        if dataset and is_menu_dataset(dataset):
            # ── NEW structured menu dataset ───────────────────────────────────
            currency = dataset.get("currency", "INR")
            items, categories, price_intent, broad_intent, price_range = menu_search(dataset, user_message)

            if items:
                body = build_menu_response(
                    items, categories, price_intent, response_mode, currency,
                    broad_intent=broad_intent,
                    price_range=price_range
                )
                prefix = f"🍗 **KFC Assistant** *({mode_label})*\n\n"
                assistant_message = prefix + body
            else:
                assistant_message = (
                    "🍗 **KFC Assistant:**\n\n"
                    "Sorry, I couldn't find that in our menu! 😊\n\n"
                    "Try asking about:\n"
                    "• **Buckets** — family packs & large portions\n"
                    "• **Burgers** — Zinger, Krisper, Double Down\n"
                    "• **Snacks** — Popcorn Chicken, Nuggets, Hot Wings\n"
                    "• **Sides** — Fries, Coleslaw, Mashed Potato\n"
                    "• **Beverages** — Pepsi, 7UP, Mirinda\n"
                    "• **Desserts** — Chocolate Lava Cake, Ice Cream\n"
                    "• **Cheapest / Most expensive / Popular** picks in any category!"
                )

        elif dataset and not is_menu_dataset(dataset):
            # ── Legacy Q&A dataset ────────────────────────────────────────────
            if model_id == "rag":
                results, best_score = rag_search(dataset_name, user_message)
                if best_score >= 0.5 and results:
                    _, _, matched_a = results[0]
                    processed = apply_legacy_mode(matched_a, response_mode)
                    prefix = f"🔍 **RAG Assistant** *({mode_label} · {best_score:.0%})*\n\n"
                    assistant_message = prefix + format_answer(processed)
                else:
                    assistant_message = (
                        f"🔍 **RAG Assistant:**\n\n"
                        f"No good match found *(best: {best_score:.0%})*. "
                        "Please try a different question about the KFC menu! 🍗"
                    )
            else:
                results, best_score = legacy_keyword_search(dataset, user_message)
                if best_score >= 2 and results:
                    _, _, matched_a = results[0]
                    processed = apply_legacy_mode(matched_a, response_mode)
                    prefix = f"🍗 **KFC Assistant** *({mode_label})*\n\n"
                    assistant_message = prefix + format_answer(processed)
                else:
                    assistant_message = (
                        "🍗 **KFC Assistant:**\n\n"
                        "Sorry! I only have information about KFC's menu, prices, and products. "
                        "Please ask me something about KFC! 😊"
                    )
        else:
            assistant_message = "⚠️ Dataset could not be loaded. Please check the file."

    # ── 3. GENERAL LLM MODE (no dataset selected) ─────────────────────────────
    if assistant_message is None:
        mode_instructions = {
            "easy":   "Give a complete, friendly, detailed answer.",
            "medium": "Give a moderately detailed answer with key facts only. Skip lengthy backstory.",
            "hard":   "Give an extremely short answer — list names or items only, no descriptions."
        }
        system_msg = f"You are a helpful AI assistant. {mode_instructions.get(response_mode, '')}"
        messages = [{"role": "system", "content": system_msg}]
        for msg in history:
            messages.append({"role": msg['role'], "content": msg['content']})
        messages.append({"role": "user", "content": user_message})

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": model_id, "messages": messages, "stream": False,
                      "options": {"temperature": 0.7, "num_ctx": 4096}},
                timeout=120
            )
            if response.status_code != 200:
                return jsonify({"error": f"Ollama error: {response.text}"}), 500
            assistant_message = response.json()['message']['content']
        except requests.exceptions.ConnectionError:
            return jsonify({"error": "Cannot connect to Ollama. Please run: ollama serve"}), 503
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── SAVE TO DB ────────────────────────────────────────────────────────────
    now = datetime.now().isoformat()
    if session_id:
        db.execute(
            "INSERT INTO conversations (model_id, role, content, timestamp, session_id) VALUES (?,?,?,?,?)",
            (model_id, "user", user_message, now, session_id)
        )
        db.execute(
            "INSERT INTO conversations (model_id, role, content, timestamp, session_id) VALUES (?,?,?,?,?)",
            (model_id, "assistant", assistant_message, now, session_id)
        )
        if len(history) == 0:
            title = user_message[:50] + ("..." if len(user_message) > 50 else "")
            db.execute("UPDATE sessions SET title=?, updated_at=? WHERE id=?", (title, now, session_id))
        else:
            db.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id))
        db.commit()

    db.close()
    return jsonify({
        "response":      assistant_message,
        "model":         model_id,
        "session_id":    session_id,
        "response_mode": response_mode
    })


@app.route('/api/ollama/status', methods=['GET'])
def ollama_status():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        models = r.json().get('models', [])
        return jsonify({"running": True, "installed_models": [m['name'] for m in models]})
    except:
        return jsonify({"running": False, "installed_models": []})


# ─── INIT ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    print("\n" + "="*50)
    print("🤖 Local AI Chat Studio — Backend Running!")
    print("="*50)
    print("📍 URL:    http://localhost:5000")
    print("📦 Ollama: ollama serve")
    print("🔍 RAG:    ollama pull nomic-embed-text")
    print("="*50 + "\n")
    app.run(debug=True, port=5000, host='0.0.0.0')
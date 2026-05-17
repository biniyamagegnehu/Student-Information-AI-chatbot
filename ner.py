# ner.py
"""
University Student Information Chatbot - Named Entity Recognition (NER) System
Phase 7: Rule-Based, Typo-Tolerant Named Entity Recognition (NER) Engine

This module is responsible for:
1. Normalizing raw query text by removing punctuation and converting to lowercase.
2. Detecting predefined university entities: DEPARTMENTS, BUILDINGS, OFFICES, SERVICES, DATES, and CAMPUS LOCATIONS.
3. Matching entities using typo-tolerant fuzzy matching (via difflib.SequenceMatcher).
4. Supporting common acronyms and abbreviations (e.g., 'se' for 'software engineering', 'cs' for 'computer science').
5. Resolving overlapping matches using a greedy, length-first interval packing algorithm.
"""

import re
import difflib

# --- ENTITY DICTIONARY & ALIAS TEMPLATES ---
# Maps standard values to their common user patterns, acronyms, and common typos.
ENTITY_TEMPLATES = {
    "DEPARTMENT": {
        "computer science": ["computer science", "cs", "comp sci", "computer science department"],
        "software engineering": ["software engineering", "se", "soft eng", "software engineering department"],
        "information technology": ["information technology", "it", "info tech", "information technology department"],
        "business administration": ["business administration", "business", "bus admin", "business admin", "business administration department"],
        "accounting": ["accounting", "accounting department", "acct"],
        "civil engineering": ["civil engineering", "civil", "civil engineering department", "civ eng"]
    },
    "BUILDING": {
        "block a": ["block a", "building a", "hall a"],
        "block b": ["block b", "building b", "hall b"],
        "block c": ["block c", "building c", "hall c"],
        "library": ["library", "central library", "libary", "central plaza building"],
        "cafeteria": ["cafeteria", "cafe", "canteen", "dining hall"],
        "main hall": ["main hall", "auditorium", "assembly hall"]
    },
    "OFFICE": {
        "registrar office": ["registrar office", "registrar", "registrars office", "registration office"],
        "finance office": ["finance office", "finance", "finances office", "fees office", "cashier office", "accounts office"],
        "student affairs": ["student affairs", "student affairs office", "student affairs desk", "affairs office"],
        "scholarship office": ["scholarship office", "scholarship desk", "scholarships office", "financial aid office"]
    },
    "SERVICE": {
        "registration": ["registration", "register", "enrollment", "enroll"],
        "exam": ["exam", "exams", "examination", "examinations", "test", "tests"],
        "fees": ["fees", "fee", "tuition", "tuition fees", "payment", "payments"],
        "scholarship": ["scholarship", "scholarships", "financial aid"],
        "student services": ["student services", "student service", "counseling", "career helpdesk"]
    },
    "DATE": {
        "today": ["today"],
        "tomorrow": ["tomorrow"],
        "next week": ["next week"],
        "this week": ["this week"],
        "monday": ["monday", "mon"],
        "friday": ["friday", "fri"]
    },
    "LOCATION": {
        "campus": ["campus", "main campus", "north campus", "south campus", "central plaza", "auditorium"]
    }
}


def normalize_text(text: str) -> str:
    """
    Normalizes query text by:
    1. Converting to lowercase.
    2. Stripping punctuation (except alphanumeric and whitespace).
    3. Collapsing multiple spaces.
    """
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Ignore punctuation: keep only letters, numbers, and spaces
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse multiple whitespaces
    return " ".join(text.split())


def get_similarity(chunk: str, target: str) -> float:
    """
    Calculates spelling similarity between two strings.
    For short words (<= 3 chars), requires exact matching to avoid false positives.
    For longer words, uses SequenceMatcher ratio.
    """
    if not chunk or not target:
        return 0.0
    
    # Exact match for very short tokens/acronyms to prevent false positives (like 'is' matching 'it')
    if len(chunk) <= 3 or len(target) <= 3:
        return 1.0 if chunk == target else 0.0
        
    return difflib.SequenceMatcher(None, chunk, target).ratio()


def extract_entities(text: str) -> list:
    """
    Scans the query text to extract university-specific named entities.
    Supports multi-word entities, typo tolerance, alias expansion, and resolves overlaps.
    
    Returns:
        list of tuples: [("ENTITY_TYPE", "standard_value")]
    """
    normalized = normalize_text(text)
    if not normalized:
        return []

    words = normalized.split()
    num_words = len(words)
    candidates = []

    # Iterate over each category, standard value, and their defined aliases
    for category, standard_map in ENTITY_TEMPLATES.items():
        for standard_val, aliases in standard_map.items():
            for alias in aliases:
                alias_clean = normalize_text(alias)
                alias_words = alias_clean.split()
                n = len(alias_words)
                if n == 0:
                    continue

                # Sliding window search over query text
                for start_idx in range(num_words - n + 1):
                    end_idx = start_idx + n - 1
                    chunk_words = words[start_idx:end_idx + 1]
                    chunk = " ".join(chunk_words)

                    # For multi-word aliases, check the full phrase similarity
                    similarity = get_similarity(chunk, alias_clean)

                    # Also check word-by-word similarity for typo tolerance within phrases
                    if similarity < 0.8 and n > 1:
                        # Compute average word-by-word similarity
                        word_sims = []
                        for cw, aw in zip(chunk_words, alias_words):
                            word_sims.append(get_similarity(cw, aw))
                        avg_sim = sum(word_sims) / len(word_sims)
                        if avg_sim >= 0.8:
                            similarity = avg_sim

                    if similarity >= 0.8:
                        candidates.append({
                            "category": category,
                            "value": standard_val,
                            "start": start_idx,
                            "end": end_idx,
                            "similarity": similarity,
                            "length": n
                        })

    # --- OVERLAP RESOLUTION (Greedy Interval Packing) ---
    # Sort candidates by:
    # 1. Number of words spanned (longer phrases first)
    # 2. Similarity score (higher similarity first)
    candidates.sort(key=lambda x: (-x["length"], -x["similarity"]))

    accepted_entities = []
    used_indices = set()

    for cand in candidates:
        cand_indices = set(range(cand["start"], cand["end"] + 1))
        # If this candidate does not overlap with already accepted ones
        if not (cand_indices & used_indices):
            accepted_entities.append((cand["category"], cand["value"]))
            used_indices.update(cand_indices)

    # Maintain original order of appearance in text
    # To do that, we can reconstruct accepted candidates ordered by their starting index
    ordered_accepted = []
    # Let's keep them unique and sorted by start index
    seen = set()
    for cand in sorted(candidates, key=lambda x: x["start"]):
        cand_indices = set(range(cand["start"], cand["end"] + 1))
        # Only keep if they were accepted (i.e. all indices are in used_indices)
        if cand_indices.issubset(used_indices):
            item = (cand["category"], cand["value"])
            if item not in seen:
                ordered_accepted.append(item)
                seen.add(item)

    return ordered_accepted

# utils.py
import json
import os
import re
import config

# --- DYNAMIC KNOWLEDGE BASE LOADER ---
_KB_DATA = {}

def load_knowledge_base(filepath: str = config.KNOWLEDGE_BASE_PATH):
    """
    Loads structural university facts dynamically from knowledge_base.json.
    """
    global _KB_DATA
    if not os.path.exists(filepath):
        _KB_DATA = {}
        return
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            _KB_DATA = json.load(f)
    except Exception as e:
        print(f"[Warning] Failed to load knowledge base: {e}")

# Initial load
load_knowledge_base()

def get_kb_data() -> dict:
    """
    Returns the loaded knowledge base global dictionary.
    """
    return _KB_DATA

# --- ENTITY EXTRACTION SYSTEM ---
ENTITY_KEYWORDS = {
    "department": {
        "computer science": ["computer science", "cs"],
        "software engineering": ["software engineering", "se"],
        "mechanical engineering": ["mechanical engineering", "mech"],
        "business administration": ["business administration", "business", "bus"]
    },
    "office": {
        "registrar": ["registrar", "registrar office", "registration office"],
        "finance": ["finance", "finance office", "fees office", "accounts", "billing"],
        "student affairs": ["student affairs", "affairs office", "activities office"],
        "it helpdesk": ["it helpdesk", "helpdesk", "tech support", "it support"]
    },
    "scholarship": {
        "merit scholarship": ["merit", "merit scholarship", "academic scholarship"],
        "need scholarship": ["need", "need scholarship", "financial aid", "need-based"],
        "sports scholarship": ["sports", "sports scholarship", "varsity scholarship", "athletic"]
    },
    "student_services": {
        "counseling": ["counseling", "mental health", "counselor", "therapy"],
        "career helpdesk": ["career", "career helpdesk", "placement helpdesk", "cv", "resume", "internship"],
        "central library": ["library", "central library", "books", "reference library"]
    },
    "semester": {
        "fall_semester": ["fall", "autumn", "fall semester", "term 1"],
        "spring_semester": ["spring", "spring semester", "term 2"]
    }
}

def extract_entities(text: str) -> dict:
    """
    Scans query text to extract whitelisted university entities.
    Returns:
    A dictionary containing found entities, e.g., {'department': 'computer science'}
    """
    extracted = {}
    normalized_text = text.lower()

    for entity_type, mapping in ENTITY_KEYWORDS.items():
        for standard_value, keywords in mapping.items():
            for kw in keywords:
                # Use regex word boundaries for precise keyword matching
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, normalized_text):
                    extracted[entity_type] = standard_value
                    break # Extract first matching value for this type
                    
    return extracted


# --- CONTEXT & AMBIGUOUS FOLLOW-UP DETECTION ---
AMBIGUOUS_PATTERNS = [
    r"^\s*when\s+(is|are|do|does)\s+(it|they|they\s+start|it\s+start)\b",
    r"^\s*where\s+(is|are)\s+(it|they)\b",
    r"^\s*how\s+much\s+(is|are|does)\s+(it|they|it\s+cost)\b",
    r"^\s*can\s+i\s+(apply|register|enroll)\b",
    r"^\s*tell\s+me\s+(more|further|about\s+it)\b",
    r"^\s*explain\s+(more|further|details)\b",
    r"^\s*what\s+time\s+(is|are)\s+(it|they)\b",
    r"^\s*how\s+do\s+i\s+(do\s+it|apply)\b",
    r"^\s*is\s+it\s+(open|available|still\s+open)\b",
    r"^\s*what\s+about\s+(it|they)\b"
]

def is_ambiguous_query(text: str) -> bool:
    """
    Returns True if the user query is a short, dependent follow-up statement.
    """
    clean_q = text.lower().strip()
    
    # 1. Match regex patterns
    for pattern in AMBIGUOUS_PATTERNS:
        if re.search(pattern, clean_q):
            return True
            
    # Phase 2 follow-ups: e.g. "what about software engineering?", "how about counseling?"
    if re.search(r"^\s*(what|how)\s+about\s+", clean_q):
        return True
        
    if re.search(r"^\s*tell\s+me\s+about\s+", clean_q) and len(clean_q.split()) <= 4:
        return True

    # If the user just types an entity alone (e.g. "software engineering" or "registrar office") as a follow-up
    words = clean_q.split()
    if len(words) <= 3:
        ents = extract_entities(clean_q)
        if ents:
            return True
            
    # 2. Check if it's extremely short (1-2 words) and matches generic pronouns/keywords
    if len(words) <= 2:
        generic_words = {"when", "where", "why", "cost", "price", "apply", "register", "more", "explain", "it", "they"}
        if any(w in generic_words for w in words):
            return True
            
    return False

def resolve_context(query: str, last_state: dict) -> (str, str, dict):
    """
    Infers intent and topic for ambiguous follow-up queries based on session memory.
    Returns:
    - inferred_intent (str)
    - inferred_topic (str)
    - resolved_entities (dict)
    """
    inferred_intent = "fallback"
    inferred_topic = last_state.get("last_topic")
    resolved_entities = last_state.get("last_entities", {})
    
    query_clean = query.lower().strip()
    last_intent = last_state.get("last_intent")
    
    # If no memory exists, we cannot resolve context -> safe fallback
    if not last_intent or last_intent == "fallback":
        return "fallback", "missing_context", {}

    # Check question dimensions (time, spatial, cost, application)
    is_time = any(w in query_clean for w in ["when", "time", "date"])
    is_spatial = any(w in query_clean for w in ["where", "location", "address", "block", "room"])
    is_cost = any(w in query_clean for w in ["how much", "cost", "price", "fee", "pay"])
    is_apply = any(w in query_clean for w in ["apply", "register", "enroll", "open", "deadline"])
    is_more = any(w in query_clean for w in ["more", "explain", "further", "about"])

    # 1. TIME INQUIRIES ("when is it?")
    if is_time:
        if last_intent in ["exams", "registration", "academic_calendar", "scholarship"]:
            inferred_intent = last_intent
        else:
            inferred_intent = "academic_calendar" # General default for dates
            
    # 2. SPATIAL INQUIRIES ("where is it?")
    elif is_spatial:
        if last_intent in ["locations", "student_services", "contacts"]:
            inferred_intent = last_intent
        else:
            inferred_intent = "locations" # Redirect to locations

    # 3. COST INQUIRIES ("how much is it?")
    elif is_cost:
        inferred_intent = "fees"

    # 4. APPLICATION / DEADLINE INQUIRIES ("can I apply?")
    elif is_apply:
        if last_intent in ["registration", "scholarship"]:
            inferred_intent = last_intent
        else:
            inferred_intent = "registration"

    # 5. EXPANSIONS ("tell me more")
    elif is_more:
        inferred_intent = last_intent
        
    else:
        # Fallback to last active topic/intent
        inferred_intent = last_intent

    return inferred_intent, inferred_topic, resolved_entities

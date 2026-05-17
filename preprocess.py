# preprocess.py
"""
University Student Information Chatbot - NLP Preprocessing Module
Phase 3: Production-Grade Preprocessing Pipeline

This module is responsible for cleaning, normalizing, and tokenizing raw user 
queries before feature extraction (TF-IDF) and machine learning inference.
It provides advanced typo correction, slang normalization, repeated letter collapse,
and stopword preservation for critical educational keywords.
"""
import config
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- NLTK SAFE DOWNLOAD SYSTEM ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Initialize NLP Utilities
lemmatizer = WordNetLemmatizer()

# --- PRESERVED STOPWORDS ---
# We keep question markers (how, when, where) and core educational terms.
# This prevents over-cleaning short conversational inputs.
PRESERVED_KEYWORDS = {
    "where", "when", "how", "what", "which", "who", "why",
    "fees", "fee", "exam", "exams", "registration", "register",
    "courses", "course", "location", "locations", "contact", "contacts",
    "services", "service", "scholarship", "scholarships", "schedule", "schedules"
}
try:
    BASE_STOPWORDS = set(stopwords.words('english'))
except Exception:
    BASE_STOPWORDS = set()

PRESERVED_STOPWORDS = BASE_STOPWORDS - PRESERVED_KEYWORDS

# --- STEP 3: TYPO MAP (50+ Mappings) ---
TYPO_MAP = {
    "regstration": "registration",
    "registraion": "registration",
    "regestration": "registration",
    "registation": "registration",
    "scheduel": "schedule",
    "scheduelle": "schedule",
    "schedul": "schedule",
    "examn": "exam",
    "exm": "exam",
    "exms": "exam",
    "examnation": "examination",
    "cources": "courses",
    "cource": "course",
    "scholrship": "scholarship",
    "scholership": "scholarship",
    "calender": "calendar",
    "acadmic": "academic",
    "acadmics": "academic",
    "libary": "library",
    "librery": "library",
    "locaton": "location",
    "locatn": "location",
    "locationss": "location",
    "contat": "contact",
    "contcts": "contacts",
    "admision": "admission",
    "admissin": "admission",
    "tution": "tuition",
    "tuitionn": "tuition",
    "universty": "university",
    "univ": "university",
    "clases": "classes",
    "clase": "class",
    "programe": "program",
    "programes": "programs",
    "departement": "department",
    "deparment": "department",
    "studnt": "student",
    "studnts": "students",
    "servic": "service",
    "servces": "services",
    "offic": "office",
    "opn": "open",
    "dealine": "deadline",
    "deadlin": "deadline",
    "financal": "financial",
    "documets": "documents",
    "requriments": "requirements",
    "requirments": "requirements",
    "regster": "register",
    "enrolment": "enrollment",
    "enrollmnt": "enrollment"
}

# --- STEP 4: SLANG MAP (40+ Mappings) ---
SLANG_MAP = {
    "u": "you",
    "ur": "your",
    "pls": "please",
    "plz": "please",
    "bro": "",
    "broo": "",
    "yo": "hello",
    "yooo": "hello",
    "wassup": "hello",
    "sup": "hello",
    "idk": "i do not know",
    "thx": "thanks",
    "tmrw": "tomorrow",
    "wanna": "want to",
    "gonna": "going to",
    "r": "are",
    "k": "ok",
    "okay": "ok",
    "hey": "hello",
    "heyy": "hello",
    "info": "information",
    "cs": "computer science",
    "se": "software engineering",
    "mech": "mechanical engineering",
    "bus": "business administration",
    "admin": "administration",
    "dept": "department",
    "sch": "school",
    "scholar": "scholarship",
    "tel": "telephone",
    "cell": "phone",
    "number": "phone",
    "email": "contact",
    "mail": "contact",
    "pics": "images",
    "map": "location",
    "site": "website",
    "web": "website",
    "link": "website",
    "portal": "website",
    "uni": "university",
    "coll": "college"
}

def normalize_repeated_letters(text: str) -> str:
    """
    Collapses 3 or more repeated letters down to a single letter.
    Preserves valid double spellings (e.g., 'fees', 'school', 'good').
    Example: 'helloooo' -> 'hello', 'brooo' -> 'bro'
    """
    if not text:
        return ""
    return re.sub(r'(.)\1{2,}', r'\1', text)

def apply_mapping_boundaries(text: str, mapping: dict) -> str:
    """
    Safely applies slang or typo replacements using regex word boundaries.
    Prevents replacing sub-strings inside larger words.
    """
    if not text:
        return ""
    
    # Process word by word or phrase by phrase
    for key, value in mapping.items():
        pattern = r'\b' + re.escape(key) + r'\b'
        text = re.sub(pattern, value, text)
    return text

def preprocess_text(text: str, debug: bool = False) -> str:
    """
    Main Phase 3 Preprocessing Pipeline.
    Pipeline stages:
    1. Lowecase conversion
    2. Punctuation & useless symbols removal
    3. Normalize repeated letters (collapse 3+ repeats)
    4. Typo correction
    5. Slang normalization
    6. Whitespace cleanup
    7. Tokenization and stopword removal (preserving core educational words)
    8. Optional Lemmatization (dual-pass noun & verb)
    
    Returns clean, deterministic, normalized text.
    Guaranteed never to crash.
    """
    original_input = text
    
    # 0. Crash safety guard for null or non-string inputs
    if not text or not isinstance(text, str):
        if debug:
            print(f"Original: {original_input}")
            print("Processed: ")
        return ""

    # Stage 1: Lowercase conversion
    processed = text.lower().strip()

    # Stage 2: Punctuation and useless symbols removal
    # Replace punctuation symbols with a single space to prevent string merging
    punctuation_pattern = re.compile(f"[{re.escape(string.punctuation)}]")
    processed = punctuation_pattern.sub(" ", processed)

    # Stage 3: Normalize repeated letters (e.g., 'helloooo' -> 'hello')
    processed = normalize_repeated_letters(processed)

    # Stage 4: Typo correction (multi-word and single-word)
    processed = apply_mapping_boundaries(processed, TYPO_MAP)

    # Stage 5: Slang normalization
    processed = apply_mapping_boundaries(processed, SLANG_MAP)

    # Stage 6: Whitespace cleanup (collapses multiple spaces)
    processed = re.sub(r'\s+', ' ', processed).strip()

    # Stage 7: Token cleanup (Tokenization & Stopword removal)
    try:
        tokens = word_tokenize(processed)
    except Exception:
        tokens = processed.split()

    filtered_tokens = []
    for token in tokens:
        if token not in PRESERVED_STOPWORDS:
            # Stage 8: Lemmatization (Dual-pass noun and verb)
            lemma = lemmatizer.lemmatize(token, pos='v')
            lemma = lemmatizer.lemmatize(lemma, pos='n')
            filtered_tokens.append(lemma)

    final_text = " ".join(filtered_tokens).strip()

    # Stage 9: Debug logging (no emojis used for console safety)
    if debug:
        print(f"Original: {original_input}")
        print(f"Processed: {final_text}")
        print("-" * 40)

    return final_text

# --- BACKWARD COMPATIBILITY ALIAS ---
def clean_text(text: str) -> str:
    """
    Alias wrapper pointing to preprocess_text to preserve compatibility
    with external calling modules.
    """
    return preprocess_text(text, debug=False)


def detect_ood(text: str) -> bool:
    """
    Step 8.3: Detects if the user query is out-of-domain (OOD).
    Converts query to lowercase, checks for matches against config.OOD_KEYWORDS, 
    and returns True if any match is found, otherwise False.
    """
    if not text:
        return False
    
    # 1. Lowercase input
    lower_text = text.lower()
    
    # 2. Clean input punctuation to get clean words
    clean_text = re.sub(r"[^\w\s]", "", lower_text)
    words = clean_text.split()
    
    for kw in config.OOD_KEYWORDS:
        kw_clean = kw.lower()
        if " " in kw_clean:
            if kw_clean in lower_text:
                return True
        else:
            # Check for word boundaries or simple substring inclusions (e.g. 'cook' matches 'cooking')
            if kw_clean in words:
                return True
            if any(kw_clean in w for w in words):
                return True
                
    return False


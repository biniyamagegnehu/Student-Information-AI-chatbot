import re
import unicodedata
import nltk
import spacy
from spacy.pipeline import EntityRuler
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
from datetime import datetime

# Optional: RapidFuzz for typo correction
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    import difflib
    HAS_RAPIDFUZZ = False

# --- NLTK Setup ---
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- spaCy NER Setup ---
try:
    nlp = spacy.load("en_core_web_sm")
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        patterns = [
            {"label": "STUDENT_ID", "pattern": [{"TEXT": {"REGEX": "^[0-9]{5,8}$"}}]},
            {"label": "DEPARTMENT", "pattern": [{"LOWER": "computer"}, {"LOWER": "science"}]},
            {"label": "DEPARTMENT", "pattern": [{"LOWER": "software"}, {"LOWER": "engineering"}]},
            {"label": "BUILDING", "pattern": [{"LOWER": "block"}, {"LOWER": {"REGEX": "^[a-z0-9]+$"}}]},
            {"label": "BUILDING", "pattern": [{"LOWER": "library"}]},
            {"label": "BUILDING", "pattern": [{"LOWER": "hostel"}]},
            {"label": "OFFICE", "pattern": [{"LOWER": "registrar"}]},
            {"label": "INSTRUCTOR", "pattern": [{"LOWER": {"IN": ["professor", "prof", "dr", "dr."]}}, {"IS_ALPHA": True}]},
            {"label": "SEMESTER", "pattern": [{"LOWER": "semester"}, {"LIKE_NUM": True}]},
            {"label": "PAYMENT", "pattern": [{"LOWER": "tuition"}]},
            {"label": "PAYMENT", "pattern": [{"LOWER": "fee"}]}
        ]
        ruler.add_patterns(patterns)
except Exception:
    print("Warning: spaCy model 'en_core_web_sm' not found. NER will be limited.")
    nlp = None

# --- Preprocessing Constants ---
SHORTHAND_EXPANSIONS = {
    "pls": "please", "plz": "please", "info": "information", "admin": "administration",
    "dept": "department", "cs": "computer science", "it": "information technology",
    "lab": "laboratory", "dorm": "dormitory", "registar": "registrar", "uni": "university",
    "asap": "urgently", "wher": "where"
}

DOMAIN_VOCABULARY = [
    "registration", "semester", "transcript", "library", "registrar", "fee", "tuition",
    "scholarship", "exam", "course", "hostel", "admission", "department", "office", "campus",
    "block", "dormitory", "schedule", "deadline", "payment", "instructor", "professor",
    "timetable", "routine", "dorm", "cafeteria", "clearance", "enrollment", "transfer",
    "holiday", "vacation", "ceremony", "grade", "results", "clinic", "stadium"
]

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class InputValidator:
    """
    Sanitizes and validates user input before it hits the NLP pipeline.
    """
    MAX_INPUT_LENGTH = 150
    
    @staticmethod
    def validate_and_sanitize(text):
        if not text or not text.strip():
            return False, "", "Please enter a valid message."
            
        text = unicodedata.normalize('NFKC', text)
            
        if len(text) > InputValidator.MAX_INPUT_LENGTH:
            return False, "", f"Your message is too long (Max {InputValidator.MAX_INPUT_LENGTH} chars)."
            
        if re.search(r'<[^>]+>', text):
            text = re.sub(r'<[^>]+>', '', text)
            
        if re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text):
            return False, "", "URLs are not allowed for security reasons."
            
        text = re.sub(r'(.)\1{3,}', r'\1\1', text) 
        text = re.sub(r'[^\w\s\.\?\!\,\-\'\"]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            return False, "", "Your message contained invalid characters."
            
        return True, text, ""

def apply_typo_correction(word):
    """
    Applies shorthand expansion and fuzzy spell correction.
    """
    if word in SHORTHAND_EXPANSIONS:
        return SHORTHAND_EXPANSIONS[word]
        
    if len(word) <= 3 or word in stop_words:
        return word
        
    if HAS_RAPIDFUZZ:
        match = process.extractOne(word, DOMAIN_VOCABULARY, scorer=fuzz.ratio, score_cutoff=80)
        if match:
            return match[0]
    else:
        import difflib
        matches = difflib.get_close_matches(word, DOMAIN_VOCABULARY, n=1, cutoff=0.8)
        if matches:
            return matches[0]
            
    return word

def preprocess_text(text):
    """
    Cleans and normalizes text for the ML model.
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        words = word_tokenize(text)
    except LookupError:
        words = text.split()
    
    clean_words = []
    for word in words:
        corrected_phrase = apply_typo_correction(word)
        for corrected_word in corrected_phrase.split():
            if corrected_word not in stop_words:
                lemma = lemmatizer.lemmatize(corrected_word)
                clean_words.append(lemma)
            
    return ' '.join(clean_words)

def extract_all_entities(text):
    """
    Extracts university-related entities using spaCy and Regex.
    """
    entities = []
    if nlp:
        doc = nlp(text)
        entities = [(ent.label_, ent.text) for ent in doc.ents]
    
    # Regex Fallback for ID
    id_match = re.search(r'\b\d{5,8}\b', text)
    if id_match:
        entities.append(("STUDENT_ID", id_match.group(0)))
        
    return list(set(entities))


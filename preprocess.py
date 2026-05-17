# preprocess.py
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
stop_words = set(stopwords.words('english'))

# --- TYPO NORMALIZATION DICTIONARY ---
TYPO_DICTIONARY = {
    "regstration": "registration",
    "registraion": "registration",
    "regestration": "registration",
    "scheduel": "schedule",
    "scheduelle": "schedule",
    "examn": "exam",
    "exm": "exam",
    "cources": "courses",
    "cource": "course",
    "scholrship": "scholarship",
    "scholership": "scholarship",
    "calender": "calendar",
    "acadmic": "academic",
    "libary": "library",
    "locaton": "location",
    "contat": "contact",
    "contcts": "contacts",
    "admision": "admission", # While admission is disallowed, keep correction clean
    "fee": "fees"
}

def clean_text(text: str) -> str:
    """
    Standard preprocessing pipeline used consistently during training and inference.
    Processes string through:
    1. Lowercase conversion
    2. Typo normalization
    3. Punctuation removal
    4. Tokenization
    5. Stopword filtering
    6. Lemmatization
    7. Whitespace cleanup
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Lowercase conversion
    text = text.lower().strip()

    # 2. Split words to normalize typos using dictionary
    words = text.split()
    normalized_words = [TYPO_DICTIONARY.get(word, word) for word in words]
    text = " ".join(normalized_words)

    # 3. Punctuation removal (keep spaces)
    # Replaces punctuation symbols with a single space
    punctuation_pattern = re.compile(f"[{re.escape(string.punctuation)}]")
    text = punctuation_pattern.sub(" ", text)

    # 4. Tokenization (handles multi-spacing cleanly)
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # 5 & 6. Stopword removal & Lemmatization
    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words:
            # Lemmatize both noun and verb forms for max generalizability
            lemma = lemmatizer.lemmatize(token, pos='v')
            lemma = lemmatizer.lemmatize(lemma, pos='n')
            cleaned_tokens.append(lemma)

    # 7. Whitespace cleanup & join
    return " ".join(cleaned_tokens).strip()

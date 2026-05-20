# config.py
import os

# --- RANDOM SEED ---
RANDOM_SEED = 42

# --- CONFIGURATION PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
LOG_DIR = os.path.join(BASE_DIR, "logs")

INTENTS_JSON_PATH = os.path.join(BASE_DIR, "intents.json")
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "knowledge_base.json")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

CONVERSATION_HISTORY_PATH = os.path.join(LOG_DIR, "conversation_history.csv")
APP_LOG_PATH = os.path.join(LOG_DIR, "app.log")

# --- ML SETTINGS ---
CONFIDENCE_THRESHOLD = 0.65

INTENT_THRESHOLDS = {
    "greeting": 0.35,
    "goodbye": 0.35,
    "thanks": 0.35,
    "help": 0.35,
    "registration": 0.60,
    "courses": 0.55,
    "fees": 0.60,
    "exam": 0.60,
    "schedule": 0.55,
    "location": 0.60,
    "contacts": 0.55,
    "scholarship": 0.60,
    "student_services": 0.55
}

CHATBOT_DEBUG = False

# --- safety & ood keywords ---
OOD_KEYWORDS = [
    "weather",
    "football",
    "bitcoin",
    "crypto",
    "joke",
    "cook",
    "movie",
    "celebrity",
    "stock",
    "match",
    "music",
    "basketball",
    "politics",
    "dating",
    "hacking",
    "programming"
]


# --- STRICT INTENT WHITELIST ---
ALLOWED_INTENTS = {
    "registration",
    "courses",
    "fees",
    "exam",
    "schedule",
    "location",
    "contacts",
    "scholarship",
    "student_services",
    "greeting",
    "goodbye",
    "thanks",
    "help",
    "fallback"
}

# --- GLOBAL FALLBACK RESPONSES ---
FALLBACK_RESPONSES = [
    "I'm designed only for university student information such as registration, courses, tuition & fees, exams, academic calendar, campus locations, contacts, scholarships, and student services. Please ask a question related to these topics.",
    "I can only assist with university student services, registration, fees, courses, exams, academic calendars, locations, and scholarships. Let me know how I can help with these areas.",
    "I am only trained to answer questions within the university domain (e.g., exams, registration, courses, fees, calendars, locations, scholarships). Please specify a topic within these boundaries."
]

# --- ADVANCED FALLBACK TYPES ---
FALLBACK_TYPES = {
    "out_of_domain": "I can only assist with university student information such as registration, courses, fees, exams, calendars, locations, and scholarships.",
    "missing_context": "Could you clarify which service, office, department, or topic you mean?",
    "low_confidence": "I'm not fully sure I understood your query. Could you please rephrase it?",
    "missing_location": "I can help with university buildings, departments, offices, and student services. Which location are you asking about?"
}



import os
import csv
import logging
from datetime import datetime
import config

# STEP 10.1 — CREATE LOGS DIRECTORY
if not os.path.exists(config.LOG_DIR):
    os.makedirs(config.LOG_DIR)

def setup_logging():
    """
    STEP 10.3 — IMPLEMENT PYTHON LOGGING
    Configures the root logger to write to logs/app.log
    """
    logger = logging.getLogger("ChatbotApp")
    logger.setLevel(logging.INFO)
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        file_handler = logging.FileHandler(config.APP_LOG_PATH, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

logger = setup_logging()

def init_history_log():
    """
    STEP 10.2 & 10.6 — IMPLEMENT CSV CONVERSATION LOGGING & SAFE CSV WRITING
    Auto-creates header if file does not exist.
    """
    # Merging Phase 8 (is_ood) with Phase 10 (entities)
    expected_header = ["timestamp", "user_input", "intent", "confidence", "entities", "response", "is_fallback", "is_ood"]
    file_exists = os.path.exists(config.CONVERSATION_HISTORY_PATH)
    needs_init = not file_exists
    
    if file_exists:
        try:
            with open(config.CONVERSATION_HISTORY_PATH, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header != expected_header:
                    needs_init = True
        except Exception:
            needs_init = True

    if needs_init:
        try:
            with open(config.CONVERSATION_HISTORY_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(expected_header)
        except Exception as e:
            logger.error(f"Failed to initialize history CSV: {e}")

# Initialize history log at module import
init_history_log()

def log_interaction(query: str, intent: str, confidence: float, response: str, is_fallback: bool, is_ood: bool, entities: list = None):
    """
    STEP 10.2 & 10.4 & 10.5 — LOG CONVERSATION, ENTITIES, AND FALLBACKS
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Handle None entities safely
    entities_str = str(entities) if entities else "None"
    
    # 1. Log to CSV
    try:
        with open(config.CONVERSATION_HISTORY_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, query, intent, f"{confidence:.4f}", entities_str,
                response, str(is_fallback), str(is_ood)
            ])
    except Exception as e:
        logger.error(f"Failed to write to conversation history CSV: {e}")

    # 2. Log to app.log
    if is_ood:
        logger.warning(f"OOD query rejected: '{query}'")
    elif is_fallback:
        logger.warning(f"Fallback triggered for query: '{query}' | Intent: {intent} | Confidence: {confidence:.2f}")
    else:
        logger.info(f"Query: '{query}' | Intent: {intent} | Confidence: {confidence:.2f} | Entities: {entities_str}")

# STEP 10.8 — IMPLEMENT AUDIT UTILITIES
def count_fallback_rate():
    """Returns (total_fallbacks, total_queries, fallback_percentage)"""
    total = 0
    fallbacks = 0
    try:
        with open(config.CONVERSATION_HISTORY_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                if row.get('is_fallback') == 'True' or row.get('intent') == 'fallback':
                    fallbacks += 1
    except Exception:
        pass
    rate = (fallbacks / total * 100) if total > 0 else 0.0
    return fallbacks, total, rate

def get_intent_frequency():
    """Returns a dictionary of intent counts"""
    counts = {}
    try:
        with open(config.CONVERSATION_HISTORY_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                intent = row.get('intent', 'unknown')
                counts[intent] = counts.get(intent, 0) + 1
    except Exception:
        pass
    return counts

def get_weak_intents(threshold=0.75):
    """Returns a list of queries with confidence below threshold but not OOD"""
    weak = []
    try:
        with open(config.CONVERSATION_HISTORY_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    conf = float(row.get('confidence', 0))
                    is_ood = row.get('is_ood') == 'True'
                    if not is_ood and conf < threshold:
                        weak.append({
                            'query': row.get('user_input'),
                            'intent': row.get('intent'),
                            'confidence': conf
                        })
                except ValueError:
                    continue
    except Exception:
        pass
    return weak

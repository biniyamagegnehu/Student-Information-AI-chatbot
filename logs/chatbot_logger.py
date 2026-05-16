import os
import csv
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

# --- DIRECTORY SETUP ---
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# --- 1. SYSTEM LOGGER (INFO, WARNING, ERROR) ---
# Used for debugging, session tracking, and system errors
sys_logger = logging.getLogger("ChatbotSystem")
sys_logger.setLevel(logging.DEBUG)

# File Handler with basic rotation (Max 1MB per file, keep last 3)
sys_handler = RotatingFileHandler(os.path.join(LOG_DIR, "system_debug.log"), maxBytes=1024*1024, backupCount=3)
sys_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
sys_handler.setFormatter(sys_formatter)
sys_logger.addHandler(sys_handler)

# Console Handler (optional, for terminal output)
console_handler = logging.StreamHandler()
console_handler.setFormatter(sys_formatter)
console_handler.setLevel(logging.INFO)
sys_logger.addHandler(console_handler)

# --- 2. CONVERSATION & DATASET LOGGER ---
class ChatbotLogger:
    """
    Handles structured logging of user interactions for dataset improvement and evaluation.
    Logs to both a human-readable JSONL (JSON Lines) and a dataset-ready CSV file.
    """
    CSV_FILE = os.path.join(LOG_DIR, "conversation_history.csv")
    JSON_FILE = os.path.join(LOG_DIR, "conversation_history.jsonl")
    LOW_CONFIDENCE_FILE = os.path.join(LOG_DIR, "low_confidence_queries.csv")

    @classmethod
    def setup(cls):
        """Initializes the CSV headers if the files do not exist."""
        sys_logger.info("Chatbot Session Started.")
        if not os.path.exists(cls.CSV_FILE):
            with open(cls.CSV_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "user_input", "intent", "confidence", "entities", "response", "is_fallback"])
                
        if not os.path.exists(cls.LOW_CONFIDENCE_FILE):
            with open(cls.LOW_CONFIDENCE_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "user_input", "confidence", "predicted_intent"])

    @classmethod
    def log_interaction(cls, user_input, intent, confidence, entities, response, is_fallback=False):
        """Logs a structured interaction to CSV and JSONL."""
        timestamp = datetime.now().isoformat()
        
        # 1. Log to JSONL (Great for complex data like lists of entities)
        log_entry = {
            "timestamp": timestamp,
            "user_input": user_input,
            "intent": intent,
            "confidence": round(confidence, 4),
            "entities": entities,
            "response": response,
            "is_fallback": is_fallback
        }
        with open(cls.JSON_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        # 2. Log to CSV (Great for pandas dataset retraining)
        entities_str = str(entities) if entities else "None"
        with open(cls.CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, user_input, intent, round(confidence, 4), entities_str, response, is_fallback])
            
        # 3. Dedicated Low-Confidence Logging
        if is_fallback:
            sys_logger.warning(f"Fallback Triggered! Input: '{user_input}' | Confidence: {confidence:.2f}")
            with open(cls.LOW_CONFIDENCE_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, user_input, round(confidence, 4), intent])
        else:
            sys_logger.info(f"Successful Prediction: '{user_input}' -> {intent} (Conf: {confidence:.2f})")

    @classmethod
    def log_invalid_input(cls, text, reason):
        """Logs suspicious, spam, or malicious input."""
        sys_logger.warning(f"INVALID INPUT [{reason}]: {text}")
        
    @classmethod
    def end_session(cls):
        sys_logger.info("Chatbot Session Ended.")

# Initialize headers on import
ChatbotLogger.setup()

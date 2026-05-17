# app.py
import os
import csv
import logging
import joblib
from datetime import datetime

# Local modular imports
import config
from preprocess import clean_text
from responses import get_response

# --- AUTOMATIC LOGGING DIRECTORY SETUP ---
if not os.path.exists(config.LOG_DIR):
    os.makedirs(config.LOG_DIR)

# --- 1. GENERAL APP LOGGER SETUP ---
logging.basicConfig(
    filename=config.APP_LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger("ChatbotApp")

# --- 2. CONVERSATION HISTORY LOG INITIALIZER ---
def init_history_log():
    """
    Creates conversation_history.csv with appropriate columns if it doesn't exist.
    """
    if not os.path.exists(config.CONVERSATION_HISTORY_PATH):
        with open(config.CONVERSATION_HISTORY_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "user_query", "predicted_intent", "confidence", "response", "fallback_triggered"])

init_history_log()

def log_interaction(query: str, intent: str, confidence: float, response: str, fallback: bool):
    """
    Saves the interaction transaction to both conversation_history.csv and app.log.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log to CSV
    try:
        with open(config.CONVERSATION_HISTORY_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, query, intent, f"{confidence:.4f}", response, str(fallback)])
    except Exception as e:
        logger.error(f"Failed to write to conversation history CSV: {e}")

    # Log to app.log
    log_msg = f"Query: '{query}' | Intent: '{intent}' (Conf: {confidence:.2f}) | Fallback: {fallback}"
    if fallback:
        logger.warning(log_msg)
    else:
        logger.info(log_msg)


# --- CHATBOT ENGINE ---
class ChatbotEngine:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.is_loaded = False
        self.load_artifacts()

    def load_artifacts(self):
        """
        Loads trained model, vectorizer, and label encoder.
        Gracefully handles file-missing scenarios.
        """
        if (os.path.exists(config.MODEL_PATH) and 
            os.path.exists(config.VECTORIZER_PATH) and 
            os.path.exists(config.LABEL_ENCODER_PATH)):
            try:
                self.model = joblib.load(config.MODEL_PATH)
                self.vectorizer = joblib.load(config.VECTORIZER_PATH)
                self.label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
                self.is_loaded = True
                logger.info("Successfully loaded all ML model artifacts.")
            except Exception as e:
                logger.error(f"Error loading model files: {e}")
                print(f"[ERROR] Failed to load model files: {e}")
        else:
            logger.warning("Model files missing. Application starting in fallback-only mode.")

    def get_reply(self, raw_text: str):
        """
        Processes query: Clean -> Vectorize -> Predict -> Threshold Check -> Log -> Return
        """
        raw_text_clean = raw_text.strip()
        if not raw_text_clean:
            return "Please enter a valid message.", "none", 0.0, False

        # 1. Pipeline Preprocessing
        processed_query = clean_text(raw_text_clean)

        # 2. Check if model is available, otherwise trigger safety fallback
        if not self.is_loaded:
            response = get_response("fallback") # Out of scope
            log_interaction(raw_text_clean, "untrained_model", 0.0, response, True)
            return response, "untrained_model", 0.0, True

        # 3. Model Prediction
        try:
            vec = self.vectorizer.transform([processed_query])
            probabilities = self.model.predict_proba(vec)[0]
            max_index = probabilities.argmax()
            confidence = float(probabilities[max_index])
            intent = self.label_encoder.inverse_transform([max_index])[0]
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            response = get_response("fallback")
            log_interaction(raw_text_clean, "prediction_error", 0.0, response, True)
            return response, "prediction_error", 0.0, True

        # 4. Strict Scope & Confidence Threshold Guardrails
        fallback_triggered = False
        if confidence < config.CONFIDENCE_THRESHOLD:
            intent = "fallback"
            fallback_triggered = True
            
        if intent not in config.ALLOWED_INTENTS:
            intent = "fallback"
            fallback_triggered = True

        # 5. Fetch Final Response
        response = get_response(intent)

        # 6. Log Transaction
        log_interaction(raw_text_clean, intent, confidence, response, fallback_triggered)

        return response, intent, confidence, fallback_triggered


# --- INTERACTIVE CLI TEST ENVIRONMENT ---
def run_cli():
    print("=" * 60)
    print("      UNIVERSITY STUDENT INFORMATION ASSISTANT (CLI v1.0)")
    print("=" * 60)
    print("Domain Focus: Registration, Courses, Fees, Exams, Calendars, Locations,")
    print("              Contacts, Scholarships, and Student Services.")
    print("-" * 60)
    print("System is online. Type 'exit', 'quit', or 'bye' to end the session.\n")

    engine = ChatbotEngine()
    
    if not engine.is_loaded:
        print("[WARNING] Model artifacts not found inside 'model/'.")
        print("          Running in Safe Fallback/Ad-hoc mode. Please run train.py first!\n")
    else:
        print("[SYSTEM] Calibrated classifier and TF-IDF pipeline loaded successfully.")
        print(f"[SYSTEM] Confidence Threshold set to {config.CONFIDENCE_THRESHOLD:.2f}")
    
    print("-" * 60)
    print("Assistant: Welcome! How can I help you with your student inquiries today?\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAssistant: Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nAssistant: Goodbye! Have a wonderful day studying.")
            break

        # Process prediction
        response, intent, conf, fallback = engine.get_reply(user_input)

        # Output reply
        print(f"Assistant: {response}")
        print(f"  [Debug] Predicted Intent: {intent} (Confidence: {conf:.2f}) | Fallback Triggered: {fallback}")
        print("-" * 60)

if __name__ == "__main__":
    run_cli()

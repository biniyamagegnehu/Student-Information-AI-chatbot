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
from memory import ConversationMemory
from utils import extract_entities, is_ambiguous_query, resolve_context

# --- AUTOMATIC LOGGING DIRECTORY SETUP ---
if not os.path.exists(config.LOG_DIR):
    os.makedirs(config.LOG_DIR)

# --- 1. GENERAL APP SYSTEM LOGGER ---
logging.basicConfig(
    filename=config.APP_LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger("ChatbotApp")

# --- 2. ENHANCED CONVERSATION HISTORY LOG INITIALIZER ---
def init_history_log():
    """
    Creates conversation_history.csv with updated fields for Phase 2 auditing.
    """
    if not os.path.exists(config.CONVERSATION_HISTORY_PATH):
        try:
            with open(config.CONVERSATION_HISTORY_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "user_query", "predicted_intent", "confidence", 
                    "response", "fallback_triggered", "context_used", 
                    "previous_intent", "fallback_reason"
                ])
        except Exception as e:
            print(f"[Warning] Failed to initialize history CSV: {e}")

init_history_log()

def log_interaction(query: str, intent: str, confidence: float, response: str, 
                    fallback: bool, context_used: bool, prev_intent: str, fallback_reason: str):
    """
    Saves a complete conversation interaction transaction with context fields to CSV and app.log.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Log to CSV
    try:
        with open(config.CONVERSATION_HISTORY_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, query, intent, f"{confidence:.4f}", 
                response, str(fallback), str(context_used), 
                str(prev_intent), str(fallback_reason)
            ])
    except Exception as e:
        logger.error(f"Failed to write to conversation history CSV: {e}")

    # 2. Log to app.log
    log_msg = (
        f"Query: '{query}' | Intent: '{intent}' (Conf: {confidence:.2f}) | "
        f"Fallback: {fallback} | ContextUsed: {context_used} | "
        f"PrevIntent: {prev_intent} | FallbackReason: {fallback_reason}"
    )
    if fallback:
        logger.warning(log_msg)
    else:
        logger.info(log_msg)


# --- CHATBOT RUNTIME ENGINE ---
class ChatbotEngine:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.is_loaded = False
        
        # Instantiate in-session memory tracking
        self.memory = ConversationMemory()
        
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
            logger.warning("Model files missing. Chatbot starting in fallback-only mode.")

    def get_reply(self, raw_text: str):
        """
        Processes query: 
        1. Extract entities
        2. Detect ambiguous follow-up
        3. If ambiguous -> Resolve context using Memory
        4. If normal -> ML Predict
        5. Check Thresholds, Safety Whitelists & Fallback Types
        6. Dynamic Template Dispatch
        7. Save Memory state & Log
        """
        raw_text_clean = raw_text.strip()
        if not raw_text_clean:
            return "Please enter a valid message.", "none", 0.0, False

        # --- SESSION MEMORY VIEW ---
        memory_state = self.memory.get_state()
        prev_intent = memory_state.get("last_intent")

        # --- STEP 1: ENTITY EXTRACTION ---
        extracted_entities = extract_entities(raw_text_clean)

        # --- VARIABLES INITIALIZATION ---
        intent = "fallback"
        confidence = 1.0
        fallback_triggered = False
        context_used = False
        fallback_reason = None

        # --- STEP 1.5: HYBRID INTENT RESOLUTION (HEURISTIC ENGINE OVERRIDE) ---
        raw_lower = raw_text_clean.lower()
        heuristics_matched = False

        if extracted_entities:
            # A. Spatial locations check
            if any(w in raw_lower for w in ["where", "location", "block", "room", "office", "find", "map", "address", "building", "floor"]):
                if any(k in extracted_entities for k in ["department", "office", "student_services"]):
                    intent = "location"
                    confidence = 1.0
                    heuristics_matched = True
            
            # B. Fees check
            elif any(w in raw_lower for w in ["how much", "cost", "price", "tuition", "fee", "fees", "pay", "payment"]):
                if "department" in extracted_entities:
                    intent = "fees"
                    confidence = 1.0
                    heuristics_matched = True

            # C. Exams check
            elif any(w in raw_lower for w in ["exam", "exams", "examination", "test", "schedule", "timetable", "routine", "midterm", "final"]):
                if "department" in extracted_entities:
                    intent = "exam"
                    confidence = 1.0
                    heuristics_matched = True

            # D. Contacts check
            elif any(w in raw_lower for w in ["contact", "reach", "email", "phone", "call", "write", "number"]):
                if any(k in extracted_entities for k in ["department", "office"]):
                    intent = "contacts"
                    confidence = 1.0
                    heuristics_matched = True

            # E. Scholarships check
            elif any(w in raw_lower for w in ["scholarship", "scholarships", "coverage", "waiver", "aid"]):
                if "scholarship" in extracted_entities:
                    intent = "scholarship"
                    confidence = 1.0
                    heuristics_matched = True

            # F. Student Services check
            elif any(w in raw_lower for w in ["service", "services", "counseling", "career", "library"]):
                if "student_services" in extracted_entities:
                    intent = "student_services"
                    confidence = 1.0
                    heuristics_matched = True

        # --- STEP 2: CONTEXT RESOLUTION & FOLLOW-UP HANDLING ---
        if heuristics_matched:
            pass
        elif is_ambiguous_query(raw_text_clean):
            # Resolve intent/topic based on context
            inferred_intent, inferred_topic, resolved_entities = resolve_context(raw_text_clean, memory_state)
            
            if inferred_intent == "fallback" and inferred_topic == "missing_context":
                fallback_reason = "missing_context"
                fallback_triggered = True
            else:
                intent = inferred_intent
                # Merge entities
                extracted_entities.update(resolved_entities)
                context_used = True
                confidence = 0.90 # Standard assumed confidence for contextual match
        else:
            # --- STEP 3: ML PREDICTION FOR NORMAL QUERIES ---
            processed_query = clean_text(raw_text_clean)

            if not self.is_loaded:
                # Untrained fallback trigger
                intent = "fallback"
                fallback_reason = "low_confidence"
                fallback_triggered = True
            else:
                try:
                    vec = self.vectorizer.transform([processed_query])
                    probabilities = self.model.predict_proba(vec)[0]
                    max_index = probabilities.argmax()
                    confidence = float(probabilities[max_index])
                    predicted_intent = self.label_encoder.inverse_transform([max_index])[0]
                    
                    # Confidence Guardrail
                    effective_threshold = config.CONFIDENCE_THRESHOLD
                    if extracted_entities:
                        # Lower the required threshold if we have extracted a matching domain entity
                        if predicted_intent == "location" and any(k in extracted_entities for k in ["department", "office", "student_services"]):
                            effective_threshold = 0.25
                        elif predicted_intent == "fees" and "department" in extracted_entities:
                            effective_threshold = 0.25
                        elif predicted_intent == "exam" and "department" in extracted_entities:
                            effective_threshold = 0.25
                        elif predicted_intent == "contacts" and any(k in extracted_entities for k in ["department", "office"]):
                            effective_threshold = 0.25
                        elif predicted_intent == "scholarship" and "scholarship" in extracted_entities:
                            effective_threshold = 0.25
                        elif predicted_intent == "student_services" and "student_services" in extracted_entities:
                            effective_threshold = 0.25

                    if confidence < effective_threshold:
                        intent = "fallback"
                        fallback_reason = "low_confidence"
                        fallback_triggered = True
                    else:
                        intent = predicted_intent
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    intent = "fallback"
                    fallback_reason = "low_confidence"
                    fallback_triggered = True

        # --- STEP 4: STRICT WHITELIST SCOPE VALIDATION ---
        if not fallback_triggered and (intent not in config.ALLOWED_INTENTS or intent == "fallback"):
            intent = "fallback"
            if not fallback_reason:
                fallback_reason = "out_of_domain"
            fallback_triggered = True

        # --- STEP 5: RESPONSE DISPATCHING ---
        response = get_response(intent, extracted_entities, fallback_reason)

        # --- STEP 6: CONVERSATION MEMORY UPDATE ---
        # We don't save greeting, goodbye, thanks or fallbacks as active follow-up topics to prevent context pollution
        if intent not in ["greeting", "goodbye", "thanks", "fallback"]:
            self.memory.update(intent, intent, extracted_entities, response, raw_text_clean)
        elif intent == "fallback" and fallback_reason == "missing_context":
            # Keep previous context alive if context clarification was requested
            pass
        else:
            # Clear memory on goodbyes/greetings to reset context
            if intent in ["greeting", "goodbye"]:
                self.memory.clear()

        # --- STEP 7: LOG INTERACTION ---
        log_interaction(
            raw_text_clean, intent, confidence, response, 
            fallback_triggered, context_used, prev_intent, fallback_reason
        )

        return response, intent, confidence, fallback_triggered


# --- INTERACTIVE CLI TEST ENVIRONMENT ---
def run_cli():
    print("=" * 60)
    print("      UNIVERSITY STUDENT INFORMATION ASSISTANT (CLI v2.0)")
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

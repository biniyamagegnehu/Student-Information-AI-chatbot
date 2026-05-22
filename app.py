# app.py
import os
import csv
import logging
import joblib
from datetime import datetime

# Local modular imports
import config
from preprocess import preprocess_text, detect_ood
from context_manager import ContextManager
from responses import get_response
from ner import extract_entities

from logger_utils import logger, log_interaction


# --- CHATBOT RUNTIME ENGINE ---
class ChatbotEngine:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.is_loaded = False
        
        # Instantiate in-session memory tracking
        self.memory = ContextManager()
        
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
        1. Preprocess text
        2. Run OOD detection
        3. If OOD -> Bypass classifier entirely, trigger fallback response
        4. If normal -> ML Predict
        5. Check Thresholds, Safety Whitelists & Fallback Types
        6. Dynamic Template Dispatch
        7. Save Memory state & Log
        """
        raw_text_clean = raw_text.strip()
        if not raw_text_clean:
            return "Please enter a valid message.", "none", 0.0, False

        # --- STEP 0.5: OOD DETECTION (Step 8.4) ---
        is_ood = detect_ood(raw_text_clean)
        if is_ood:
            intent = "fallback"
            confidence = 0.0
            fallback_triggered = True
            fallback_reason = "out_of_domain"
            
            # Generate the professional fallback response (Step 8.6)
            response = get_response(
                intent, 
                entities={}, 
                fallback_reason=fallback_reason, 
                query=raw_text_clean, 
                context_used=False,
                debug=config.CHATBOT_DEBUG
            )
            
            # Log OOD warning (Step 8.7)
            logger.warning(f"OOD Event Detected! Input: '{raw_text_clean}' | Bypassing classifier.")
            
            # Log to CSV (Step 8.7)
            log_interaction(
                raw_text_clean, intent, confidence, response, fallback_triggered, is_ood=True, entities=[]
            )
            
            # Print debug info (Step 8.8) - only when debug mode is enabled
            if config.CHATBOT_DEBUG:
                print("\n==================================================")
                print("[DEBUG]")
                print(f"Input: {raw_text_clean}")
                print(f"OOD Detected: {is_ood}")
                print("Classifier Skipped: True")
                print("Fallback Triggered: True")
                print("==================================================\n")
            
            return response, intent, confidence, fallback_triggered

        # --- SESSION MEMORY VIEW ---
        memory_state = self.memory.get_state()
        prev_intent = memory_state.get("last_intent")

        # --- STEP 1: ENTITY EXTRACTION ---
        extracted_entities_raw = extract_entities(raw_text_clean)
        
        # Build backward-compatible and uppercase dictionary representation of entities
        extracted_entities = {}
        for k, v in extracted_entities_raw:
            k_upper = k.upper()
            k_lower = k.lower()
            extracted_entities[k_upper] = v
            extracted_entities[k_lower] = v
            
            # Phase 6 backward-compatibility mapping
            if k_upper == "BUILDING" and v == "library":
                is_lib_location = any(
                    w in raw_text_clean.lower()
                    for w in ["where", "location", "find", "directions", "nearby", "map", "navigate", "wher"]
                )
                if not is_lib_location:
                    extracted_entities["student_services"] = "central library"
                    extracted_entities["STUDENT_SERVICES"] = "central library"
            if k_upper == "OFFICE":
                short_val = v.replace(" office", "").strip()
                extracted_entities["office"] = short_val
                extracted_entities["OFFICE"] = short_val
                if short_val == "student affairs":
                    extracted_entities["student_services"] = "student affairs"
            if k_upper == "SERVICE":
                if v == "scholarship":
                    extracted_entities["scholarship"] = "merit scholarship"
                if v in ["student services", "counseling", "career helpdesk", "central library", "health center", "student support office"]:
                    extracted_entities["student_services"] = v

        # --- VARIABLES INITIALIZATION ---
        intent = "fallback"
        confidence = 1.0
        fallback_triggered = False
        context_used = False
        fallback_reason = None

        # --- STEP 1.5: HYBRID INTENT RESOLUTION (HEURISTIC ENGINE OVERRIDE) ---
        # Priority: greeting > thanks > goodbye > contacts > student_services >
        # fees > registration > scholarship > location > schedule > help (last)
        raw_lower = raw_text_clean.lower()
        heuristics_matched = False

        # --- Phrase / keyword groups ---
        greeting_phrases = [
            "good morning", "good afternoon", "good evening", "hello campus",
            "hello", "hi ", "hi,", "hey i have", "hey ", "selam", "yo mate", "sup bot", "sup "
        ]
        thanks_phrases = [
            "thank you", "thanks", "thx", "appreciate it", "appreciate your",
            "much appreciated", "lifesaver", "great job bot", "cool thanks", "you are a lifesaver"
        ]
        goodbye_phrases = [
            "goodbye", "bye bye", "see you later", "take care", "exit chat",
            "shutting down", "i am done here", "i am done", "farewell"
        ]

        comm_words = ["phone", "number", "email", "contact", "reach", "telegram", "link"]
        office_context_words = ["support", "help desk", "registrar", "office", "department", "desk"]
        has_comm = any(w in raw_lower for w in comm_words)
        has_office_context = any(w in raw_lower for w in office_context_words)

        location_words = ["where", "wher", "location", "locate", "find", "directions", "nearby", "map", "navigate"]
        has_location_word = any(w in raw_lower for w in location_words)

        student_services_words = [
            "student support office", "support office", "health center location",
            "health center", "clinic location", "clinic", "counseling office",
            "counseling services", "counseling", "career helpdesk", "career services",
            "student services"
        ]
        is_library_location_query = (
            "library" in raw_lower
            and has_location_word
            and not any(w in raw_lower for w in ["service", "services", "borrow", "book", "open", "hours"])
        )
        has_student_services = (
            not is_library_location_query
            and (
                any(w in raw_lower for w in student_services_words)
                or any(k in extracted_entities for k in ["student_services", "STUDENT_SERVICES"])
                or extracted_entities.get("SERVICE", "") in [
                    "student services", "student support office", "counseling",
                    "career helpdesk", "health center"
                ]
            )
        )

        fees_core_words = ["pay", "payment", "fee", "fees", "tuition", "billing"]
        fees_amount_words = ["how much", "cost", "price", "deadline"]
        has_fees_payment = (
            any(w in raw_lower for w in fees_core_words)
            or (
                any(w in raw_lower for w in fees_amount_words)
                and any(w in raw_lower for w in fees_core_words + ["cafeteria", "dorm", "semester", "installment", "penalty"])
            )
        )
        finance_office_location = (
            has_fees_payment
            and any(w in raw_lower for w in ["where", "location", "find", "directions"])
            and any(w in raw_lower for w in ["finance office", "finance department", "cashier office"])
            and not any(w in raw_lower for w in ["cafeteria", "dorm", "registration fee", "tuition"])
        )

        registration_words = [
            "registration portal", "registration deadline", "late registration",
            "add/drop", "add a course", "drop a course", "drop a module",
            "enroll", "enrollment", "register", "registration"
        ]
        has_registration = (
            any(w in raw_lower for w in registration_words)
            or ("regist" in raw_lower and any(w in raw_lower for w in ["deadline", "portal", "error", "fee", "course", "module"]))
            or ("how to apply" in raw_lower and "scholarship" not in raw_lower)
            or ("apply" in raw_lower and "scholarship" not in raw_lower and "where" in raw_lower)
        )

        scholarship_words_broad = [
            "waiver", "aid", "grant", "sponsorship", "disabled students",
            "financial assistance", "scholarship", "financial aid", "tuition waiver", "stipend", "loan"
        ]
        has_broad_scholarship = any(w in raw_lower for w in scholarship_words_broad)

        location_buildings = [
            "stadium", "faculty building", "toilet", "restroom", "washroom",
            "library", "cafeteria", "block a", "block b", "block c", "registrar", "admin"
        ]
        has_location_target = (
            any(w in raw_lower for w in location_buildings)
            or any(k in extracted_entities for k in [
                "DEPARTMENT", "OFFICE", "BUILDING", "LOCATION",
                "department", "office", "student_services"
            ])
        )

        schedule_words = [
            "timetable", "routine", "class schedule", "lecture time", "class routine",
            "lecture timetable", "weekly class schedule", "lecture schedule",
            "when is my class", "academic calendar", "semester end", "semester start"
        ]
        exam_words = ["exam", "final", "midterm", "test", "quiz", "examination"]
        has_schedule_word = any(w in raw_lower for w in schedule_words)
        has_exam_word = any(w in raw_lower for w in exam_words)

        help_patterns = [
            "i need help", "help please", "what can you do", "how do you work",
            "show me available topics", "guide me", "tell me what you support",
            "tell me more", "can you assist me", "how can you help",
            "what can you do for me", "what kind of questions can i ask",
            "show me available categories", "what are your capabilities",
            "help menu", "chatbot options", "how does this chatbot work"
        ]
        help_blocked = (
            has_comm
            or has_student_services
            or has_fees_payment
            or has_registration
            or has_location_word
            or any(w in raw_lower for w in thanks_phrases + goodbye_phrases)
            or "help desk" in raw_lower
            or ("support" in raw_lower and has_comm)
            or ("support" in raw_lower and "office" in raw_lower)
        )
        has_generic_help = any(p in raw_lower for p in help_patterns) and not help_blocked

        # --- Ordered routing (most specific first, help last) ---
        if any(p in raw_lower for p in greeting_phrases) and "help desk" not in raw_lower:
            intent = "greeting"
            confidence = 1.0
            heuristics_matched = True
        elif any(p in raw_lower for p in thanks_phrases):
            intent = "thanks"
            confidence = 1.0
            heuristics_matched = True
        elif any(p in raw_lower for p in goodbye_phrases):
            intent = "goodbye"
            confidence = 1.0
            heuristics_matched = True
        elif (has_comm and has_office_context) or (
            has_comm and any(w in raw_lower for w in ["registrar", "help desk", "support desk", "admin"])
        ) or (
            any(w in raw_lower for w in ["office hours", "opening hours", "hours"])
            and any(w in raw_lower for w in ["registrar", "finance", "admin", "office", "department"])
        ):
            intent = "contacts"
            confidence = 1.0
            heuristics_matched = True
        elif has_student_services and "help desk" not in raw_lower:
            intent = "student_services"
            confidence = 1.0
            heuristics_matched = True
        elif has_registration or "late registration" in raw_lower:
            intent = "registration"
            confidence = 1.0
            heuristics_matched = True
        elif has_broad_scholarship and not finance_office_location:
            intent = "scholarship"
            confidence = 1.0
            heuristics_matched = True
        elif has_fees_payment and not finance_office_location and not has_broad_scholarship:
            intent = "fees"
            confidence = 1.0
            heuristics_matched = True
        elif has_location_word and has_location_target and not (has_fees_payment and "pay" in raw_lower):
            intent = "location"
            confidence = 1.0
            heuristics_matched = True
        elif has_schedule_word and not has_exam_word:
            intent = "schedule"
            confidence = 1.0
            heuristics_matched = True
        elif has_generic_help:
            intent = "help"
            confidence = 1.0
            heuristics_matched = True

        if extracted_entities and not heuristics_matched:
            # A. Fees check — before location when payment/fee keywords present
            if any(w in raw_lower for w in ["how much", "cost", "price", "tuition", "fee", "fees", "pay", "payment", "deadline"]):
                if any(k in extracted_entities for k in ["DEPARTMENT", "department"]) or extracted_entities.get("SERVICE") == "fees" or extracted_entities.get("service") == "fees":
                    intent = "fees"
                    confidence = 1.0
                    heuristics_matched = True
                elif extracted_entities.get("BUILDING") == "cafeteria" and "pay" in raw_lower:
                    intent = "fees"
                    confidence = 1.0
                    heuristics_matched = True

            # B. Registration — late registration fee is a registration penalty, not tuition
            elif "late registration" in raw_lower or (
                any(w in raw_lower for w in ["register", "registration", "enroll", "add", "drop"])
                and extracted_entities.get("SERVICE") == "registration"
            ):
                intent = "registration"
                confidence = 1.0
                heuristics_matched = True

            # C. Spatial locations check — expanded to include BUILDING (Phase 7)
            elif any(w in raw_lower for w in ["where", "location", "block", "room", "find", "map", "address", "building", "floor", "nearby", "directions"]):
                if any(k in extracted_entities for k in ["DEPARTMENT", "OFFICE", "BUILDING", "LOCATION",
                                                          "department", "office", "student_services"]):
                    intent = "location"
                    confidence = 1.0
                    heuristics_matched = True

            # D. Exams check
            elif any(w in raw_lower for w in ["exam", "exams", "examination", "test", "schedule", "timetable", "routine", "midterm", "final"]):
                if any(k in extracted_entities for k in ["DEPARTMENT", "department"]):
                    intent = "exam"
                    confidence = 1.0
                    heuristics_matched = True

            # D. Contacts check
            elif any(w in raw_lower for w in ["contact", "reach", "email", "phone", "call", "write", "number"]):
                if any(k in extracted_entities for k in ["DEPARTMENT", "OFFICE", "department", "office"]):
                    intent = "contacts"
                    confidence = 1.0
                    heuristics_matched = True

            # E. Scholarships check
            elif any(w in raw_lower for w in ["scholarship", "scholarships", "coverage", "waiver", "aid"]):
                if any(k in extracted_entities for k in ["SERVICE", "scholarship"]):
                    intent = "scholarship"
                    confidence = 1.0
                    heuristics_matched = True

            # F. Student Services / Library check
            elif any(w in raw_lower for w in ["service", "services", "counseling", "career", "library"]):
                if any(k in extracted_entities for k in ["BUILDING", "SERVICE", "student_services"]):
                    intent = "student_services"
                    confidence = 1.0
                    heuristics_matched = True

            # G. Registration check — SERVICE:registration entity present (no department required)
            elif any(w in raw_lower for w in ["register", "registration", "enroll", "enrollment", "deadline"]):
                if extracted_entities.get("SERVICE") == "registration" or extracted_entities.get("service") == "registration":
                    intent = "registration"
                    confidence = 1.0
                    heuristics_matched = True

            # H. Courses check — SERVICE:registration present with course keywords
            elif any(w in raw_lower for w in ["course", "courses", "subject", "program", "class", "module"]):
                if any(k in extracted_entities for k in ["DEPARTMENT", "department", "SERVICE"]):
                    intent = "courses"
                    confidence = 1.0
                    heuristics_matched = True

        # --- STEP 2: CONTEXT RESOLUTION & FOLLOW-UP HANDLING ---
        is_followup = self.memory.is_followup_query(raw_text_clean)
        
        if heuristics_matched:
            pass
        elif is_followup:
            inferred_intent, inferred_topic, resolved_entities, context_resolved = self.memory.resolve_context(raw_text_clean)
            
            if context_resolved:
                intent = inferred_intent
                extracted_entities.update(resolved_entities)
                context_used = True
                confidence = 0.90
            else:
                intent = "fallback"
                fallback_reason = "missing_context"
                fallback_triggered = True
        else:
            # --- STEP 3: ML PREDICTION FOR NORMAL QUERIES ---
            processed_query = preprocess_text(raw_text_clean)

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
                    
                    # Confidence Guardrail — uses Phase 7 uppercase keys + legacy lowercase keys
                    effective_threshold = config.INTENT_THRESHOLDS.get(predicted_intent, config.CONFIDENCE_THRESHOLD)
                    if extracted_entities:
                        # Lower the required threshold if we have extracted a matching domain entity
                        if predicted_intent == "location" and any(k in extracted_entities for k in [
                                "DEPARTMENT", "OFFICE", "BUILDING", "LOCATION",
                                "department", "office", "student_services"]):
                            effective_threshold = 0.25
                        elif predicted_intent == "fees" and any(k in extracted_entities for k in ["DEPARTMENT", "department"]):
                            effective_threshold = 0.25
                        elif predicted_intent == "exam" and any(k in extracted_entities for k in ["DEPARTMENT", "department"]):
                            effective_threshold = 0.25
                        elif predicted_intent == "contacts" and any(k in extracted_entities for k in [
                                "DEPARTMENT", "OFFICE", "department", "office"]):
                            effective_threshold = 0.25
                        elif predicted_intent == "scholarship" and any(k in extracted_entities for k in ["SERVICE", "scholarship"]):
                            effective_threshold = 0.25
                        elif predicted_intent == "student_services" and any(k in extracted_entities for k in [
                                "BUILDING", "SERVICE", "student_services"]):
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

        # --- STEP 3.5: LOCATION FALLBACK GUARDRAIL (Step 7.6) ---
        if intent == "location" and not fallback_triggered:
            campus_landmarks = ["stadium", "faculty building", "toilet", "restroom", "washroom"]
            has_landmark = any(lm in raw_lower for lm in campus_landmarks)
            has_location_entity = any(
                k in extracted_entities for k in ["BUILDING", "DEPARTMENT", "OFFICE", "LOCATION"]
            )
            if not has_location_entity and not has_landmark:
                intent = "fallback"
                fallback_reason = "missing_location"
                fallback_triggered = True

        # --- STEP 4: STRICT WHITELIST SCOPE VALIDATION ---
        if not fallback_triggered and (intent not in config.ALLOWED_INTENTS or intent == "fallback"):
            intent = "fallback"
            if not fallback_reason:
                fallback_reason = "out_of_domain"
            fallback_triggered = True

        # --- STEP 4.5: CONTEXT DEBUG LOGGING (Step 13) ---
        if config.CHATBOT_DEBUG and (context_used or is_followup):
            print("\n" + "-" * 40)
            print(" [CONTEXT DEBUG] MULTI-TURN DIALOGUE FLOW")
            print("-" * 40)
            print(f" Original Query     : {raw_text_clean}")
            print(f" Detected Follow-Up : {is_followup}")
            print(f" Previous Intent    : {prev_intent}")
            print(f" Resolved Intent    : {intent}")
            print(f" Topic Active       : {intent}")
            print("-" * 40 + "\n")

        # --- STEP 5: RESPONSE DISPATCHING ---
        response = get_response(
            intent, 
            extracted_entities, 
            fallback_reason, 
            query=raw_text_clean, 
            context_used=context_used,
            debug=config.CHATBOT_DEBUG
        )

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
            raw_text_clean, intent, confidence, response, fallback_triggered, is_ood=False, entities=extracted_entities_raw
        )

        # Print detected entities in debug logs format (Step 7.2)
        if config.CHATBOT_DEBUG:
            print("\n[DEBUG]")
            print(f"Intent: {intent}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Entities:\n{extracted_entities_raw}")
            print("-" * 30 + "\n")

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
        print(f"[Debug] Predicted Intent: {intent} (Confidence: {conf:.2f}) | Fallback: {fallback}")
        print("-" * 60)

if __name__ == "__main__":
    run_cli()

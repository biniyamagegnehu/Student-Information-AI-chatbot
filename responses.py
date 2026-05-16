import random
import sqlite3
import os
from datetime import datetime

# --- Context Memory System (Advanced) ---
class ChatbotMemory:
    """
    Tracks conversation context (entities, intents, and state) for robust multi-turn dialogue.
    """
    def __init__(self):
        self.last_entity = None
        self.last_intent = None
        self.last_interaction_time = None
        self.interaction_count = 0
        self.EXPIRATION_SECONDS = 300 # 5 minutes
        self.PRONOUNS = {"it", "they", "there", "that", "this", "he", "she"}
        self.FOLLOW_UP_PHRASES = {"when", "where", "how much", "tell me more", "details"}

    def resolve_context(self, text):
        """
        Replaces pronouns with the last entity and detects if the query is a follow-up.
        """
        is_follow_up = False
        
        # 1. Activity Check (Context Reset)
        if self.last_interaction_time:
            time_diff = (datetime.now() - self.last_interaction_time).total_seconds()
            if time_diff > self.EXPIRATION_SECONDS:
                self.reset()
                
        # 2. Pronoun Replacement
        words = text.lower().split()
        if any(w in self.PRONOUNS for w in words) and self.last_entity:
            resolved = [self.last_entity if w in self.PRONOUNS else w for w in words]
            text = " ".join(resolved)
            is_follow_up = True
            
        # 3. Short Query Intent Persistence
        # If user asks a very short question like "when is it?" or "how much?", 
        # we check if it's related to the previous intent.
        if len(words) <= 4 and any(w in self.FOLLOW_UP_PHRASES for w in words):
            is_follow_up = True
            
        return text, is_follow_up

    def update_memory(self, text, intent, entities):
        if entities:
            self.last_entity = entities[0][1]
        self.last_intent = intent
        self.last_interaction_time = datetime.now()
        self.interaction_count += 1

    def reset(self):
        self.last_entity = None
        self.last_intent = None
        self.last_interaction_time = None
        self.interaction_count = 0

# Global memory instance
memory = ChatbotMemory()

# --- Database Integration ---
def query_university_db(prediction, entity):
    db_path = "university_data.db"
    if not os.path.exists(db_path): return None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        table_map = {"location": "locations", "fees": "fees", "contacts": "contacts", "schedule": "schedules", "exam": "schedules"}
        if prediction in table_map:
            table = table_map[prediction]
            col = "department" if table == "fees" else "entity_name"
            cursor.execute(f"SELECT * FROM {table} WHERE {col} LIKE ?", (f"%{entity}%",))
            result = cursor.fetchone()
            if result: return result[1]
        cursor.execute("SELECT info FROM general_info WHERE topic LIKE ?", (f"%{entity}%",))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except Exception: return None

# --- Response Templates ---
STATIC_RESPONSES = {
    "greeting": ["Hello! How can I help you with campus information today?", "Hi there! I'm your Student Assistant. What's on your mind?"],
    "bye": ["Goodbye! Have a productive day!", "See you soon! Feel free to ask more later."],
    "thanks": ["You're very welcome!", "Happy to help!", "Anytime!"],
    "admission": ["Admissions are currently open for the Fall semester. You'll need your high school certificate and ID."],
    "courses": ["We offer degree programs in Computer Science, Software Engineering, and Business. Which interest you?"],
    "results": ["Semester results are posted on the official Student Portal under the 'Grades' tab."],
    "help": ["I can help you with Information about:\n- Admissions & Registration\n- Course details & Fees\n- Campus Locations & Contacts\n- Exam Schedules & Results\n\nJust ask me something like 'Where is the library?' or 'When are the exams?'"],
    "registration": ["Registration is done online through the Student Portal. Make sure you clear your fees first!"]
}

FALLBACK_RESPONSES = [
    "I'm not quite sure about that. Could you try rephrasing?",
    "I don't have enough data on that yet. Are you asking about registration, fees, or locations?",
    "I'm still learning! Could you ask that in a different way?"
]

def get_final_response(prediction, max_prob, entities, sanitized_input):
    """
    Refined decision logic with context-aware thresholding.
    """
    # 1. Context Resolution
    context_text, is_follow_up = memory.resolve_context(sanitized_input)
    
    # 2. Dynamic Confidence Thresholding
    # Lowered for better usability (0.55 for new, 0.40 for follow-ups)
    BASE_THRESHOLD = 0.55
    CONTEXT_THRESHOLD = 0.40
    
    current_threshold = CONTEXT_THRESHOLD if is_follow_up else BASE_THRESHOLD
    
    if max_prob < current_threshold:
        # If we fail confidence but have a strong last intent, try to use it
        if is_follow_up and memory.last_intent:
            prediction = memory.last_intent # Persistence
        else:
            return random.choice(FALLBACK_RESPONSES), True

    # 3. Update Memory
    memory.update_memory(context_text, prediction, entities)
    
    # 4. Specific Intent Logic (e.g. Help)
    if prediction == "help":
        return STATIC_RESPONSES["help"][0], False

    # 5. Entity/DB Logic
    entity_to_check = memory.last_entity
    if entity_to_check:
        db_info = query_university_db(prediction, entity_to_check)
        if db_info: return db_info, False
            
    # 6. Final Static Fallback
    response_list = STATIC_RESPONSES.get(prediction, FALLBACK_RESPONSES)
    return random.choice(response_list), False


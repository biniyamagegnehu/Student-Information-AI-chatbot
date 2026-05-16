import random
import sqlite3
import os
from datetime import datetime

# --- Context Memory System ---
class ChatbotMemory:
    """
    Tracks conversation context (entities and intents) for multi-turn dialogue.
    """
    def __init__(self):
        self.last_entity = None
        self.last_intent = None
        self.last_interaction_time = None
        self.EXPIRATION_SECONDS = 180 
        self.PRONOUNS = {"it", "they", "there", "that", "this", "he", "she"}

    def resolve_context(self, text):
        """
        Replaces pronouns with the last mentioned entity if context is fresh.
        """
        if self.last_interaction_time:
            time_diff = (datetime.now() - self.last_interaction_time).total_seconds()
            if time_diff > self.EXPIRATION_SECONDS:
                self.last_entity = None # Expire context
                
        words = text.split()
        if any(w in self.PRONOUNS for w in words) and self.last_entity:
            resolved = [self.last_entity if w in self.PRONOUNS else w for w in words]
            return " ".join(resolved), True
        return text, False

    def update_memory(self, text, intent, entities):
        """
        Saves the most relevant entity from the current turn.
        """
        if entities:
            # Sort to find most relevant or just take first
            self.last_entity = entities[0][1]
        self.last_intent = intent
        self.last_interaction_time = datetime.now()

# Global memory instance
memory = ChatbotMemory()

# --- Database Integration ---
def query_university_db(prediction, entity):
    """
    Retrieves dynamic data from the SQLite database.
    """
    db_path = "university_data.db"
    if not os.path.exists(db_path):
        return None
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Mapping intent labels to database tables
        table_map = {
            "location": "locations",
            "fees": "fees",
            "contacts": "contacts",
            "schedule": "schedules",
            "exam": "schedules"
        }
        
        if prediction in table_map:
            table = table_map[prediction]
            # Column naming varies by table in current schema
            col = "department" if table == "fees" else "entity_name"
            cursor.execute(f"SELECT * FROM {table} WHERE {col} LIKE ?", (f"%{entity}%",))
            result = cursor.fetchone()
            if result:
                conn.close()
                return result[1]
        
        # Fallback to general info table
        cursor.execute("SELECT info FROM general_info WHERE topic LIKE ?", (f"%{entity}%",))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except Exception:
        return None

# --- Response Templates ---
STATIC_RESPONSES = {
    "greeting": ["Hello! How can I help you with campus information today?", "Hi there! I'm your Student Assistant. What's on your mind?"],
    "bye": ["Goodbye! Have a productive day!", "See you soon! Feel free to ask more later."],
    "thanks": ["You're very welcome!", "Happy to help!", "Anytime!"],
    "admission": ["Admissions are currently open for the Fall semester. You'll need your high school certificate and ID."],
    "courses": ["We offer degree programs in Computer Science, Software Engineering, and Business. Which interest you?"],
    "results": ["Semester results are posted on the official Student Portal under the 'Grades' tab."]
}

FALLBACK_RESPONSES = [
    "I'm not quite sure about that. Could you try rephrasing?",
    "I don't have enough data on that yet. Are you asking about registration, fees, or locations?",
    "I'm still learning! Could you ask that in a different way?"
]

def get_final_response(prediction, max_prob, entities, sanitized_input):
    """
    Main logic to determine the best response for the user.
    Uses a strict confidence threshold for production stability.
    """
    # 1. Resolve Context (e.g., "where is it?")
    context_text, used_context = memory.resolve_context(sanitized_input)
    
    # 2. Threshold Check (Safety Gate)
    # Increased to 0.70 to better reject Out-of-Domain/Nonsense queries
    CONFIDENCE_THRESHOLD = 0.70
    if max_prob < CONFIDENCE_THRESHOLD:
        return random.choice(FALLBACK_RESPONSES), True

    # 3. Update Memory for future turns
    memory.update_memory(context_text, prediction, entities)
    
    # 4. Critical Entity Handling (IDs, specific names)
    for label, text in entities:
        if label == "STUDENT_ID":
            return f"I've noted Student ID {text}. How can I assist with this account specifically?", False

    # 5. Database Lookup (Priority over static)
    entity_to_check = memory.last_entity
    if entity_to_check:
        db_info = query_university_db(prediction, entity_to_check)
        if db_info:
            return db_info, False
            
    # 6. Static Response Fallback (Domain logic)
    response_list = STATIC_RESPONSES.get(prediction, FALLBACK_RESPONSES)
    return random.choice(response_list), False

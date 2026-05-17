# memory.py

class ConversationMemory:
    """
    Handles lightweight, stateful in-session memory to track conversation context across turns.
    Saves:
    - last_intent: Predicted class tag of the previous message
    - last_topic: The structural university topic inferred (e.g. registration, fees)
    - last_entities: Map of recognized entity values (e.g. {'department': 'computer science'})
    - last_response: Exact text reply returned to the user
    - last_user_query: Exact query received from the user
    """
    def __init__(self):
        self.last_intent = None
        self.last_topic = None
        self.last_entities = {}
        self.last_response = None
        self.last_user_query = None

    def update(self, intent: str, topic: str, entities: dict, response: str, query: str):
        """
        Updates the active conversation memory states.
        """
        self.last_intent = intent
        self.last_topic = topic
        # Keep old entities if none are extracted in the current turn (pronoun resolution context)
        self.last_entities = entities if entities else self.last_entities
        self.last_response = response
        self.last_user_query = query

    def clear(self):
        """
        Resets conversation memory state.
        """
        self.last_intent = None
        self.last_topic = None
        self.last_entities = {}
        self.last_response = None
        self.last_user_query = None

    def get_state(self) -> dict:
        """
        Returns a dictionary view of the current session state.
        """
        return {
            "last_intent": self.last_intent,
            "last_topic": self.last_topic,
            "last_entities": self.last_entities,
            "last_response": self.last_response,
            "last_user_query": self.last_user_query
        }

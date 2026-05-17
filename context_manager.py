# context_manager.py
"""
University Student Information Chatbot - Context Management System
Phase 6: Multi-Turn Context & Dialogue State Management Engine

This module is responsible for:
1. Maintaining stateful session conversation states.
2. Detecting ambiguous dependent follow-up student queries.
3. Overriding and resolving intents based on active contexts.
4. Keeping historical entity context alive (entity memory tracking).
5. Preventing stale context drift via automatic expiration limits.
6. Managing topic switches cleanly across different student domains.
"""

import re
import os
import config
from ner import extract_entities

MAX_CONTEXT_TURNS = 5

class ContextManager:
    """
    Step 1 & 2: Dialogue State and Context Resolution Engine.
    Tracks previous intents, discussion topics, merged entities, and handles overrides.
    """
    def __init__(self):
        self.state = {
            "last_intent": None,
            "last_topic": None,
            "last_entities": {},
            "last_response": None,
            "history": [],  # List of dicts: {"query": ..., "intent": ..., "response": ...}
            "context_turns": 0
        }

    def get_state(self) -> dict:
        """
        Returns the current dialogue memory state.
        """
        return self.state

    def update(self, intent: str, topic: str, entities: dict, response: str, query: str):
        """
        Step 7 & 14: Updates conversation memory state.
        Appends to history, tracks entity memory, and increments/resets context turns.
        """
        if not intent or intent == "fallback":
            # Do not increment turns or change active topic on fallbacks
            return

        # Step 9: Topic Switch Detection
        if topic != self.state["last_topic"]:
            self.state["context_turns"] = 0
        else:
            self.state["context_turns"] += 1

        self.state["last_intent"] = intent
        self.state["last_topic"] = topic
        
        # Step 7: Entity Memory Tracking
        if entities:
            self.state["last_entities"].update(entities)
            
        self.state["last_response"] = response

        # Step 14: Maintain lightweight history log (prevent memory overflow)
        self.state["history"].append({
            "query": query,
            "intent": intent,
            "response": response
        })
        if len(self.state["history"]) > 10:
            self.state["history"].pop(0)

    def is_followup_query(self, text: str) -> bool:
        """
        Step 4: Typo, slang, and punctuation tolerant follow-up detection.
        Matches common dependent patterns or extremely short pronominal queries.
        """
        if not text:
            return False
            
        clean_q = text.lower().strip()
        
        # Punctuation tolerant check
        clean_q = re.sub(r'[^\w\s]', '', clean_q)
        
        # Typo and slang tolerant check
        FOLLOW_UP_PATTERNS = [
            r"\b(when|where|how|how\s+much|who|can|is|are|does|do)\s+(it|they|them|that|this)\b",
            r"\b(can|how)\s+(i|do\s+i)\s+(still|apply|register|enroll|do|do\s+it|get)\b",
            r"\b(where|how)\s+(can|do)\s+i\s+(apply|see|register|find|get)\b",
            r"\b(tell|explain)\s+(me\s+)?(more|further|details)\b",
            r"\b(what\s+about|how\s+about)\b",
            r"\b(who)\s+(should\s+i|to)\s+(contact|write|email|call)\b",
            r"\b(how\s+does\s+it\s+work)\b",
            r"\b(where\s+can\s+i\s+see\s+it)\b",
            r"\b(when\s+does\s+it\s+start)\b",
            r"\b(when\s+does\s+it\s+open)\b",
            r"\b(who\s+can\s+apply)\b"
        ]
        
        for pattern in FOLLOW_UP_PATTERNS:
            if re.search(pattern, clean_q):
                return True
                
        # Extremely short dependent queries (1-3 words) matching generic pronouns or keywords
        words = clean_q.split()
        if len(words) <= 3:
            generic_pronouns = {
                "it", "they", "them", "this", "that", "there", "open", 
                "timing", "hours", "deadlines", "cost", "fees", "price"
            }
            if any(w in generic_pronouns for w in words):
                return True
                
        return False

    def resolve_context(self, query: str) -> (str, str, dict, bool):
        """
        Step 5 & 8: Resolve follow-up dialogue context or expire stale sessions.
        Prevents carrying context over goodbye/greetings.
        Returns:
            resolved_intent (str)
            resolved_topic (str)
            resolved_entities (dict)
            context_resolved (bool)
        """
        # Step 8: Context Expiration after MAX_CONTEXT_TURNS
        if not self.state["last_intent"] or self.state["context_turns"] >= MAX_CONTEXT_TURNS:
            return "fallback", "missing_context", {}, False

        # Reset context if last active intent was a greeting/goodbye (prevent carryover)
        if self.state["last_intent"] in ["greeting", "goodbye", "thanks"]:
            return "fallback", "missing_context", {}, False

        # Otherwise, reuse active previous context
        resolved_intent = self.state["last_intent"]
        resolved_topic = self.state["last_topic"]
        resolved_entities = self.state["last_entities"]

        return resolved_intent, resolved_topic, resolved_entities, True

    def clear(self):
        """
        Resets conversation memory state completely.
        """
        self.state = {
            "last_intent": None,
            "last_topic": None,
            "last_entities": {},
            "last_response": None,
            "history": [],
            "context_turns": 0
        }

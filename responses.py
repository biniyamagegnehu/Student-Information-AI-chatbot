# responses.py
import random
import json
import os
from config import INTENTS_JSON_PATH, FALLBACK_RESPONSES

# --- GLOBAL RESPONSE POOL ---
_RESPONSES = {}
_LAST_RETURNED = {} # Tracks {intent: last_response} to avoid consecutive repeats

def load_responses(filepath: str = INTENTS_JSON_PATH):
    """
    Loads responses dynamically from intents.json into a global dictionary.
    """
    global _RESPONSES
    if not os.path.exists(filepath):
        # Graceful fallback pool if the json file hasn't been created yet
        _RESPONSES = {}
        return
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for intent in data.get('intents', []):
                tag = intent.get('tag')
                responses = intent.get('responses', [])
                if tag and responses:
                    _RESPONSES[tag] = responses
    except Exception as e:
        print(f"[Warning] Failed to load responses dynamically: {e}")

# Perform initial load of responses
load_responses()

def get_response(intent: str) -> str:
    """
    Returns a random response for a given intent tag.
    Ensures that identical responses are not consecutively returned if multiple choices exist.
    If the intent is invalid, unknown, or out of scope, it triggers a polite refusal fallback.
    """
    global _LAST_RETURNED
    
    # 1. Scope / Fallback Check
    if intent not in _RESPONSES or not _RESPONSES[intent]:
        return random.choice(FALLBACK_RESPONSES)
        
    responses = _RESPONSES[intent]
    
    # 2. Return direct answer if only one option exists
    if len(responses) == 1:
        return responses[0]
        
    # 3. Prevent consecutive duplicates
    last_response = _LAST_RETURNED.get(intent)
    choices = [r for r in responses if r != last_response]
    
    # If all options are excluded somehow, fallback to complete pool
    if not choices:
        choices = responses
        
    selected = random.choice(choices)
    _LAST_RETURNED[intent] = selected
    return selected

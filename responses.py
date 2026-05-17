# responses.py
import random
import json
import os
import config
from utils import get_kb_data

# --- GLOBAL RESPONSE POOL & STATE ---
_RESPONSES = {}
_LAST_RETURNED = {} # Tracks {intent: last_response} to avoid consecutive repeats

def load_responses(filepath: str = config.INTENTS_JSON_PATH):
    """
    Loads fallback and standard response lists from intents.json.
    """
    global _RESPONSES
    if not os.path.exists(filepath):
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
        print(f"[Warning] Failed to load responses: {e}")

# Initial load
load_responses()


# --- ENTITY-AWARE INTEGRATION FACT ENGINE ---

def generate_entity_response(intent: str, entities: dict) -> str:
    """
    Looks up facts in the structured knowledge base and generates an entity-aware response.
    Returns None if no matching entity facts could be resolved (allowing generic fallback).
    """
    kb = get_kb_data()
    if not kb:
        return None

    # 1. INTENT: LOCATIONS (Dynamic Blocks, Rooms, Maps)
    if intent == "location":
        if "department" in entities:
            dept = entities["department"]
            data = kb.get("departments", {}).get(dept)
            if data:
                templates = [
                    f"The {dept.title()} department is located in {data['location']}.",
                    f"You can find the {dept.title()} department in {data['location']}. If you need to contact them, write to {data['contact']}.",
                    f"Head over to {data['location']} to find the {dept.title()} offices."
                ]
                return random.choice(templates)
            else:
                return f"I currently don't have the campus location details for the {dept.title()} department in our records."

        elif "office" in entities:
            off = entities["office"]
            data = kb.get("offices", {}).get(off)
            if data:
                templates = [
                    f"The {off.title()} office is situated on the {data['location']}.",
                    f"To visit the {off.title()} office, go to {data['location']}. Their office hours are {data['hours']}.",
                    f"You will find the {off.title()} desk in {data['location']} (Contact: {data['contact']})."
                ]
                return random.choice(templates)
            else:
                return f"I currently don't have location details for the {off.title()} office in our records."

        elif "student_services" in entities:
            serv = entities["student_services"]
            data = kb.get("student_services", {}).get(serv)
            if data:
                return f"The {serv.title()} service is located in the {data['location']}. {data['description']}"

    # 2. INTENT: TUITION & FEES
    elif intent == "fees":
        if "department" in entities:
            dept = entities["department"]
            data = kb.get("fees", {}).get(dept)
            if data:
                templates = [
                    f"The tuition fee for {dept.title()} is {data['per_semester']} per semester ({data['per_year']} per academic year).",
                    f"For the {dept.title()} program, tuition is set at {data['per_semester']} per semester, with a {data['fine_rate']}.",
                    f"Semester fees for {dept.title()} are {data['per_semester']}. Late enrollment incurs a {data['fine_rate']}."
                ]
                return random.choice(templates)
            else:
                return f"I currently do not have fee examples for the {dept.title()} program."

    # 3. INTENT: EXAMINATIONS
    elif intent == "exam":
        if "department" in entities:
            dept = entities["department"]
            date_info = kb.get("exam_schedule", {}).get(dept)
            if date_info:
                templates = [
                    f"The examination schedule for {dept.title()} is set for {date_info}.",
                    f"Exams for {dept.title()} students will run from {date_info}.",
                    f"Please note that the {dept.title()} exam block is scheduled during {date_info}."
                ]
                return random.choice(templates)
            else:
                return f"I currently don't have the final exam schedule for {dept.title()}."

    # 4. INTENT: CONTACTS DIRECTORY
    elif intent == "contacts":
        if "department" in entities:
            dept = entities["department"]
            data = kb.get("departments", {}).get(dept)
            if data:
                return f"You can reach the {dept.title()} department via email at {data['contact']}."
        elif "office" in entities:
            off = entities["office"]
            data = kb.get("offices", {}).get(off)
            if data:
                return f"You can contact the {off.title()} desk at {data['contact']}. They operate from {data['hours']}."

    # 5. INTENT: SCHOLARSHIPS
    elif intent == "scholarship":
        if "scholarship" in entities:
            sch = entities["scholarship"]
            data = kb.get("scholarships", {}).get(sch)
            if data:
                return f"The {sch.title()} offers {data['coverage']}. Requirements: {data['requirements']}. The deadline is {data['deadline']}."
            else:
                return f"I currently don't have details for the {sch.title()} in our records."

    # 6. INTENT: STUDENT SERVICES
    elif intent == "student_services":
        if "student_services" in entities:
            serv = entities["student_services"]
            data = kb.get("student_services", {}).get(serv)
            if data:
                return f"The {serv.title()} office is situated at {data['location']}. Contact them at {data['contact']} for inquiries."

    return None


# --- GENERAL RESPONSE DISPATCHER ---

def get_response(intent: str, entities: dict = None, fallback_reason: str = None) -> str:
    """
    Fetches the best response sentence for the intent.
    - If entities are present, it dynamically constructs an entity-aware reply.
    - If no entities match or are present, it draws a random whitelisted response from intents.json,
      ensuring that identical responses are not consecutively returned.
    - Handles advanced fallback reasons (out_of_domain, missing_context, low_confidence).
    """
    global _LAST_RETURNED

    # 1. Advanced Fallback Handling
    if intent == "fallback":
        reason = fallback_reason if fallback_reason in config.FALLBACK_TYPES else "out_of_domain"
        return config.FALLBACK_TYPES[reason]

    # 2. Entity-Aware Response Generation
    if entities:
        entity_resp = generate_entity_response(intent, entities)
        if entity_resp:
            # We skip consecutive repeat checks here as entity responses are highly contextual
            return entity_resp

    # 3. Standard Random Response (with repeat prevention)
    if intent not in _RESPONSES or not _RESPONSES[intent]:
        return random.choice(config.FALLBACK_RESPONSES)

    responses = _RESPONSES[intent]
    if len(responses) == 1:
        return responses[0]

    last_response = _LAST_RETURNED.get(intent)
    choices = [r for r in responses if r != last_response]
    if not choices:
        choices = responses

    selected = random.choice(choices)
    _LAST_RETURNED[intent] = selected
    return selected

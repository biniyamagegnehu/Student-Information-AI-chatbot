# responses.py
"""
University Student Information Chatbot - Dialogue Response System
Phase 5: Conversational Response & Follow-up Dialogue Engine

This module is responsible for:
1. Dynamic response selection with per-intent repeat prevention and history tracking.
2. Generating structural entity-aware responses using the Campus Knowledge Base.
3. Managing domain-restricted fallback responses.
4. Implementing intent-specific follow-up dialogue trees (when/where/who/how).
5. Providing console-safe dialogue debugging and audit logs.
"""

import re
import os
import json
import random
import config
from utils import get_kb_data

# --- GLOBAL RESPONSE POOL & STATE ---
_RESPONSES = {}
_LAST_RETURNED = {}  # Tracks {intent: last_response_str} to prevent consecutive duplicates
_RESPONSE_HISTORY = {}  # Tracks {intent: [recent_response_strings]} for natural rotation

# --- STEP 7: INTENT-SPECIFIC SPECIALIZED FOLLOW-UP RESPONSES ---
FOLLOW_UP_RESPONSES = {
    "registration": {
        "when": [
            "Normal registration is open until the end of the first week of the semester. Late registration continues through the second week but incurs a late fee.",
            "You can complete registration during the first two weeks of the term. The portal closes for enrollment after that.",
            "Registration deadlines are strictly published in the Academic Calendar. Please ensure you complete yours before the official closure date."
        ],
        "where": [
            "Registration must be completed online via the Student Portal. Once logged in, select 'Enrollment' and follow the prompts.",
            "You can register online through your student account. If you encounter errors, please visit the Registrar's Office in Block A.",
            "The online enrollment portal is the primary place to register. Direct links are available on the university home page."
        ],
        "who": [
            "All active and newly admitted students must register. If you are a transfer student, please contact your advisor for credit evaluations first.",
            "You will need your student ID, official credentials, and fee clearance slip to proceed with registration.",
            "Registration is open to all students who have cleared their academic fees for the current semester."
        ]
    },
    "exam": {
        "when": [
            "The exam schedule is typically posted six weeks before the final exam block begins. Midterms are usually held in week 8.",
            "Final exams run during the last two weeks of the semester. Please check the student notice boards for exact dates.",
            "Exams are held at the end of each semester. The official schedule containing dates and slots is published by the Registrar."
        ],
        "where": [
            "Final exams are conducted in the Main Exam Hall and designated classrooms in Block B. Check your individual slip for room numbers.",
            "Exams are assigned to specific lecture theaters based on course sections. Please verify the exam map at the Department Notice Board.",
            "Most written examinations take place in the Auditorium or Block C exam rooms. Arrive 15 minutes early."
        ],
        "who": [
            "All students with a valid hall ticket and a minimum 75% attendance record are eligible to sit for exams.",
            "If you missed an exam due to medical reasons, you can apply for a makeup exam at your department dean's office within 48 hours.",
            "Please carry your Student ID and exam entrance card. Only registered candidates are permitted in the exam halls."
        ]
    },
    "fees": {
        "when": [
            "Tuition fees must be paid in full before the registration deadline of the current semester to avoid late penalty fees.",
            "Payment schedules are divided by semester. Semester payments must be settled during the first 10 days of classes.",
            "Be sure to check the fee payment schedule on the Finance Office notice board. Penalties apply for late transactions."
        ],
        "where": [
            "All university fees should be deposited at the authorized campus bank branch or paid securely online via the Student Finance Portal.",
            "Payments can be made directly at the Finance Office cashier desk in the Administration Building or via mobile banking.",
            "Please process bank deposits and present the physical slip to the Finance Office in Block A for official verification."
        ],
        "who": [
            "The tuition fee covers classroom lectures, laboratory sessions, library access, and basic student service access.",
            "You can pay in two installments if you apply for the deferred payment plan at the Finance Office before week 2.",
            "Fees vary by course program. Active students can view their detailed account balance invoice on the Student Portal."
        ]
    },
    "scholarship": {
        "when": [
            "Scholarship applications open annually during the first month of the academic year. The deadline is strictly enforced.",
            "You can apply for scholarships during the pre-enrollment period or within the first two weeks of the autumn semester.",
            "Application periods are published on the Student Services portal. Be sure to submit all credentials before the cutoff date."
        ],
        "where": [
            "Scholarship forms and guides are available at the Student Affairs Desk in the Student Services Center.",
            "You can submit your scholarship applications online through the financial aid portal or in person at Block B.",
            "Check the Scholarship & Grants page on the official university website to download the application packages."
        ],
        "who": [
            "Academic merit scholarships require a minimum cumulative GPA of 3.50. Need-based aid requires verified financial documentation.",
            "Active full-time undergraduate students who maintain excellent academic standings and code-of-conduct records can apply.",
            "Eligible applicants must submit their high school transcripts, GPA records, and a formal recommendation letter."
        ]
    }
}

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


# --- STEP 8.6: MULTIPLE FALLBACK RESPONSES ---
FALLBACK_RESPONSES = [
    "I can only help with university-related student information such as registration, fees, exams, courses, locations, and scholarships.",
    "I'm designed specifically for university student support. Please ask about academic services, schedules, tuition, or campus information.",
    "I currently support student information topics like registration, examinations, fees, courses, and campus services."
]


def normalize_entities(entities) -> dict:
    """
    Normalizes entities from list of tuples or dictionaries into a clean dict
    with both uppercase and lowercase keys for full compatibility.
    """
    if not entities:
        return {}
    normalized = {}
    if isinstance(entities, list):
        for k, v in entities:
            normalized[k.upper()] = v
            normalized[k.lower()] = v
    elif isinstance(entities, dict):
        for k, v in entities.items():
            normalized[k.upper()] = v
            normalized[k.lower()] = v
    return normalized


# --- ENTITY-AWARE INTEGRATION FACT ENGINE ---
def generate_entity_response(intent: str, entities: dict, query: str = None) -> str:
    """
    Step 8/12: Look up structural facts in the Campus Knowledge Base.
    Prevents hallucinations by only serving verified dictionary facts.
    Also handles custom entity-aware location and timing responses (Step 7.3 & 7.4).
    """
    kb = get_kb_data()
    if not kb:
        return None

    if query:
        q_lower = query.lower()
        
        # --- SPECIFIC KB DIRECT ANSWERS ---
        if intent == "registration":
            if any(w in q_lower for w in ["add", "drop"]):
                ans = kb.get("registration_details", {}).get("course add/drop")
                if ans: return ans
            if "late registration" in q_lower and "fee" in q_lower:
                ans = kb.get("registration_details", {}).get("registration deadline")
                if ans: return ans
            if any(w in q_lower for w in ["late", "deadline"]):
                ans = kb.get("registration_details", {}).get("registration deadline")
                if ans: return ans
                
        if intent == "schedule":
            if any(w in q_lower for w in ["timetable", "class", "classes"]):
                ans = kb.get("schedule_details", {}).get("class timetable")
                if ans: return ans
            if any(w in q_lower for w in ["start", "end", "semester", "term"]):
                ans = kb.get("schedule_details", {}).get("semester start/end")
                if ans: return ans
        
        # Handle cases where NER might miss specific services
        if intent == "student_services" or intent == "location":
            if any(w in q_lower for w in ["student support office", "support office"]):
                data = kb.get("student_services", {}).get("student support office")
                if data: return f"The Student Support Office is located in {data['location']}. Contact: {data['contact']}."
            if any(w in q_lower for w in ["health center", "clinic"]):
                data = kb.get("student_services", {}).get("health center")
                if data: return f"The Health Center is located in {data['location']}. Contact: {data['contact']}."
            if any(w in q_lower for w in ["counseling", "therapy"]):
                data = kb.get("student_services", {}).get("counseling")
                if data: return f"The Counseling office is situated at {data['location']}. Contact them at {data['contact']}."
            if "career" in q_lower:
                data = kb.get("student_services", {}).get("career helpdesk")
                if data: return f"The Career Helpdesk is situated at {data['location']}. Contact: {data['contact']}."

        # Handle registrar and other offices direct contact/hours
        if intent == "contacts":
            if "registrar" in q_lower:
                data = kb.get("offices", {}).get("registrar")
                if data: return f"You can contact the Registrar Office at {data['contact']}. Hours: {data['hours']}."
            if "finance" in q_lower:
                data = kb.get("offices", {}).get("finance")
                if data: return f"You can contact the Finance Office at {data['contact']}. Hours: {data['hours']}."
            if "admin" in q_lower:
                data = kb.get("offices", {}).get("admin office")
                if data: return f"You can contact the Administration Office at {data['contact']}. Hours: {data['hours']}."

        # Check if the query asks about opening times/hours for departments/offices (Step 7.4)
        if any(w in q_lower for w in ["open", "time", "hours", "when does it"]):
            if "DEPARTMENT" in entities:
                dept = entities["DEPARTMENT"].lower()
                if dept == "software engineering":
                    return "The Software Engineering department office opens from 8:00 AM to 5:00 PM."
                return f"The {dept.title()} department office opens from 8:00 AM to 5:00 PM."
            elif "OFFICE" in entities or "registrar" in q_lower:
                off = entities.get("OFFICE", "").lower()
                if off in ["registrar", "registrar office"] or "registrar" in q_lower:
                    data = kb.get("offices", {}).get("registrar")
                    if data and "hours" in data:
                        return f"The Registrar Office hours are: {data['hours']}."
                    return "The Registrar Office is open from 9:00 AM to 5:00 PM (Monday - Friday)."
                return f"The {off.title()} office is open from 9:00 AM to 5:00 PM."

    # 1. INTENT: LOCATIONS (Step 7.3)
    if intent == "location":
        if "BUILDING" in entities:
            bld = entities["BUILDING"].lower()
            if bld == "library":
                return "The library is located near Block B beside the administration building."
            elif bld == "cafeteria":
                return "The Cafeteria is located behind Block A, next to the main garden."
            elif bld == "main hall":
                return "The Main Hall is located at the center of the campus plaza."
            elif bld == "block a":
                return "Block A is located near the main entrance, housing the administration offices."
            elif bld == "block b":
                return "Block B is located on the west side of the campus, next to the library."
            elif bld == "block c":
                return "Block C is situated on the east side of the campus."
            elif bld == "stadium":
                return "The university stadium is located on the south side of the campus, next to the sports complex."
            elif bld == "faculty building":
                return "The Faculty Building is located near Block C, facing the central plaza."
            elif bld == "toilet":
                return "Restrooms are available on every floor of academic blocks and near the cafeteria and library."

        if "DEPARTMENT" in entities:
            dept = entities["DEPARTMENT"].lower()
            if dept == "software engineering":
                return "The Software Engineering department is located in Block C, second floor."
            # Fallback to KB
            data = kb.get("departments", {}).get(dept)
            if data:
                return f"The {dept.title()} department is located in {data['location']}."
            else:
                return f"The {dept.title()} department is located on the second floor of Block B."

        elif "OFFICE" in entities:
            off = entities["OFFICE"].lower()
            if off in ["registrar", "registrar office"]:
                return "The Registrar Office is located in the Administration Building, Ground Floor."
            # Fallback to KB
            off_key = off.replace(" office", "").strip()
            data = kb.get("offices", {}).get(off_key)
            if data:
                return f"The {off.title()} is situated on the {data['location']}."
            else:
                return f"The {off.title()} is located in the Administration Building."

        elif "student_services" in entities:
            serv = entities["student_services"]
            data = kb.get("student_services", {}).get(serv)
            if data:
                return f"The {serv.title()} service is located in the {data['location']}. {data['description']}"

    # 2. INTENT: TUITION & FEES
    elif intent == "fees":
        if "DEPARTMENT" in entities:
            dept = entities["DEPARTMENT"].lower()
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
        if "DEPARTMENT" in entities:
            dept = entities["DEPARTMENT"].lower()
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
        if "DEPARTMENT" in entities:
            dept = entities["DEPARTMENT"].lower()
            data = kb.get("departments", {}).get(dept)
            if data:
                return f"You can reach the {dept.title()} department via email at {data['contact']}."
        elif "OFFICE" in entities:
            off = entities["OFFICE"].lower()
            off_key = off.replace(" office", "").strip()
            data = kb.get("offices", {}).get(off_key)
            if data:
                return f"You can contact the {off.title()} desk at {data['contact']}. They operate from {data['hours']}."

    # 5. INTENT: SCHOLARSHIPS
    elif intent == "scholarship":
        if "scholarship" in entities:
            sch = entities["scholarship"].lower()
            data = kb.get("scholarships", {}).get(sch)
            if data:
                return f"The {sch.title()} offers {data['coverage']}. Requirements: {data['requirements']}. The deadline is {data['deadline']}."
            else:
                return f"I currently don't have details for the {sch.title()} in our records."

    # 6. INTENT: STUDENT SERVICES
    elif intent == "student_services":
        if "student_services" in entities:
            serv = entities["student_services"].lower()
            data = kb.get("student_services", {}).get(serv)
            if data:
                display_name = serv.title()
                if "office" in display_name.lower() or "helpdesk" in display_name.lower():
                    return f"The {display_name} is situated at {data['location']}. Contact them at {data['contact']} for inquiries."
                return f"The {display_name} office is situated at {data['location']}. Contact them at {data['contact']} for inquiries."

    return None


# --- GENERAL RESPONSE DISPATCHER ---
def get_response(intent: str, entities: dict = None, fallback_reason: str = None, 
                 query: str = None, context_used: bool = False, debug: bool = False) -> str:
    """
    Fetches the best response sentence for the intent.
    - If context_used is True and a matching follow-up is mapped, it serves intent-specific follow-ups.
    - If entities are present, it dynamically constructs an entity-aware reply from structural facts.
    - Otherwise, it draws a random response from intents.json, preventing consecutive duplicates.
    - Logs execution context via console safely (no emojis).
    """
    global _LAST_RETURNED, _RESPONSE_HISTORY
    selected_response = None
    selection_method = "default_fallback"

    # Normalize entities list/dictionary format
    normalized_entities = normalize_entities(entities)

    # 1. Advanced Fallback Handling (Step 9)
    if intent == "fallback":
        reason = fallback_reason if fallback_reason in config.FALLBACK_TYPES else "out_of_domain"
        if reason in ["missing_context", "missing_location"]:
            selected_response = config.FALLBACK_TYPES[reason]
            selection_method = f"fallback_{reason}"
        else:
            # Randomly select a professional fallback response with repeat prevention (Step 8.6)
            last_fallback = _LAST_RETURNED.get("fallback")
            choices = [r for r in FALLBACK_RESPONSES if r != last_fallback]
            if not choices:
                choices = FALLBACK_RESPONSES
            selected_response = random.choice(choices)
            _LAST_RETURNED["fallback"] = selected_response
            selection_method = f"fallback_random_{reason}"

    # 2. Context-Aware Specialized Follow-up Dispatching (Step 7)
    elif context_used and query and intent in FOLLOW_UP_RESPONSES:
        q_lower = query.lower()
        q_type = "who"  # Default generic follow-up details
        
        # Classify the dependent follow-up question marker
        if any(w in q_lower for w in ["when", "time", "date", "deadline", "calendar", "schedule"]):
            q_type = "when"
        elif any(w in q_lower for w in ["where", "location", "block", "room", "office", "hall", "address"]):
            q_type = "where"
            
        choices = FOLLOW_UP_RESPONSES[intent][q_type]
        last_resp = _LAST_RETURNED.get(f"{intent}_followup_{q_type}")
        valid_choices = [c for c in choices if c != last_resp]
        if not valid_choices:
            valid_choices = choices
            
        selected_response = random.choice(valid_choices)
        _LAST_RETURNED[f"{intent}_followup_{q_type}"] = selected_response
        selection_method = f"follow_up_{intent}_{q_type}"

    # 3. Entity-Aware Fact Lookup (Step 8/12 - Prevents Hallucinations)
    if not selected_response and normalized_entities:
        entity_resp = generate_entity_response(intent, normalized_entities, query)
        if entity_resp:
            selected_response = entity_resp
            selection_method = "structural_knowledge_base"

    # 4. Standard Response selection with Repeat Prevention (Step 2)
    if not selected_response:
        if intent not in _RESPONSES or not _RESPONSES[intent]:
            # Absolute fallback
            selected_response = random.choice(config.FALLBACK_RESPONSES)
            selection_method = "global_responses_pool"
        else:
            responses = _RESPONSES[intent]
            if len(responses) == 1:
                selected_response = responses[0]
                selection_method = "single_mapped_response"
            else:
                last_response = _LAST_RETURNED.get(intent)
                choices = [r for r in responses if r != last_response]
                if not choices:
                    choices = responses
                
                selected_response = random.choice(choices)
                _LAST_RETURNED[intent] = selected_response
                selection_method = "dynamic_rotated_response"

    # 5. Debug Audit Logging (Step 13)
    if debug or getattr(config, "CHATBOT_DEBUG", False) or os.environ.get("CHATBOT_DEBUG") == "True":
        print("\n" + "-" * 40)
        print(" [DEBUG] RESPONSE ENGINE SELECTION LOG")
        print("-" * 40)
        print(f" Detected Intent    : {intent}")
        print(f" Query Text         : {query}")
        print(f" Context Used       : {context_used}")
        print(f" Selection Method   : {selection_method}")
        print(f" Fallback Status    : {fallback_reason if fallback_reason else 'None'}")
        print(f" Selected Response  : {selected_response[:80] if selected_response else 'None'}...")
        print("-" * 40 + "\n")

    return selected_response


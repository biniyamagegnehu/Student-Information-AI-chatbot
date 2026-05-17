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


# --- ENTITY-AWARE INTEGRATION FACT ENGINE ---
def generate_entity_response(intent: str, entities: dict) -> str:
    """
    Step 8/12: Look up structural facts in the Campus Knowledge Base.
    Prevents hallucinations by only serving verified dictionary facts.
    """
    kb = get_kb_data()
    if not kb:
        return None

    # 1. INTENT: LOCATIONS
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

    # 1. Advanced Fallback Handling (Step 9)
    if intent == "fallback":
        reason = fallback_reason if fallback_reason in config.FALLBACK_TYPES else "out_of_domain"
        selected_response = config.FALLBACK_TYPES[reason]
        selection_method = f"fallback_{reason}"

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
    if not selected_response and entities:
        entity_resp = generate_entity_response(intent, entities)
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
    if debug or os.environ.get("CHATBOT_DEBUG") == "True":
        print("\n" + "-" * 40)
        print(" [DEBUG] RESPONSE ENGINE SELECTION LOG")
        print("-" * 40)
        print(f" Detected Intent    : {intent}")
        print(f" Query Text         : {query}")
        print(f" Context Used       : {context_used}")
        print(f" Selection Method   : {selection_method}")
        print(f" Fallback Status    : {fallback_reason if fallback_reason else 'None'}")
        print(f" Selected Response  : {selected_response[:80]}...")
        print("-" * 40 + "\n")

    return selected_response

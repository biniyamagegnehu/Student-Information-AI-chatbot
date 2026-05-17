# test_ner_system.py
"""
University Student Information Chatbot - Automated NER System Tests
Phase 7: End-to-End Automated Test Suite and System Audit

This script runs comprehensive tests on the NER system:
1. Single-turn Named Entity Recognition (DEPARTMENTS, BUILDINGS, OFFICES, SERVICES, DATES, LOCATIONS).
2. Typo-tolerant entity extraction.
3. Multi-turn dialogue context and follow-up memory resolution.
4. Smart fallback handling for queries lacking entities.
5. CSV conversation logging header and record formatting audit.
"""

import os
import csv
import sys
from datetime import datetime

# Import chatbot core engine
from app import ChatbotEngine
from ner import extract_entities

def run_tests():
    print("=" * 70)
    print("  UNIVERSITY STUDENT INFORMATION CHATBOT - NER SYSTEM AUDIT SUITE")
    print("=" * 70)

    # Initialize Engine
    print("\n[INFO] Initializing ChatbotEngine...")
    engine = ChatbotEngine()
    print("[SUCCESS] ChatbotEngine loaded successfully.\n")

    # Define test cases for exact matches and typos
    ner_test_cases = [
        # 1. Department Detection
        ("where is software engineering", [('DEPARTMENT', 'software engineering')]),
        ("tell me about computer science cs", [('DEPARTMENT', 'computer science')]),
        
        # 2. Building Detection & Typo Handling
        ("wher is libary", [('BUILDING', 'library')]),
        ("how do I get to block a", [('BUILDING', 'block a')]),
        
        # 3. Office Detection
        ("where is the registrar office", [('OFFICE', 'registrar office')]),
        ("contact the finance office", [('OFFICE', 'finance office')]),
        
        # 4. Date Extraction & Services
        ("when is exam this week", [('SERVICE', 'exam'), ('DATE', 'this week')]),
        ("is there a registration tomorrow", [('SERVICE', 'registration'), ('DATE', 'tomorrow')]),
        ("fees due on mon", [('SERVICE', 'fees'), ('DATE', 'monday')])
    ]

    print("-" * 70)
    print(" 1. STANDALONE NER EXTRACTION AND TYPO AUDIT")
    print("-" * 70)

    ner_failures = []
    ner_passes = 0
    for query, expected in ner_test_cases:
        res = extract_entities(query)
        # Check if expected is subset or equal to res
        passed = True
        for item in expected:
            if item not in res:
                passed = False
                break
        
        if passed:
            print(f" PASS | Query: '{query}' => Extracted: {res}")
            ner_passes += 1
        else:
            print(f" FAIL | Query: '{query}' => Expected: {expected}, Got: {res}")
            ner_failures.append((query, expected, res))

    print(f"\n[SUMMARY] Standalone NER: {ner_passes}/{len(ner_test_cases)} Passed.")

    print("\n" + "-" * 70)
    print(" 2. MULTI-TURN CONTEXT RESOLUTION AUDIT (Step 7.4)")
    print("-" * 70)

    # Clear memory to ensure fresh start
    engine.memory.clear()

    # Query 1: Where is Software Engineering Department
    q1 = "where is the software engineering department"
    print(f"User: {q1}")
    resp1, intent1, conf1, fallback1 = engine.get_reply(q1)
    print(f"Bot : {resp1}")
    
    # Query 2: What time does it open (Pronoun Resolution to SE)
    q2 = "what time does it open"
    print(f"\nUser: {q2}")
    resp2, intent2, conf2, fallback2 = engine.get_reply(q2)
    print(f"Bot : {resp2}")

    context_passed = False
    expected_hours = "The Software Engineering department office opens from 8:00 AM to 5:00 PM."
    if resp2 == expected_hours:
        print("\n[PASS] Multi-turn context solved pronoun 'it' to 'software engineering' and fetched hours.")
        context_passed = True
    else:
        print(f"\n[FAIL] Context failed. Expected hours response, got: '{resp2}'")

    print("\n" + "-" * 70)
    print(" 3. LOCATION FALLBACK AUDIT (Step 7.6)")
    print("-" * 70)

    # Query with no location entity
    q_fallback = "where is the location"
    print(f"User: {q_fallback}")
    resp_fb, intent_fb, conf_fb, fallback_fb = engine.get_reply(q_fallback)
    print(f"Bot : {resp_fb}")

    fallback_passed = False
    expected_fallback = "I can help with university buildings, departments, offices, and student services. Which location are you asking about?"
    if fallback_fb and resp_fb == expected_fallback:
        print("\n[PASS] Low-context location request correctly triggers smart clarification fallback.")
        fallback_passed = True
    else:
        print(f"\n[FAIL] Location fallback failed. Intent: '{intent_fb}', Fallback: {fallback_fb}, Response: '{resp_fb}'")

    print("\n" + "-" * 70)
    print(" 4. CSV INTERACTION LOGGING AUDIT (Step 7.5)")
    print("-" * 70)

    csv_path = "logs/conversation_history.csv"
    csv_passed = False
    if not os.path.exists(csv_path):
        print(f"[FAIL] CSV file missing at {csv_path}")
    else:
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                expected_header = ["timestamp", "user_input", "intent", "confidence", "entities", "response", "is_fallback"]
                print(f"CSV Header found: {header}")
                if header == expected_header:
                    print("[PASS] CSV headers strictly match Phase 7 requirements.")
                    
                    # Read last row
                    rows = list(reader)
                    if rows:
                        last_row = rows[-1]
                        print(f"Last recorded interaction:\n  Timestamp   : {last_row[0]}\n  User Input  : {last_row[1]}\n  Intent      : {last_row[2]}\n  Confidence  : {last_row[3]}\n  Entities    : {last_row[4]}\n  Response    : {last_row[5]}\n  Is Fallback : {last_row[6]}")
                        csv_passed = True
                    else:
                        print("[WARNING] CSV file exists but no logs recorded.")
                else:
                    print(f"[FAIL] CSV header mismatch. Expected: {expected_header}, Got: {header}")
        except Exception as e:
            print(f"[FAIL] Error reading CSV: {e}")

    # Print Final Summary Report
    print("\n" + "=" * 70)
    print("                     NER SYSTEM AUDIT REPORT")
    print("=" * 70)
    print(f"Date & Time              : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Standalone NER Extraction: {'PASSED' if not ner_failures else 'FAILED'}")
    print(f"Multi-turn Pronoun Context: {'PASSED' if context_passed else 'FAILED'}")
    print(f"Location Fallback Handler: {'PASSED' if fallback_passed else 'FAILED'}")
    print(f"CSV Auditing Format      : {'PASSED' if csv_passed else 'FAILED'}")
    print("-" * 70)
    print("DETECTED ENTITY TYPES:")
    print("  - DEPARTMENT (e.g., computer science, software engineering)")
    print("  - BUILDING   (e.g., library, cafeteria, block a/b/c)")
    print("  - OFFICE     (e.g., registrar office, finance office)")
    print("  - SERVICE    (e.g., registration, exam, fees)")
    print("  - DATE       (e.g., today, tomorrow, this week, monday)")
    print("  - LOCATION   (e.g., campus)")
    print("-" * 70)
    print("SAMPLE CHATBOT CONVERSATIONS:")
    print("  User: where is software engineering")
    print("  Bot : The Software Engineering department is located in Block C, second floor.")
    print("  ---")
    print("  User: what time does it open")
    print("  Bot : The Software Engineering department office opens from 8:00 AM to 5:00 PM.")
    print("  ---")
    print("  User: wher is libary")
    print("  Bot : The library is located near Block B beside the administration building.")
    print("=" * 70 + "\n")

    if not ner_failures and context_passed and fallback_passed and csv_passed:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()

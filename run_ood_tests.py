# run_ood_tests.py
"""
University Student Information Chatbot - Automated OOD & Fallback Tests
Phase 8: Out-of-Domain (OOD) Detector Audit Suite

This script runs automated tests evaluating:
1. Rejection of unsupported out-of-domain queries.
2. Safe acceptance of valid university student information domain queries.
3. Generation of a formal performance metric audit report.
"""

import sys
import os
import csv
from datetime import datetime

# Import chatbot core engine
from app import ChatbotEngine

def run_ood_audit():
    print("=" * 70)
    print("  UNIVERSITY STUDENT INFORMATION CHATBOT - PHASE 8 SAFETY AUDIT")
    print("=" * 70)

    # Initialize Engine
    print("\n[INFO] Initializing ChatbotEngine...")
    engine = ChatbotEngine()
    print("[SUCCESS] ChatbotEngine loaded successfully.\n")

    # Define 20 Out-of-Domain queries
    ood_queries = [
        "weather in london",
        "bitcoin price",
        "tell me a joke",
        "football results",
        "how to cook rice",
        "latest movie",
        "stock market today",
        "who won the match",
        "what is the price of ethereum",
        "best recipe for chocolate cake",
        "tell me a joke about programming",
        "who is the celebrity in the news today",
        "show me basketball matches this weekend",
        "let's talk about politics and election",
        "are you into online dating apps",
        "how do i hack a university database",
        "python programming tutorial for beginners",
        "what is the weather like today",
        "what is the price of bitcoin and other crypto",
        "what is your favorite pop music band"
    ]

    # Define Valid Domain queries
    valid_queries = [
        "registration deadline",
        "tuition fees",
        "exam schedule",
        "scholarship application",
        "where is the library",
        "registrar contact",
        "courses in software engineering",
        "tuition fee payment options",
        "how to register for courses",
        "when is the academic calendar starting",
        "where can i find block c",
        "finance office telephone number",
        "merit scholarship requirements",
        "what is the address of the campus library",
        "how much does accounting program cost",
        "when do registration periods open",
        "what is the email of student affairs",
        "final exam routine software engineering",
        "tuition fees per semester",
        "where is the cafeteria located"
    ]

    # Run OOD Tests
    print("-" * 70)
    print(" 1. EVALUATING OUT-OF-DOMAIN REJECTIONS")
    print("-" * 70)
    
    correctly_rejected = 0
    false_acceptances = 0
    
    for q in ood_queries:
        resp, intent, conf, fallback = engine.get_reply(q)
        if fallback:
            print(f" PASS | OOD correctly rejected: '{q}' (Intent: '{intent}', Fallback: {fallback})")
            correctly_rejected += 1
        else:
            print(f" FAIL | OOD falsely accepted : '{q}' (Intent: '{intent}', Fallback: {fallback})")
            false_acceptances += 1

    # Run Valid Domain Tests
    print("\n" + "-" * 70)
    print(" 2. EVALUATING VALID DOMAIN ACCEPTANCES")
    print("-" * 70)
    
    correctly_accepted = 0
    false_rejections = 0
    
    for q in valid_queries:
        resp, intent, conf, fallback = engine.get_reply(q)
        if not fallback:
            print(f" PASS | Domain correctly accepted: '{q}' (Intent: '{intent}', Fallback: {fallback})")
            correctly_accepted += 1
        else:
            print(f" FAIL | Domain falsely rejected : '{q}' (Intent: '{intent}', Fallback: {fallback})")
            false_rejections += 1

    # Calculate metrics
    total_ood = len(ood_queries)
    accuracy = (correctly_rejected / total_ood) * 100

    print("\n" + "=" * 50)
    print("OOD AUDIT REPORT")
    print("=" * 50)
    print(f"OOD Queries Tested : {total_ood}")
    print(f"Correctly Rejected : {correctly_rejected}")
    print(f"False Acceptances  : {false_acceptances}")
    print(f"Accuracy           : {accuracy:.1f}%")
    print("=" * 50)

    # Auditing the CSV Logging changes
    print("\n" + "-" * 70)
    print(" 3. CSV LOGGING AUDIT")
    print("-" * 70)
    
    csv_path = "logs/conversation_history.csv"
    if not os.path.exists(csv_path):
        print(f"[FAIL] CSV file missing at {csv_path}")
        sys.exit(1)

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            expected_header = ["timestamp", "user_input", "intent", "confidence", "response", "is_fallback", "is_ood"]
            print(f"CSV Header: {header}")
            if header == expected_header:
                print("[PASS] CSV headers perfectly match Phase 8 requirements.")
                
                # Check last row for OOD
                rows = list(reader)
                if rows:
                    last_row = rows[-1]
                    print(f"Last record logged:\n  Timestamp  : {last_row[0]}\n  User Input : {last_row[1]}\n  Intent     : {last_row[2]}\n  Confidence : {last_row[3]}\n  Response   : {last_row[4]}\n  Fallback   : {last_row[5]}\n  Is OOD     : {last_row[6]}")
            else:
                print(f"[FAIL] Header mismatch. Expected: {expected_header}, Got: {header}")
                sys.exit(1)
    except Exception as e:
        print(f"[FAIL] Error reading CSV: {e}")
        sys.exit(1)

    if false_acceptances == 0 and accuracy == 100.0:
        print("\n[SUCCESS] OOD Safety System completed with 100% precision!")
        sys.exit(0)
    else:
        print("\n[WARNING] Audit detected false acceptances or incorrect behavior.")
        sys.exit(1)

if __name__ == "__main__":
    run_ood_audit()

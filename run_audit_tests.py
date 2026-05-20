import os
import time
import config
from app import ChatbotEngine

def run_automated_test_suite():
    print("\n" + "="*80)
    print(" CHATBOT FUNCTIONAL AUDIT: AUTOMATED TEST SUITE")
    print("="*80)

    # Initialize Engine
    engine = ChatbotEngine()
    if not engine.is_loaded:
        print(" CRITICAL ERROR: Model files missing. Run train.py first.")
        return

    # Define test cases: Category -> List of (Query, Expected Intent)
    test_categories = {
        "1. Greetings": [
            ("hi", "greeting"),
            ("hello", "greeting"),
            ("hey", "greeting"),
            ("good morning", "greeting"),
            ("yo", "greeting")
        ],
        "2. Help Queries": [
            ("i need help", "help"),
            ("assist me please", "help")
        ],
        "3. Registration Queries": [
            ("registration deadline", "registration"),
            ("how to apply", "registration")
        ],
        "4. Course/Program Queries": [
            ("software engineering", "courses"),
            ("what programs do you offer", "courses")
        ],
        "5. Fee Queries": [
            ("tuition fee", "fees"),
            ("how much is payment", "fees")
        ],
        "6. Exam Queries": [
            ("exam schedule", "exam"),
            ("midterm dates", "exam")
        ],
        "7. Schedule Queries": [
            ("class timetable", "schedule"),
            ("when does semester start", "schedule")
        ],
        "8. Scholarship Queries": [
            ("financial aid", "scholarship"),
            ("scholarship application", "scholarship")
        ],
        "9. Location Queries": [
            ("where is library", "location"),
            ("admin office location", "location")
        ],
        "10. Contact Queries": [
            ("registrar phone number", "contacts"),
            ("email of cs dept", "contacts")
        ],
        "11. Student Services Queries": [
            ("health center", "student_services"),
            ("counseling services", "student_services")
        ],
        "12. Thanks/Goodbye": [
            ("thank you", "thanks"),
            ("bye", "goodbye")
        ],
        "13. Typo Handling": [
            ("regstration", "registration"),
            ("exam scheduel", "exam"),
            ("cources available", "courses"),
            ("wher is library", "location"),
            ("hosstel room", "fallback") # hostel not in domain, triggers fallback or OOD
        ],
        "14. Slang Handling": [
            ("yo when registration open", "registration"),
            ("bro where cs dept at", "location"),
            ("can u help me with fees", "fees"),
            ("where admin office pls", "location")
        ],
        "17. OOD Rejection": [
            ("bitcoin price today", "fallback"),
            ("weather in london", "fallback"),
            ("who won football match", "fallback"),
            ("tell me a joke", "fallback"),
            ("how to cook rice", "fallback")
        ],
        "18. Stress Testing": [
            ("aaaaaaaaaaaaaaaaa", "fallback"),
            ("??", "fallback"),
            ("yo bro regstration pls", "registration"),
            ("sdfghjkl", "fallback"),
            ("", "fallback") # Will return none, fallback
        ]
    }

    # Context & Follow-up Test definitions
    context_tests = [
        # Sequence 1
        ("tell me about registration", "registration", True),
        ("when is it", "registration", False),
        # Sequence 2
        ("where is the library", "location", True),
        ("when does it open", "location", False),
        # Sequence 3
        ("tell me about exams", "exam", True),
        ("what time are they", "exam", False)
    ]

    failures = []
    total_passed = 0
    total_tests = 0
    confidences = []

    print(f"{'INPUT QUERY':<35} | {'EXPECTED':<12} | {'PREDICTED':<12} | {'RESULT'}")
    print("-" * 80)

    # Run Standard Tests
    for category, tests in test_categories.items():
        for query, expected in tests:
            # Clear memory to isolate test
            engine.memory.clear()
            
            response, intent, conf, fallback = engine.get_reply(query)
            
            # Map None to fallback for empty inputs
            if intent == "none":
                intent = "fallback"

            if intent == expected:
                result = "PASS"
                total_passed += 1
            elif expected == "fallback" and fallback:
                result = "PASS"
                total_passed += 1
            else:
                result = "FAIL"
                failures.append({
                    "query": query,
                    "expected": expected,
                    "predicted": intent,
                    "confidence": conf,
                    "fallback": fallback
                })
            
            total_tests += 1
            confidences.append(conf)
            print(f"{query[:35]:<35} | {expected:<12} | {intent:<12} | {result}")

    # Run Context & Follow-up Tests
    print("\n--- Running Context Memory Tests ---")
    for query, expected, clear_memory in context_tests:
        if clear_memory:
            engine.memory.clear()
            
        response, intent, conf, fallback = engine.get_reply(query)
        
        if intent == expected:
            result = "PASS"
            total_passed += 1
        else:
            result = "FAIL"
            failures.append({
                "query": f"[Context] {query}",
                "expected": expected,
                "predicted": intent,
                "confidence": conf,
                "fallback": fallback
            })
            
        total_tests += 1
        confidences.append(conf)
        print(f"{query[:35]:<35} | {expected:<12} | {intent:<12} | {result}")

    # Calculations
    success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    print("-" * 80)
    print(f"OVERALL TEST SUCCESS RATE: {success_rate:.1f}% ({total_passed}/{total_tests})")
    print(f"AVERAGE CONFIDENCE SCORE:  {avg_conf:.2f}")
    print("="*80)

    # Failure Reporting
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    log_path = "logs/audit_failures.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"--- AUTOMATED AUDIT FAILURES [{time.strftime('%Y-%m-%d %H:%M:%S')}] ---\n")
        f.write(f"Total Failures: {len(failures)}\n\n")
        for fail in failures:
            f.write(f"Query: {fail['query']}\n")
            f.write(f"Expected: {fail['expected']}\n")
            f.write(f"Predicted: {fail['predicted']}\n")
            f.write(f"Confidence: {fail['confidence']:.2f}\n")
            f.write(f"Fallback Triggered: {fail['fallback']}\n")
            f.write("-" * 40 + "\n")

    print(f"\n[INFO] Failure report saved to {log_path}")
    print("[INFO] Stress test completed. No runtime crashes occurred.")

if __name__ == "__main__":
    run_automated_test_suite()

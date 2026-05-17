# run_final_audit.py
"""
University Student Information Chatbot - Phase 9: Master End-to-End Audit Suite

This script runs all validation phases in sequence and produces a unified
production-readiness report covering:

  1. NER Entity Extraction Accuracy (Phase 7)
  2. OOD Safety Rejection Rate (Phase 8)
  3. Confidence Threshold & Fallback Behavior
  4. Multi-Turn Context Resolution
  5. CSV Log Schema Validation

Exit code 0 = ALL CLEAR. Exit code 1 = AUDIT FAILURE.
"""

import os
import sys
import csv
from datetime import datetime

from ner import extract_entities
from app import ChatbotEngine

# ---------------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------------

def header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def section(title: str):
    print("\n" + "-" * 70)
    print(f" {title}")
    print("-" * 70)

def result(passed: bool, label: str):
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {label}")
    return passed


# ---------------------------------------------------------------------------
# PHASE 7 — NER ACCURACY
# ---------------------------------------------------------------------------

def audit_ner():
    section("PHASE 7 — NER ENTITY EXTRACTION & TYPO TOLERANCE")
    tests = [
        # (query, required_entity_tuples_must_be_a_subset)
        ("where is software engineering",      [("DEPARTMENT", "software engineering")]),
        ("wher is libary",                     [("BUILDING",    "library")]),
        ("when is exam this week",             [("SERVICE",     "exam"), ("DATE", "this week")]),
        ("where is the registrar office",      [("OFFICE",      "registrar office")]),
        ("contact the finance office",         [("OFFICE",      "finance office")]),
        ("fees due on mon",                    [("SERVICE",     "fees"), ("DATE", "monday")]),
        ("is there a registration tomorrow",   [("SERVICE",     "registration"), ("DATE", "tomorrow")]),
        ("tell me about cs department",        [("DEPARTMENT",  "computer science")]),
        ("where is block c",                   [("BUILDING",    "block c")]),
        ("scholarship application deadline",   [("SERVICE",     "scholarship")]),
    ]

    passed = total = 0
    for query, required in tests:
        total += 1
        got = extract_entities(query)
        ok = all(item in got for item in required)
        if result(ok, f"'{query}' => {got}"):
            passed += 1

    print(f"\n  NER Score: {passed}/{total}")
    return passed, total


# ---------------------------------------------------------------------------
# PHASE 8 — OOD SAFETY
# ---------------------------------------------------------------------------

def audit_ood(engine: ChatbotEngine):
    section("PHASE 8 — OUT-OF-DOMAIN REJECTION")
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
        "celebrity news today",
        "let's talk about politics",
        "online dating apps",
        "how do i hack a database",
        "python programming tutorial",
        "what is the weather like today",
        "crypto prices",
        "pop music band",
        "basketball highlights",
        "i need a joke about exams",
    ]

    passed = total = 0
    for q in ood_queries:
        total += 1
        _, _, _, is_fallback = engine.get_reply(q)
        if result(is_fallback, f"OOD rejected: '{q}'"):
            passed += 1

    print(f"\n  OOD Rejection Score: {passed}/{total}")
    return passed, total


# ---------------------------------------------------------------------------
# PHASE 8 — VALID DOMAIN ACCEPTANCE
# ---------------------------------------------------------------------------

def audit_domain(engine: ChatbotEngine):
    section("PHASE 8 — VALID DOMAIN ACCEPTANCE")
    valid_queries = [
        "registration deadline",
        "tuition fees",
        "exam schedule",
        "scholarship application",
        "where is the library",
        "registrar contact",
        "courses in software engineering",
        "how to register for courses",
        "finance office telephone number",
        "merit scholarship requirements",
        "what is the email of student affairs",
        "final exam routine software engineering",
    ]

    passed = total = 0
    for q in valid_queries:
        total += 1
        _, intent, _, is_fallback = engine.get_reply(q)
        if result(not is_fallback, f"Accepted: '{q}' (intent={intent})"):
            passed += 1

    print(f"\n  Domain Acceptance Score: {passed}/{total}")
    return passed, total


# ---------------------------------------------------------------------------
# PHASE 7.4 — MULTI-TURN CONTEXT RESOLUTION
# ---------------------------------------------------------------------------

def audit_context(engine: ChatbotEngine):
    section("PHASE 7.4 — MULTI-TURN CONTEXT RESOLUTION")

    engine.memory.clear()

    q1 = "where is the software engineering department"
    r1, _, _, _ = engine.get_reply(q1)
    print(f"  User: {q1}")
    print(f"  Bot : {r1}")

    q2 = "what time does it open"
    r2, _, _, _ = engine.get_reply(q2)
    print(f"\n  User: {q2}")
    print(f"  Bot : {r2}")

    expected = "The Software Engineering department office opens from 8:00 AM to 5:00 PM."
    ok = (r2 == expected)
    result(ok, f"Pronoun 'it' resolved to 'Software Engineering' => '{r2}'")
    return (1 if ok else 0), 1


# ---------------------------------------------------------------------------
# PHASE 7.5/8.7 — CSV LOG SCHEMA
# ---------------------------------------------------------------------------

def audit_csv():
    section("CSV LOGGING SCHEMA (Phase 8 Format)")
    path = "logs/conversation_history.csv"
    expected = ["timestamp", "user_input", "intent", "confidence", "entities", "response", "is_fallback", "is_ood"]
    if not os.path.exists(path):
        result(False, f"CSV file not found at {path}")
        return 0, 1

    try:
        with open(path, "r", encoding="utf-8") as f:
            header_row = next(csv.reader(f), None)
        ok = (header_row == expected)
        result(ok, f"CSV header: {header_row}")
        return (1 if ok else 0), 1
    except Exception as e:
        result(False, f"CSV read error: {e}")
        return 0, 1


# ---------------------------------------------------------------------------
# MASTER RUNNER
# ---------------------------------------------------------------------------

def run_final_audit():
    header("UNIVERSITY STUDENT CHATBOT — PHASE 9 FINAL AUDIT REPORT")
    print(f"  Date & Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python Path : {sys.executable}")

    engine = ChatbotEngine()

    scores = {}

    p, t = audit_ner()
    scores["NER Extraction (Phase 7)"] = (p, t)

    p, t = audit_ood(engine)
    scores["OOD Rejection (Phase 8)"] = (p, t)

    p, t = audit_domain(engine)
    scores["Domain Acceptance (Phase 8)"] = (p, t)

    p, t = audit_context(engine)
    scores["Multi-Turn Context (Phase 7.4)"] = (p, t)

    p, t = audit_csv()
    scores["CSV Schema (Phase 8.7)"] = (p, t)

    # Final scorecard
    header("FINAL AUDIT SCORECARD")
    total_pass = total_tests = 0
    all_pass = True
    for category, (p, t) in scores.items():
        pct = (p / t * 100) if t else 0
        status = "PASS" if p == t else "FAIL"
        if p != t:
            all_pass = False
        print(f"  {status:4}  {category:35}  {p:2}/{t:2}  ({pct:.0f}%)")
        total_pass += p
        total_tests += t

    overall_pct = (total_pass / total_tests * 100) if total_tests else 0
    print(f"\n  {'-' * 60}")
    print(f"  OVERALL : {total_pass}/{total_tests} tests passed  ({overall_pct:.1f}%)")

    if all_pass:
        print("\n  [SUCCESS] Chatbot is PRODUCTION-READY. All audits passed.")
        sys.exit(0)
    else:
        print("\n  [WARNING] Some audit checks failed. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    run_final_audit()

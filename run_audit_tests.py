import joblib
import os
import sys
from preprocess import preprocess_text, extract_all_entities

# Load model and vectorizer
MODEL_PATH = "model/model.pkl"
VEC_PATH = "model/vectorizer.pkl"

def run_automated_test_suite():
    print("\n" + "="*50)
    print(" [BOT] CHATBOT FUNCTIONAL AUDIT: AUTOMATED TEST SUITE")
    print("="*50)

    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found. Please run train.py first.")
        return

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)

    # Test Cases: (Input, Expected Intent)
    test_cases = [
        ("how to apply for admission", "admission"),
        ("when is the registration deadline", "registration"),
        ("what engineering courses do you have", "courses"),
        ("show me the class routine", "schedule"),
        ("when is the software engineering exam", "exam"),
        ("how much is the fee for pharmacy", "fees"),
        ("where is block c located", "location"),
        ("help me what can you do", "help"),
        ("thanks for the help", "thanks"),
        ("banana price today", "unknown") # Testing for low confidence/unknown
    ]

    passed = 0
    total = len(test_cases)

    print(f"{'INPUT QUERY':<35} | {'EXPECTED':<12} | {'PREDICTED':<12} | {'RESULT'}")
    print("-" * 85)

    for query, expected in test_cases:
        # Preprocess
        clean = preprocess_text(query)
        vec = vectorizer.transform([clean])
        
        # Predict
        probs = model.predict_proba(vec)[0]
        max_prob = max(probs)
        pred = model.classes_[probs.argmax()]
        
        # Handle OOD
        if expected == "unknown":
            result = "PASS" if max_prob < 0.70 else "FAIL (False Positive)"
            pred_display = f"{pred} ({max_prob:.2f})"
        else:
            is_correct = (pred == expected) and (max_prob >= 0.60)
            result = "PASS" if is_correct else "FAIL"
            pred_display = f"{pred} ({max_prob:.2f})"

        if "PASS" in result: passed += 1
        print(f"{query[:35]:<35} | {expected:<12} | {pred_display:<12} | {result}")

    print("-" * 85)
    print(f"OVERALL PERFORMANCE: {passed}/{total} ({(passed/total)*100:.1f}%)")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_automated_test_suite()


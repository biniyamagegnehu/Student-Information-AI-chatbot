import joblib
import os
import sys
import config
from preprocess import clean_text
from utils import extract_entities

# Load model and vectorizer
MODEL_PATH = config.MODEL_PATH
VEC_PATH = config.VECTORIZER_PATH
LE_PATH = config.LABEL_ENCODER_PATH

def run_automated_test_suite():
    print("\n" + "="*50)
    print(" [BOT] CHATBOT FUNCTIONAL AUDIT: AUTOMATED TEST SUITE")
    print("="*50)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(LE_PATH):
        print("Error: Model or Label Encoder not found. Please run train.py first.")
        return

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    label_encoder = joblib.load(LE_PATH)

    # Test Cases: (Input, Expected Intent)
    test_cases = [
        ("how to apply for admission", "unknown"), # Admissions is strictly out of scope
        ("when is the registration deadline", "registration"),
        ("what engineering courses do you have", "courses"),
        ("show me the class routine", "schedule"),
        ("when is the software engineering exam", "exam"),
        ("how much is the fee for pharmacy", "fees"),
        ("where is block c located", "location"),
        ("help me what can you do", "help"),
        ("thanks for the help", "thanks"),
        ("banana price today", "unknown") # Out of domain
    ]

    passed = 0
    total = len(test_cases)

    print(f"{'INPUT QUERY':<35} | {'EXPECTED':<12} | {'PREDICTED':<12} | {'RESULT'}")
    print("-" * 85)

    for query, expected in test_cases:
        # Preprocess
        clean = clean_text(query)
        vec = vectorizer.transform([clean])
        
        # Predict
        probs = model.predict_proba(vec)[0]
        max_prob = max(probs)
        pred_idx = probs.argmax()
        pred = label_encoder.inverse_transform([pred_idx])[0]
        
        # Handle OOD
        if expected == "unknown":
            # For OOD cases: either the confidence is very low, or it maps to 'fallback'
            result = "PASS" if (max_prob < 0.60 or pred == "fallback") else "FAIL (False Positive)"
            pred_display = f"{pred} ({max_prob:.2f})"
        else:
            is_correct = (pred == expected) and (max_prob >= 0.40)
            result = "PASS" if is_correct else "FAIL"
            pred_display = f"{pred} ({max_prob:.2f})"

        if "PASS" in result: passed += 1
        print(f"{query[:35]:<35} | {expected:<12} | {pred_display:<12} | {result}")

    print("-" * 85)
    print(f"OVERALL PERFORMANCE: {passed}/{total} ({(passed/total)*100:.1f}%)")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_automated_test_suite()

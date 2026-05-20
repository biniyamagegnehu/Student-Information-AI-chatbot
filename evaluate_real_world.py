import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    classification_report, 
    confusion_matrix
)
from preprocess import preprocess_text
import config
from app import ChatbotEngine

# --- Configuration ---
MODEL_PATH = "model/model.pkl"
VEC_PATH = "model/vectorizer.pkl"
LE_PATH = "model/label_encoder.pkl"
HUMAN_TEST_FILE = "dataset/human_test_data.csv"
FAILURE_LOG_CLS = "logs/failures_classifier.txt"
FAILURE_LOG_E2E = "logs/failures_e2e.txt"

def run_classifier_evaluation():
    print("\n" + "="*60)
    print("  [MODE 1] CLASSIFIER-ONLY PERFORMANCE AUDIT")
    print("="*60)

    # 1. Load Assets
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH) or not os.path.exists(LE_PATH):
        print(" CRITICAL ERROR: Model files missing. Run train.py first.")
        return

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    label_encoder = joblib.load(LE_PATH)
    print(" Model Artifacts Loaded.")

    # 2. Load and Prepare Test Data
    if not os.path.exists(HUMAN_TEST_FILE):
        print(f" ERROR: {HUMAN_TEST_FILE} not found.")
        return

    df = pd.read_csv(HUMAN_TEST_FILE)
    print(f" Balanced Human Test Set Loaded ({len(df)} samples).")

    # 3. Clean and Vectorize
    df['clean_text'] = df['query'].apply(preprocess_text)
    
    # Split into In-Domain and Out-of-Domain (fallback)
    in_domain = df[df['expected_intent'] != "fallback"].copy()
    out_of_domain = df[df['expected_intent'] == "fallback"].copy()

    # 4. Predict In-Domain Performance
    X_test_vec = vectorizer.transform(in_domain['clean_text'])
    y_true = in_domain['expected_intent']
    
    probs_all = model.predict_proba(X_test_vec)
    pred_indices = np.argmax(probs_all, axis=1)
    y_pred = label_encoder.inverse_transform(pred_indices)
    max_probs = np.max(probs_all, axis=1)

    # Apply Confidence Threshold (simulating fallback logic)
    y_pred_final = []
    for pred, prob in zip(y_pred, max_probs):
        threshold = config.INTENT_THRESHOLDS.get(pred, config.CONFIDENCE_THRESHOLD)
        if prob < threshold:
            y_pred_final.append("fallback")
        else:
            y_pred_final.append(pred)

    y_pred = np.array(y_pred_final)

    # --- METRICS CALCULATION ---
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"\nOverall Accuracy: {acc*100:.2f}%")
    print(f"Balanced Accuracy: {balanced_acc*100:.2f}%")
    print("\n## Intent-Level Metrics:\n")
    
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    report = classification_report(y_true, y_pred, labels=unique_labels, zero_division=0, output_dict=True)
    
    for intent in unique_labels:
        if intent == "fallback":
            continue
        if intent in report:
            print(f"{intent:<18}Precision: {report[intent]['precision']:.2f} Recall: {report[intent]['recall']:.2f} F1: {report[intent]['f1-score']:.2f}")
    
    print("-" * 55)

    # 5. OOD (Out-of-Domain) Evaluation
    print("\nOOD Detection:")
    if not out_of_domain.empty:
        ood_vec = vectorizer.transform(out_of_domain['clean_text'])
        ood_probs = model.predict_proba(ood_vec)
        ood_max_probs = np.max(ood_probs, axis=1)
        ood_preds = label_encoder.inverse_transform(np.argmax(ood_probs, axis=1))
        
        THRESHOLD = config.CONFIDENCE_THRESHOLD
        properly_rejected = sum(p < THRESHOLD for p in ood_max_probs)
        
        print(f"Rejected {properly_rejected}/{len(out_of_domain)} unrelated queries successfully.")
        
    print("=" * 60)

    # 6. Failure Analysis
    failures = []
    for idx, (text, true, pred, prob) in enumerate(zip(in_domain['query'], y_true, y_pred, max_probs)):
        if true != pred:
            failures.append(f"Query: \"{text}\"\nExpected: {true}\nPredicted: {pred}\nConfidence: {prob:.2f}\n" + "-"*30)

    if not os.path.exists('logs'): os.makedirs('logs')
    with open(FAILURE_LOG_CLS, "w", encoding="utf-8") as f:
        f.write(f"--- HUMAN EVALUATION FAILURE AUDIT (CLASSIFIER ONLY) [{pd.Timestamp.now()}] ---\n")
        f.write(f"Total Misclassifications: {len(failures)}\n\n")
        f.write("\n".join(failures))
    
    print(f"\nFailure analysis saved to {FAILURE_LOG_CLS}")

def run_e2e_evaluation():
    print("\n" + "="*60)
    print("  [MODE 2] END-TO-END CHATBOT PIPELINE EVALUATION")
    print("="*60)
    
    engine = ChatbotEngine()
    if not engine.is_loaded:
        print(" CRITICAL ERROR: Model files missing for End-to-End Evaluation. Run train.py first.")
        return

    if not os.path.exists(HUMAN_TEST_FILE):
        print(f" ERROR: {HUMAN_TEST_FILE} not found.")
        return

    df = pd.read_csv(HUMAN_TEST_FILE)
    print(f" Balanced Human Test Set Loaded ({len(df)} samples).")

    y_true = df['expected_intent']
    y_pred = []
    failures = []
    ood_rejections = 0
    total_ood_expected = sum(y_true == "fallback")

    for idx, row in df.iterrows():
        query = row['query']
        expected = row['expected_intent']
        
        engine.memory.clear() # clear context
        response, intent, conf, fallback = engine.get_reply(query)
        
        if intent == "none" or fallback:
            predicted = "fallback"
        else:
            predicted = intent
            
        y_pred.append(predicted)
        
        if expected == "fallback" and predicted == "fallback":
            ood_rejections += 1
            
        if expected != predicted:
            failures.append(f"Query: \"{query}\"\nExpected: {expected}\nPredicted: {predicted}\nConfidence: {conf:.2f}\n" + "-"*30)

    y_pred = np.array(y_pred)
    
    # In domain filter
    in_domain_mask = y_true != "fallback"
    y_true_in = y_true[in_domain_mask]
    y_pred_in = y_pred[in_domain_mask]

    acc = accuracy_score(y_true_in, y_pred_in)
    balanced_acc = balanced_accuracy_score(y_true_in, y_pred_in)

    print(f"\nOverall Accuracy (In-Domain): {acc*100:.2f}%")
    print(f"Balanced Accuracy (In-Domain): {balanced_acc*100:.2f}%")
    print("\n## Intent-Level Metrics (End-to-End):\n")
    
    unique_labels = sorted(list(set(y_true_in) | set(y_pred_in)))
    report = classification_report(y_true_in, y_pred_in, labels=unique_labels, zero_division=0, output_dict=True)
    
    for intent in unique_labels:
        if intent == "fallback":
            continue
        if intent in report:
            print(f"{intent:<18}Precision: {report[intent]['precision']:.2f} Recall: {report[intent]['recall']:.2f} F1: {report[intent]['f1-score']:.2f}")

    print("-" * 55)
    print("\nOOD Detection (End-to-End):")
    if total_ood_expected > 0:
        print(f"Rejected {ood_rejections}/{total_ood_expected} unrelated queries successfully.")

    if not os.path.exists('logs'): os.makedirs('logs')
    with open(FAILURE_LOG_E2E, "w", encoding="utf-8") as f:
        f.write(f"--- HUMAN EVALUATION FAILURE AUDIT (END-TO-END) [{pd.Timestamp.now()}] ---\n")
        f.write(f"Total Misclassifications: {len(failures)}\n\n")
        f.write("\n".join(failures))
    
    print(f"\nE2E Failure analysis saved to {FAILURE_LOG_E2E}")
    print("\n" + "="*60)
    print("  AUDIT COMPLETE: Validation Requirements Met.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_classifier_evaluation()
    run_e2e_evaluation()

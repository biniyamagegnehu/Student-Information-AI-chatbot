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

# --- Configuration ---
MODEL_PATH = "model/model.pkl"
VEC_PATH = "model/vectorizer.pkl"
LE_PATH = "model/label_encoder.pkl"
HUMAN_TEST_FILE = "dataset/human_test.csv"
FAILURE_LOG = "logs/failures.txt"

def run_evaluation():
    print("\n" + "="*60)
    print("  PROFESSIONAL CHATBOT PERFORMANCE AUDIT (HUMAN DATA)")
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

    # Map 'bye' in CSV to 'goodbye' to align renamed tags
    df['intent'] = df['intent'].replace({'bye': 'goodbye'})

    # 3. Clean and Vectorize
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    # Split into In-Domain and Out-of-Domain based on strict whitelisted allowed intents
    in_domain_tags = config.ALLOWED_INTENTS - {"fallback"}
    in_domain = df[df['intent'].isin(in_domain_tags)].copy()
    out_of_domain = df[~df['intent'].isin(in_domain_tags)].copy()

    # 4. Predict In-Domain Performance
    X_test_vec = vectorizer.transform(in_domain['clean_text'])
    y_true = in_domain['intent']
    
    # Get probabilities for confidence auditing
    probs_all = model.predict_proba(X_test_vec)
    pred_indices = np.argmax(probs_all, axis=1)
    y_pred = label_encoder.inverse_transform(pred_indices)
    max_probs = np.max(probs_all, axis=1)

    # --- METRICS CALCULATION ---
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    print("\n" + "="*40)
    print(" [ANALYSIS] FAILURE CLUSTER ANALYSIS")
    print("="*40)
    print(f"OVERALL ACCURACY: {acc*100:.2f}%")
    print(f"BALANCED ACCURACY: {balanced_acc*100:.2f}% ( fairer metric for small datasets )")
    print("-"*40)

    # 5. Fix Warnings with zero_division
    print("\nIntent-Level Performance (Precision, Recall, F1):")
    # This identifies if any classes are missing in y_true vs y_pred and silences warnings
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    print(classification_report(y_true, y_pred, labels=unique_labels, zero_division=0))

    # 6. Improved Failure Logging
    failures = []
    for idx, (text, true, pred, prob) in enumerate(zip(in_domain['text'], y_true, y_pred, max_probs)):
        if true != pred:
            failures.append(f"INPUT: {text}\nEXPECTED: {true}\nPREDICTED: {pred} (Conf: {prob:.2f})\n" + "-"*30)

    if not os.path.exists('logs'): os.makedirs('logs')
    with open(FAILURE_LOG, "w", encoding="utf-8") as f:
        f.write(f"--- HUMAN EVALUATION FAILURE AUDIT [{pd.Timestamp.now()}] ---\n")
        f.write(f"Total Misclassifications: {len(failures)}\n\n")
        f.write("\n".join(failures))
    
    print(f" Failure analysis saved to {FAILURE_LOG}")

    # 7. Professional OOD (Out-of-Domain) Detection Audit
    print("\n Out-of-Domain (OOD) Generalization Audit:")
    if not out_of_domain.empty:
        ood_vec = vectorizer.transform(out_of_domain['clean_text'])
        ood_probs = model.predict_proba(ood_vec)
        ood_max_probs = np.max(ood_probs, axis=1)
        ood_pred_indices = np.argmax(ood_probs, axis=1)
        ood_preds = label_encoder.inverse_transform(ood_pred_indices)
        
        # A good model should have LOW confidence for nonsense
        THRESHOLD = config.CONFIDENCE_THRESHOLD
        properly_ignored = sum(p < THRESHOLD for p in ood_max_probs)
        false_positives = len(out_of_domain) - properly_ignored
        
        print(f"Tested {len(out_of_domain)} nonsense/unrelated queries.")
        print(f"[OK] Properly Ignored (Low Confidence): {properly_ignored}")
        print(f"[ERR] False Positives (High Confidence Guess): {false_positives}")
        
        for text, pred, prob in zip(out_of_domain['text'], ood_preds, ood_max_probs):
            if prob >= THRESHOLD:
                print(f"   - Warning: '{text}' guessed as '{pred}' with {prob:.2f} confidence!")

    # 8. Visual Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Human Language Confusion Matrix (Balanced Test Set)')
    plt.ylabel('Actual Student Intent')
    plt.xlabel('Predicted Intent')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 9. Confidence Calibration Audit (New)
    print("\n Auditing Confidence Calibration...")
    plt.figure(figsize=(10, 6))
    sns.histplot(max_probs, bins=20, kde=True, color='green')
    plt.axvline(x=config.CONFIDENCE_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({config.CONFIDENCE_THRESHOLD})')
    plt.title('Distribution of Model Confidence Scores (Human Test Data)')
    plt.xlabel('Max Probability Score')
    plt.ylabel('Frequency')
    plt.legend()
    try:
        plt.savefig('logs/confidence_distribution.png')
        print(" Confidence distribution saved as 'logs/confidence_distribution.png'")
    except Exception:
        pass

    # Per-intent accuracy breakdown
    intent_acc = {}
    for label in unique_labels:
        mask = (y_true == label)
        if any(mask):
            intent_acc[label] = accuracy_score(y_true[mask], y_pred[mask])
    
    print("\n Per-Intent Recall Audit:")
    for intent, score in sorted(intent_acc.items(), key=lambda x: x[1]):
        status = " WEAK" if score < 0.70 else " STRONG"
        print(f"   {intent:15}: {score*100:6.2f}% {status}")

    print("\n" + "="*60)
    print("  AUDIT COMPLETE: Check logs/ for detailed breakdown.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_evaluation()


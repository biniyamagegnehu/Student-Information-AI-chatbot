# train.py
"""
University Student Information Chatbot - Intent Classification Training Pipeline
Phase 4: Advanced Classification Architecture

This script is responsible for:
1. Safe loading and rigorous schema validation of intents.json.
2. Direct integration of the Phase 3 preprocessing pipeline (preprocess_text).
3. Hybrid feature engineering (Word-Level + Char-Level TF-IDF FeatureUnion).
4. Balanced, calibrated intent classifier fitting (CalibratedClassifierCV).
5. Comprehensive evaluation (Precision, Recall, Macro F1, Weak Intents, Confusion Pairs).
6. Serializing artifacts (model, hybrid vectorizer, label encoder) into model/.
7. Writing a detailed execution audit report to logs/training_report.txt.
"""

import json
import os
import sys
import time
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

# Local modular imports
import config
from preprocess import preprocess_text

def validate_and_load_dataset(filepath: str) -> (list, dict):
    """
    Step 2: Load and validate dataset schema rigorously.
    Identifies empty fields, missing tags, missing patterns/responses, and duplicates.
    Rejects invalid datasets safely.
    Returns:
        cleaned_dataset: list of tuples (pattern_str, tag_str)
        stats: dict containing vocabulary stats
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset intents.json not found at: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"CRITICAL ERROR: intents.json is not valid JSON. Parse error: {e}")

    intents = data.get("intents", [])
    if not intents:
        raise ValueError("CRITICAL ERROR: intents.json has an empty or missing 'intents' list.")

    # Validation Counters
    total_intents = 0
    total_patterns = 0
    duplicate_patterns_removed = 0
    empty_patterns_found = 0
    empty_responses_found = 0
    patterns_per_intent = {}
    
    seen_patterns = set()
    cleaned_dataset = []  # List of tuples: (pattern, tag)
    vocab_estimator = set()

    for intent_group in intents:
        tag = intent_group.get("tag", "").strip()
        patterns = intent_group.get("patterns", [])
        responses = intent_group.get("responses", [])

        # 1. Validation Checks
        if not tag:
            raise ValueError("CRITICAL SCHEMA ERROR: Found an intent group without a 'tag' identifier.")
        
        if tag not in config.ALLOWED_INTENTS:
            raise ValueError(f"CRITICAL SCHEMA ERROR: Tag '{tag}' is not whitelisted in config.ALLOWED_INTENTS.")

        total_intents += 1
        patterns_per_intent[tag] = 0

        if not patterns:
            empty_patterns_found += 1
        if not responses:
            empty_responses_found += 1

        # 2. Process and Clean Patterns
        for pattern in patterns:
            pattern_str = pattern.strip()
            if not pattern_str:
                continue

            total_patterns += 1
            normalized_pattern = pattern_str.lower()
            
            # Simple vocabulary estimation (raw tokens)
            for token in normalized_pattern.split():
                vocab_estimator.add(token)

            if normalized_pattern in seen_patterns:
                duplicate_patterns_removed += 1
            else:
                seen_patterns.add(normalized_pattern)
                cleaned_dataset.append((pattern_str, tag))
                patterns_per_intent[tag] += 1

    # 3. Final Validation Safety Shield
    if not cleaned_dataset:
        raise ValueError("CRITICAL ERROR: Cleaned dataset contains 0 usable patterns. Cannot train.")
        
    for tag, count in patterns_per_intent.items():
        if count == 0:
            raise ValueError(f"CRITICAL SCHEMA ERROR: Intent group '{tag}' contains 0 patterns. All classes must have training examples.")

    # 4. Print Summary Report (No emojis for Windows Command Prompt safety)
    print("\n" + "=" * 55)
    print(" [INFO] DATASET VALIDATION SUMMARY REPORT")
    print("=" * 55)
    print(f"[OK] Loaded {total_intents} valid whitelisted intents")
    print(f"[OK] Total raw patterns parsed: {total_patterns}")
    print(f"[OK] Removed {duplicate_patterns_removed} duplicate patterns")
    print(f"[OK] Total unique patterns loaded: {len(cleaned_dataset)}")
    print(f"[OK] Estimated raw vocabulary size: {len(vocab_estimator)} words")
    
    if empty_patterns_found > 0:
        print(f"[WARN] Found {empty_patterns_found} intent groups with empty patterns list.")
    if empty_responses_found > 0:
        print(f"[WARN] Found {empty_responses_found} intent groups with empty responses list.")
        
    print("[OK] Dataset validation completed successfully.")
    print("=" * 55 + "\n")

    stats = {
        "total_intents": total_intents,
        "total_patterns": total_patterns,
        "unique_patterns": len(cleaned_dataset),
        "duplicates_removed": duplicate_patterns_removed,
        "raw_vocab_estimate": len(vocab_estimator),
        "patterns_per_intent": patterns_per_intent
    }

    return cleaned_dataset, stats


def train_classification_pipeline():
    """
    Main training engine. Fits hybrid vectorizers and calibrated models deterministically.
    """
    start_time = time.time()
    
    # 1. Load and Validate Dataset
    try:
        raw_dataset, db_stats = validate_and_load_dataset(config.INTENTS_JSON_PATH)
    except Exception as e:
        print(f"Validation Failed: {e}")
        sys.exit(1)

    # 2. Preprocessing Integration (Step 3)
    print("[INFO] Starting Preprocessing Pipeline...")
    X_raw = [item[0] for item in raw_dataset]
    y_raw = [item[1] for item in raw_dataset]

    # Process all patterns through our Phase 3 preprocess_text function
    X_clean = [preprocess_text(text) for text in X_raw]

    # 3. Label Encoding (Step 4)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    
    # Save label encoder early to guarantee deterministic indexing mapping
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)

    # 4. Advanced Hybrid TF-IDF Vectorization & Feature Engineering (Step 5 & 6)
    print("[INFO] Engineering Hybrid Feature Pipeline (Word-Level + Char-Level TF-IDF)...")
    
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True
    )
    
    # Use FeatureUnion to stack Word-level and Char-level sparse matrices
    hybrid_feature_pipeline = FeatureUnion([
        ('word_tfidf', word_vectorizer),
        ('char_tfidf', char_vectorizer)
    ])
    
    X_features = hybrid_feature_pipeline.fit_transform(X_clean)
    feature_count = X_features.shape[1]
    print(f"[OK] Successfully engineered {feature_count} hybrid feature matrices.")

    # 5. Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded,
        test_size=0.20,
        stratify=y_encoded,
        random_state=config.RANDOM_SEED
    )

    # 6. Classifier Training (Step 7)
    # Balanced, regularized Logistic Regression classifier optimized for small NLP vocabularies
    print("[INFO] Fitting regularized base Logistic Regression estimator...")
    base_estimator = LogisticRegression(
        solver='lbfgs',
        class_weight='balanced',
        max_iter=5000,
        random_state=config.RANDOM_SEED
    )

    # 7. Sigmoid Probability Calibration (Step 8)
    print("[INFO] Fitting probability calibration wrapper (CalibratedClassifierCV)...")
    min_samples = np.bincount(y_train).min()
    n_folds = max(2, min(3, min_samples))  # Stratified cross validation

    calibrated_model = CalibratedClassifierCV(
        estimator=base_estimator,
        method='sigmoid',
        cv=n_folds
    )
    calibrated_model.fit(X_train, y_train)

    # 8. Model Evaluation & Audit Metrics (Step 9)
    y_pred = calibrated_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train, calibrated_model.predict(X_train))
    
    target_names = label_encoder.classes_
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    # Audit Weak Classes (F1-score < 0.80) and Low Recall (< 0.80)
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    
    weak_intents = []
    low_recall_intents = []
    for tag in target_names:
        metrics = report_dict.get(tag, {})
        f1 = metrics.get('f1-score', 0.0)
        rec = metrics.get('recall', 0.0)
        if f1 < 0.80:
            weak_intents.append((tag, f1))
        if rec < 0.80:
            low_recall_intents.append((tag, rec))

    # Audit Confusion Pairs
    cm = confusion_matrix(y_test, y_pred)
    confusion_pairs = []
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            if i != j and cm[i][j] > 0:
                confusion_pairs.append((target_names[i], target_names[j], cm[i][j]))
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    # Training Duration calculation
    end_time = time.time()
    duration = end_time - start_time

    # Print Evaluation Reports to Console
    print("\n" + "=" * 55)
    print(f" [SUCCESS] MODEL TRAINING COMPLETED! Test Accuracy: {test_accuracy * 100:.2f}%")
    print("=" * 55)
    print(f" - Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f" - Test Accuracy : {test_accuracy * 100:.2f}%")
    print(f" - Macro F1-Score: {macro_f1:.4f}")
    print(f" - Training Time : {duration:.2f} seconds")
    print("-" * 55)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    if weak_intents:
        print("\n[AUDIT] Weak Intents Identified (F1-score < 0.80):")
        for tag, score in weak_intents:
            print(f"   * {tag:<18} : F1 = {score:.2f}")
    else:
        print("\n[AUDIT] All intents are extremely STRONG (F1 >= 0.80)!")

    if confusion_pairs:
        print("\n[AUDIT] Top Intent Confusion Pairs:")
        for actual, predicted, count in confusion_pairs[:5]:
            print(f"   * Actual '{actual:<15}' got confused with Predicted '{predicted:<15}' ({count} times)")

    print("=" * 55 + "\n")

    # 9. Save Serialized Artifacts (Step 11)
    joblib.dump(calibrated_model, config.MODEL_PATH)
    joblib.dump(hybrid_feature_pipeline, config.VECTORIZER_PATH)
    
    # 10. Generate Automated Execution Audit Report File (Step 12)
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)
        
    report_file_path = os.path.join(config.LOG_DIR, "training_report.txt")
    try:
        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("      UNIVERSITY STUDENT ASSISTANT - MODEL TRAINING AUDIT REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training Duration: {duration:.3f} seconds\n")
            f.write(f"Random Seed: {config.RANDOM_SEED}\n")
            f.write(f"Calibration Method: Sigmoid probability wrapper (cv={n_folds})\n\n")
            
            f.write("--- DATASET STATISTICS ---\n")
            f.write(f"Total Whitelisted Intents: {db_stats['total_intents']}\n")
            f.write(f"Total Raw Patterns Parsed: {db_stats['total_patterns']}\n")
            f.write(f"Duplicate Patterns Removed: {db_stats['duplicates_removed']}\n")
            f.write(f"Total Unique Patterns Trained: {db_stats['unique_patterns']}\n")
            f.write(f"Estimated Raw Vocabulary: {db_stats['raw_vocab_estimate']} words\n\n")
            
            f.write("Patterns Per Intent Breakdown:\n")
            for t, c in sorted(db_stats['patterns_per_intent'].items()):
                f.write(f" - {t:<20}: {c} patterns\n")
            f.write("\n")
            
            f.write("--- HYBRID FEATURE STATISTICS ---\n")
            f.write(f"Total Combined Engineered Features: {feature_count}\n")
            f.write(" - Word-Level TF-IDF (Unigrams, Bigrams, Sublinear TF)\n")
            f.write(" - Character-Level TF-IDF (3-5 Ngrams, Sublinear TF)\n\n")
            
            f.write("--- MODEL PERFORMANCE METRICS ---\n")
            f.write(f"Train Accuracy: {train_accuracy * 100:.2f}%\n")
            f.write(f"Test Accuracy : {test_accuracy * 100:.2f}%\n")
            f.write(f"Macro F1-Score: {macro_f1:.4f}\n\n")
            
            f.write("Classification Report Details:\n")
            f.write(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
            f.write("\n")
            
            f.write("Confusion Matrix:\n")
            cm_str = np.array2string(cm)
            f.write(cm_str + "\n\n")
            
            f.write("Intent Confusion Audits:\n")
            for actual, predicted, count in confusion_pairs:
                f.write(f" - Actual '{actual}' confused with '{predicted}' -> {count} times\n")
            f.write("\n" + "=" * 60 + "\n")
            
        print(f"[SUCCESS] Automated training audit log saved inside '{report_file_path}'!")
    except Exception as e:
        print(f"[Warning] Failed to save training audit log file: {e}")

if __name__ == "__main__":
    train_classification_pipeline()

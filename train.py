# train.py
import json
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Local modular imports
import config
from preprocess import preprocess_text

def validate_and_load_dataset(filepath: str):
    """
    Validation system that runs checks on intents.json before training.
    Detects:
    - Invalid tags (not whitelisted)
    - Empty patterns or responses
    - Duplicate patterns
    Prints a beautiful summary report and returns a flat dataset of (pattern, tag).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    intents = data.get("intents", [])
    
    loaded_intents_count = 0
    duplicate_patterns_removed = 0
    empty_responses_warnings = 0
    empty_patterns_warnings = 0
    invalid_tags_found = []
    
    seen_patterns = set()
    cleaned_dataset = [] # List of tuples: (pattern, tag)

    for intent in intents:
        tag = intent.get("tag", "").strip()
        patterns = intent.get("patterns", [])
        responses = intent.get("responses", [])

        # 1. Check for missing/invalid tag
        if not tag:
            empty_responses_warnings += 1
            continue
            
        if tag not in config.ALLOWED_INTENTS:
            invalid_tags_found.append(tag)
            continue

        loaded_intents_count += 1

        # 2. Check for empty patterns or responses
        if not patterns:
            empty_patterns_warnings += 1
        if not responses:
            empty_responses_warnings += 1

        # 3. Process patterns and clean duplicates
        for pattern in patterns:
            pattern_str = pattern.strip()
            if not pattern_str:
                continue

            # Check duplicate case-insensitively
            normalized_pattern = pattern_str.lower()
            if normalized_pattern in seen_patterns:
                duplicate_patterns_removed += 1
            else:
                seen_patterns.add(normalized_pattern)
                cleaned_dataset.append((pattern_str, tag))

    # --- PRINT BEAUTIFUL SUMMARY REPORT ---
    print("\n" + "=" * 50)
    print(" [INFO] DATASET VALIDATION SUMMARY REPORT")
    print("=" * 50)
    print(f"[OK] Loaded {loaded_intents_count} valid whitelisted intents")
    
    if duplicate_patterns_removed > 0:
        print(f"[OK] Removed {duplicate_patterns_removed} duplicate patterns")
    else:
        print("[OK] Found 0 duplicate patterns")

    if empty_responses_warnings > 0:
        print(f"[WARN] Found {empty_responses_warnings} empty/missing responses")
    if empty_patterns_warnings > 0:
        print(f"[WARN] Found {empty_patterns_warnings} empty/missing patterns")
        
    if invalid_tags_found:
        print(f"[ERROR] Rejected {len(invalid_tags_found)} out-of-scope intent tags: {set(invalid_tags_found)}")
    else:
        print("[OK] All intent tags perfectly match allowed whitelist scope")
        
    print("[OK] Dataset validation completed")
    print("=" * 50 + "\n")

    if not cleaned_dataset:
        raise ValueError("Critical Error: Cleaned dataset contains 0 patterns. Cannot train.")

    return cleaned_dataset


def train_pipeline():
    """
    Main pipeline responsible for:
    - Preprocessing text
    - Vectorizing (TF-IDF)
    - Fitting calibrated model
    - Evaluating metrics
    - Saving serialized model artifacts
    """
    # 1. Load and Validate Dataset
    try:
        raw_dataset = validate_and_load_dataset(config.INTENTS_JSON_PATH)
    except Exception as e:
        print(f"Validation Failed: {e}")
        return

    # 2. Clean and Preprocess patterns
    print("[INFO] Starting Preprocessing Pipeline...")
    X_raw = [item[0] for item in raw_dataset]
    y_raw = [item[1] for item in raw_dataset]

    X_clean = [preprocess_text(text) for text in X_raw]
    
    # 3. Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    # 4. Feature Extraction (TF-IDF)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1
    )
    X_features = vectorizer.fit_transform(X_clean)

    # 5. Stratified Train/Test Split
    # Make sure we have enough samples to perform stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=config.RANDOM_SEED
    )

    # 6. Model Training & Probability Calibration
    print("[INFO] Training Calibrated Model (Logistic Regression)...")
    
    # Stratified cross validation count based on minimum sample size
    min_samples = np.bincount(y_train).min()
    n_folds = max(2, min(3, min_samples))
    
    base_estimator = LogisticRegression(
        solver='lbfgs',
        class_weight='balanced',
        max_iter=1000,
        random_state=config.RANDOM_SEED
    )
    
    model = CalibratedClassifierCV(
        estimator=base_estimator,
        method='sigmoid',
        cv=n_folds
    )
    model.fit(X_train, y_train)

    # 7. Evaluation & Metrics Reports
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    target_names = label_encoder.classes_

    print("\n" + "=" * 50)
    # Print metrics
    print(f"[SUCCESS] MODEL EVALUATION SUCCESSFUL! Test Accuracy: {accuracy * 100:.2f}%")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    print("Confusion Matrix Summary:")
    cm = confusion_matrix(y_test, y_pred)
    # Print simplified confusion matrix summary
    for i, class_name in enumerate(target_names):
        correct = cm[i][i]
        total = sum(cm[i])
        pct = (correct / total * 100) if total > 0 else 0
        print(f" - {class_name:20}: {correct}/{total} correctly predicted ({pct:.1f}%)")
    print("=" * 50 + "\n")

    # 8. Save Serialized Artifacts
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)

    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)

    print(f"[SUCCESS] Successfully saved model artifacts inside '{config.MODEL_DIR}/'!")

if __name__ == "__main__":
    train_pipeline()

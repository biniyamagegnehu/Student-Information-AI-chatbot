import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from preprocess import preprocess_text

# --- Project Paths ---
DATASET_FILE = "dataset/dataset.csv"
MODEL_DIR = "model"

def train_pipeline():
    """
    Complete ML pipeline: Load -> Clean -> Vectorize -> Train -> Calibrate -> Save
    """
    if not os.path.exists(DATASET_FILE):
        print(f"Critical Error: Dataset not found at {DATASET_FILE}")
        return

    # 1. Load Data
    print("\n--- [LOG] Loading Data ---")
    df = pd.read_csv(DATASET_FILE)
    
    # 2. Dataset Cleaning (Deduplication)
    initial_len = len(df)
    df = df.drop_duplicates(subset=['text'])
    print(f"Cleaned dataset: {len(df)} samples (Removed {initial_len - len(df)} duplicates)")

    # 3. Preprocessing
    print("--- [LOG] Preprocessing Text ---")
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    X = df['clean_text']
    y = df['intent']

    # 4. Vectorization (TF-IDF)
    print("--- [LOG] Vectorizing Features ---")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3), 
        max_df=0.90, 
        min_df=2, 
        sublinear_tf=True
    )
    X_features = vectorizer.fit_transform(X)

    # 5. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, stratify=y, random_state=42
    )

    # 6. Training with Calibration
    # Logistic Regression + Sigmoid Calibration gives realistic probability scores
    print("--- [LOG] Training Calibrated Model ---")
    base_clf = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000)
    model = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=5)
    model.fit(X_train, y_train)

    # 7. Evaluation
    score = model.score(X_test, y_test)
    print(f" Success! Test Accuracy: {score*100:.2f}%")

    # 8. Save Artifacts
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    print(f" Model files saved to {MODEL_DIR}/")

if __name__ == "__main__":
    train_pipeline()


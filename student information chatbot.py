"""
Student Information Chatbot - Professionally Evaluated NLP Version
"""
import os
import re
import random
import joblib
import pandas as pd
import tkinter as tk
from tkinter import scrolledtext
from datetime import datetime

# Visualization and Evaluation Imports
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV

# Fuzzy String Matching for Typo Correction
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    import difflib
    HAS_RAPIDFUZZ = False

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- 0. SETUP & INITIALIZATION ---
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- 1. TEXT PREPROCESSING SECTION ---
lemmatizer = WordNetLemmatizer()

# Load default English stopwords
stop_words = set(stopwords.words('english'))
# PROFESSIONAL NLP TIP: Do not remove question words or vital auxiliary words!
# Words like "where" and "when" strongly correlate with "location" and "schedule" intents.
important_words_to_keep = {'where', 'when', 'how', 'what', 'who', 'which', 'why', 'can', 'do', 'is', 'are', 'i', 'my'}
stop_words = stop_words - important_words_to_keep

# --- DOMAIN-SPECIFIC VOCABULARY & TYPO HANDLING ---
# A curated list of critical university terms used for fuzzy spell correction.
# We do NOT use generic English dictionaries (like TextBlob) because they often falsely 
# "correct" campus-specific acronyms or domain words.
DOMAIN_VOCABULARY = [
    "registration", "schedule", "library", "where", "please", "information", 
    "administration", "department", "computer", "science", "technology", 
    "laboratory", "dormitory", "registrar", "deadline", "tuition", "fee", 
    "payment", "transcript", "internship", "hostel", "availability", 
    "admission", "requirements", "exam", "courses", "scholarship", "campus"
]

# Explicit Shorthand & Abbreviation Expansions
SHORTHAND_EXPANSIONS = {
    "pls": "please",
    "plz": "please",
    "info": "information",
    "admin": "administration",
    "dept": "department",
    "cs": "computer science",
    "it": "information technology",
    "lab": "laboratory",
    "dorm": "dormitory",
    "registar": "registrar", 
    "uni": "university",
    "asap": "urgently",
    "wher": "where"
}

def apply_typo_correction(word):
    """
    Applies explicit shorthand expansion and confidence-aware fuzzy spell correction.
    """
    # 1. First, check explicit shorthand/abbreviation expansions
    if word in SHORTHAND_EXPANSIONS:
        # Some shorthands expand to multiple words (e.g. "cs" -> "computer science")
        # We handle this splitting logic downstream, but for now just return the expansion string.
        # Actually, it's safer to return the string and let the vectorizer handle the space.
        return SHORTHAND_EXPANSIONS[word]
        
    # 2. Skip correction for very short words, numbers, or stop words to prevent false positives (ambiguous corrections)
    if len(word) <= 3 or word in stop_words:
        return word
        
    # 3. Fuzzy Matching (Spell Correction against Domain Vocabulary)
    # Using rapidfuzz if available (lightning fast, C++ backend) else fallback to difflib
    # The 'score_cutoff' / 'cutoff' parameter acts as a confidence threshold. 
    # E.g., 80% similarity required. This protects against incorrect auto-corrections.
    if HAS_RAPIDFUZZ:
        match = process.extractOne(word, DOMAIN_VOCABULARY, scorer=fuzz.ratio, score_cutoff=80)
        if match:
            corrected_word = match[0]
            if corrected_word != word:
                with open("typo_corrections_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now()}] RapidFuzz Corrected: '{word}' -> '{corrected_word}'\n")
            return corrected_word
    else:
        matches = difflib.get_close_matches(word, DOMAIN_VOCABULARY, n=1, cutoff=0.8)
        if matches:
            corrected_word = matches[0]
            if corrected_word != word:
                with open("typo_corrections_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now()}] Difflib Corrected: '{word}' -> '{corrected_word}'\n")
            return corrected_word
            
    return word

def preprocess_text(text):
    """
    Professionally cleans and normalizes the input text for NLP intent classification.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercase conversion: standardizes all text
    text = text.lower()
    
    # 2. Number cleanup: Removes numbers which are usually noise for generic intent classification
    text = re.sub(r'\d+', '', text)
    
    # 3. Punctuation & Special Character removal: Replaces them with space
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 4. Whitespace normalization: Cleans up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Tokenization: Splits text into individual words safely
    try:
        words = word_tokenize(text)
    except LookupError:
        words = text.split() # Fallback if punkt is missing
    
    clean_words = []
    for word in words:
        # 6. Apply intelligent typo correction and shorthand expansion
        corrected_phrase = apply_typo_correction(word)
        
        # If shorthand expanded to multiple words (e.g. "computer science"), process them
        for corrected_word in corrected_phrase.split():
            # 7. Stopword removal (excluding important question words)
            if corrected_word not in stop_words:
                # 8. Lemmatization: Reduces words to their base dictionary form
                lemma = lemmatizer.lemmatize(corrected_word)
                clean_words.append(lemma)
            
    # 9. Rejoin into a single normalized string
    return ' '.join(clean_words)

# --- 2. LOGGING SECTION ---
def log_low_confidence_query(user_input, max_prob, predicted_intent="UNKNOWN"):
    """
    Logs failed, nonsense, or low-confidence queries to a file for future dataset improvement.
    This is critical for finding out what students are asking that the bot doesn't know!
    """
    with open("unanswered_queries_log.txt", "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if max_prob == 0.0:
            f.write(f"[{timestamp}] NONSENSE/UNRECOGNIZED | Query: {user_input}\n")
        else:
            f.write(f"[{timestamp}] LOW CONFIDENCE ({max_prob*100:.1f}%) | Guessed: {predicted_intent} | Query: {user_input}\n")

# --- 3. MODEL EVALUATION & TESTING SECTION ---
def test_sample_queries(model, vectorizer, model_name="Model"):
    """
    Tests the model with hard, realistic, and unseen student queries to evaluate REAL generalization.
    """
    print(f"\n--- 🧪 REALISTIC & ADVERSARIAL QUERY TESTING ({model_name}) ---")
    # HARD UNSEEN & ADVERSARIAL QUERIES
    # These queries test if the bot actually generalized the concepts, or if it just 
    # memorized the training data's exact phrasing.
    test_queries = [
        "regstration deadline",                       # Common missing vowel
        "exam scheduel pls",                          # Transposed letters + abbreviation
        "wher is librery",                            # Multiple typos in one query
        "fee payment when???",                        # Punctuation spam + short structure
        "cs dept location",                           # Multiple shorthand terms
        "transcript urgently needed",                 # Proper word, out of order structure
        "hostel avalability",                         # Missing letter
        "admisson requirements",                      # Missing letter
        "asdfghjkl",                                  # Adversarial: Pure nonsense
    ]
    
    for query in test_queries:
        clean_q = preprocess_text(query)
        if not clean_q:
            print(f"Query: '{query}'\n-> ⚠️ Dropped during preprocessing (Nonsense word)")
            print("-" * 30)
            continue
            
        vec_q = vectorizer.transform([clean_q])
        probs = model.predict_proba(vec_q)[0]
        max_prob = max(probs)
        pred = model.classes_[probs.argmax()]
        
        print(f"Query: '{query}'")
        print(f"-> Predicted Intent: {pred} (Confidence Score: {max_prob*100:.1f}%)")
        if max_prob < 0.60:  # Updated to new calibrated threshold
            print("-> ⚠️ LOW CONFIDENCE (Fallback would trigger)")
        print("-" * 30)

def train_and_evaluate_model():
    """
    Loads data, trains the NLP model, evaluates its accuracy using multiple metrics, and saves it.
    """
    print("Loading dataset...")
    try:
        data = pd.read_csv("dataset.csv")
    except FileNotFoundError:
        print("Error: dataset.csv not found! Please create it first.")
        return None, None
        
    # --- 1. DATASET AUDIT & LEAKAGE DETECTION ---
    # EXPLANATION OF DATASET LEAKAGE & ARTIFICIAL ACCURACY INFLATION:
    # "Dataset Leakage" occurs when testing data contains samples that are identical 
    # (or semantically identical) to the training data. Generated datasets often have this issue.
    # If a bot trains on "where is the library" and is tested on the exact same phrase, 
    # it scores 100% via MEMORIZATION, not real learning. This leads to artificially inflated 
    # accuracy (e.g. 99%) that completely collapses in real-world production.
    
    print("\n--- 🕵️ DATASET QUALITY AUDIT ---")
    initial_count = len(data)
    
    # Detect exact duplicate rows
    duplicates = data.duplicated(subset=['text']).sum()
    if duplicates > 0:
        print(f"⚠️ Warning: Found {duplicates} exact duplicate queries in dataset!")
        print("-> Removing duplicates to prevent memorization and accuracy inflation...")
        data = data.drop_duplicates(subset=['text'])
        
    print(f"Dataset Size after deduplication: {len(data)} (Original: {initial_count})")
    
    # HOW TO DETECT SEMANTIC DUPLICATES (NEAR-DUPLICATES):
    # Even after exact deduplication, generated data might have:
    # 1. "where is the main library"
    # 2. "where is the campus library"
    # To detect this, you can compute pairwise cosine similarities of TF-IDF vectors 
    # before splitting. If similarity > 0.95, drop one. 
    # Best practice for reducing overfitting: Keep training datasets heavily diverse, 
    # combining short, long, formal, and typo-ridden phrases.
        
    print("\nPreprocessing text data...")
    data['clean_text'] = data['text'].apply(preprocess_text)
    
    X = data['clean_text']
    y = data['intent']
    
    # --- 2. STRATIFIED TRAIN/TEST SPLITTING ---
    # WHY WE USE STRATIFIED SPLITTING (StratifiedShuffleSplit approach):
    # Random splitting might accidentally put all examples of a rare intent (e.g. 'holidays') 
    # into the training set, leaving none for the test set (or vice versa).
    # Stratified splitting ensures every intent class maintains its exact original ratio 
    # in both the training and testing sets, providing an honest evaluation.
    print("Splitting data using Stratified Splitting (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 2. Optimized TF-IDF Vectorization
    # - ngram_range=(1, 3): Captures single words, pairs, and triplets (e.g. "computer science department")
    # - max_df=0.90: Ignores terms that appear in >90% of documents (corpus-specific stopwords)
    # - min_df=2: Ignores terms that appear in less than 2 documents (removes one-off weird typos)
    # - sublinear_tf=True: Replaces raw term frequency with 1 + log(tf). Reduces dominance of repeated words.
    # - norm='l2': Normalizes vectors to length 1, ensuring short and long questions are comparable.
    # - token_pattern: The default ignores single letters. We use \b\w+\b to keep "i" and other single-character intents if needed.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_df=0.90,
        min_df=2,
        sublinear_tf=True,
        norm='l2',
        token_pattern=r"(?u)\b\w+\b"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test) # Transform only! Prevent data leakage.
    
    # 3. Advanced NLP Intent Classifier: Logistic Regression
    # WHY LOGISTIC REGRESSION OUTPERFORMS NAIVE BAYES FOR NLP:
    # - Naive Bayes treats words independently. It struggles when intents share overlapping vocabulary (e.g., "when is the exam" vs "where is the exam").
    # - Logistic Regression mathematically assigns positive/negative weights to TF-IDF n-grams. It learns that "where" pushes the prediction 
    #   towards 'location', while "when" pushes it towards 'schedule', yielding much higher accuracy and confidence scores.
    print("Training Professionally Tuned Base Logistic Regression model...")
    
    base_model = LogisticRegression(
        # solver='lbfgs': The optimal, highly efficient solver for multiclass NLP datasets.
        solver='lbfgs',
        # C=1.0: Inverse of regularization strength. 1.0 is balanced to prevent overfitting to noisy student typos.
        C=1.0, 
        # class_weight='balanced': Automatically adjusts mathematical weights so smaller intent classes aren't ignored.
        class_weight='balanced', 
        # max_iter=1000: Gives the mathematical solver enough iterations to securely converge on the complex text data.
        max_iter=1000
        # multi_class is automatically handled by the solver in modern scikit-learn versions!
    )
    base_model.fit(X_train_vec, y_train)
    
    # --- PROBABILITY CALIBRATION ---
    # EXPLANATION OF PROBABILITY CALIBRATION:
    # Probability calibration transforms model outputs into true probabilities. 
    # If a calibrated model predicts an intent with 70% confidence, it means that 
    # 70% of the time it makes this prediction, it is actually correct.
    #
    # WHY LOGISTIC REGRESSION PROBABILITIES MAY BE POORLY CALIBRATED:
    # Logistic Regression minimizes log-loss. In datasets with overlapping vocabulary 
    # or unbalanced classes, it can become overconfident in its predictions. A 90% 
    # confidence score might actually mean it's only 50% sure.
    # 
    # WHY CALIBRATION IMPROVES CHATBOT RELIABILITY:
    # By mapping the model's output to actual probabilities, our fallback threshold 
    # (e.g., < 0.60) becomes incredibly robust. Uncalibrated models often bypass 
    # fallbacks by outputting false high-confidence scores for nonsense inputs.
    #
    # BEST PRACTICES FOR CALIBRATED NLP CLASSIFIERS:
    # 1. Use 'cv=5' (Cross-Validation) so calibration is done on unseen fold data.
    # 2. 'sigmoid' (Platt scaling) is best for small/medium datasets.
    # 3. 'isotonic' can be used if you have thousands of samples, but overfits small data.
    #
    # COMMON MISTAKES TO AVOID:
    # - Using 'isotonic' on a tiny dataset (causes severe overfitting).
    # - Setting cv="prefit" without actually having an independently pre-fitted model.
    
    print("\n--- ⚖️ PROBABILITY CALIBRATION (METHOD COMPARISON) ---")
    print("Training Calibrated Models...")
    
    # 3a. Sigmoid Calibration
    print("1. Training Sigmoid (Platt) Calibrated Model...")
    sigmoid_calibrated_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=5)
    sigmoid_calibrated_model.fit(X_train_vec, y_train)
    
    # 3b. Isotonic Calibration
    print("2. Training Isotonic Calibrated Model...")
    isotonic_calibrated_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=5)
    isotonic_calibrated_model.fit(X_train_vec, y_train)
    
    # Select Sigmoid as the primary model (best practice for this type of classification)
    print("-> Selecting 'sigmoid' calibration for production pipeline.")
    model = sigmoid_calibrated_model
    
    print("\n===========================================")
    print("   📊 CALIBRATION CONFIDENCE COMPARISON    ")
    print("===========================================")
    # 4. Confidence score testing before vs after calibration
    test_sample_queries(base_model, vectorizer, "UNCALIBRATED Logistic Regression")
    test_sample_queries(model, vectorizer, "CALIBRATED (Sigmoid) Logistic Regression")
    
    print("\n===========================================")
    print("        📊 MODEL EVALUATION RESULTS        ")
    print("===========================================")
    
    # Predictions on test set
    y_pred = model.predict(X_test_vec)
    
    # 2. Accuracy Score: Percentage of exactly correct predictions overall
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy * 100:.2f}%\n")
    
    # 3. Cross-Validation: Evaluates model performance across 5 different subsets of the data
    # cross_val_score uses StratifiedKFold by default for classification tasks.
    # This proves the model is genuinely robust and not just lucky with the initial split.
    print("Running 5-Fold Cross Validation on CALIBRATED Model...")
    # Using a pipeline ensures no data leakage during cross-validation
    pipeline_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.90, min_df=2, sublinear_tf=True, norm='l2', token_pattern=r"(?u)\b\w+\b")
    pipeline_base_model = LogisticRegression(solver='lbfgs', C=1.0, class_weight='balanced', max_iter=1000)
    pipeline_calibrated_model = CalibratedClassifierCV(estimator=pipeline_base_model, method='sigmoid', cv=3)
    pipeline = make_pipeline(pipeline_vectorizer, pipeline_calibrated_model)
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"Cross-Validation Accuracy Scores: {[round(score*100, 2) for score in cv_scores]}")
    print(f"Average CV Accuracy: {cv_scores.mean() * 100:.2f}%\n")
    
    # 4. Classification Report: Shows Precision, Recall, and F1-Score for each intent
    # - Precision: When it predicted 'fees', how often was it actually 'fees'?
    # - Recall: Out of all real 'fees' queries, how many did it catch?
    # - F1-Score: Harmonic mean of precision and recall.
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("💡 Tip: If any intent has an F1-score below 0.80, add more diverse examples for it in dataset.csv!\n")
    
    # 5. Confusion Matrix Visualization
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Intent Classification Confusion Matrix')
    plt.ylabel('Actual Intent')
    plt.xlabel('Predicted Intent')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    try:
        plt.savefig('confusion_matrix.png')
        print("✅ Confusion matrix visually saved as 'confusion_matrix.png'.")
    except Exception as e:
        print(f"Could not save confusion matrix plot: {e}")
    plt.close() # Close plot so script continues
    
    # (Manual tests were moved to run before model evaluation to compare calibrations)

    print("Saving model and vectorizer...")
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print("Training and Evaluation complete!\n")
    
    return model, vectorizer

# --- 4. MODEL LOADING SECTION ---
MODEL_FILE = "chatbot_model.pkl"
VECTORIZER_FILE = "chatbot_vectorizer.pkl"

# Note: We are forcing the script to retrain on every run so you can see the 
# evaluation printouts and updated visualizations during testing! 
# In a real deployed app, you would add an `if os.path.exists(MODEL_FILE):` check here.
print("Initiating training and evaluation pipeline...")
model, vectorizer = train_and_evaluate_model()

# --- 5. PREDICTION & RESPONSE HANDLING SECTION ---
responses = {
    "greeting": ["Hello! How can I help you today?", "Hi there! What university information do you need?"],
    "registration": ["Registration is open online through the student portal. Late registration incurs a penalty."],
    "courses": ["You can check available courses and your curriculum on the university portal."],
    "schedule": ["Class schedules are posted on the departmental notice boards or the student information system."],
    "exam": ["Exam schedules are announced two weeks before the exam period. Bring your ID card."],
    "location": ["Check the campus map near the entrance. The main library is in block 5."],
    "fees": ["Tuition and cost-sharing payments can be made via CBE Birr or Telebirr."],
    "scholarship": ["We offer merit-based and need-based scholarships. Apply at the student affairs office."],
    "hostel": ["Dormitory placements are announced via the student portal. Contact the proctor for issues."],
    "admission": ["Admission requires passing the national university entrance exam. Check the MOE portal."],
    "contacts": ["You can reach the registrar at info@university.edu or call the toll-free center."],
    "results": ["Grades and cumulative GPA can be viewed on your SIS portal. Transcripts are at the registrar."],
    "holidays": ["The university observes all national and public holidays. Check the academic calendar."],
    "thanks": ["You're very welcome!", "Glad I could help!"],
    "bye": ["Goodbye! Have a great day ahead.", "See you later! Good luck with your studies."]
}

# --- 5. CONVERSATIONAL MEMORY & CONTEXT HANDLING ---
class ChatbotMemory:
    """
    Lightweight, non-deep-learning conversational memory system.
    Tracks entities and intents for pronoun resolution (e.g., "it", "there").
    """
    def __init__(self):
        self.last_entity = None
        self.last_intent = None
        self.last_interaction_time = None
        # Context expires after 3 minutes to avoid weird carry-overs
        self.EXPIRATION_SECONDS = 180 
        
        self.PRONOUNS = {"it", "they", "there", "that", "this", "he", "she"}
        self.ENTITIES = {
            "library", "registrar", "engineering", "dormitory", "dorm", "cs", 
            "transcript", "fee", "tuition", "scholarship", "exam", "course", 
            "hostel", "admission", "department", "office", "campus"
        }

    def extract_entity(self, text):
        words = text.split()
        for word in words:
            if word in self.ENTITIES:
                return word
        return None

    def resolve_context(self, text):
        """Replaces pronouns with the remembered entity if the context is fresh."""
        if self.last_interaction_time:
            time_diff = (datetime.now() - self.last_interaction_time).total_seconds()
            if time_diff > self.EXPIRATION_SECONDS:
                self.clear_memory()
                return text, False # Context expired
                
        words = text.split()
        has_pronoun = any(word in self.PRONOUNS for word in words)
        
        if has_pronoun and self.last_entity:
            # Replace pronouns with the tracked entity for the ML pipeline
            resolved_words = [self.last_entity if w in self.PRONOUNS else w for w in words]
            resolved_text = " ".join(resolved_words)
            
            # Log the context resolution for debugging
            with open("context_memory_log.txt", "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now()}] Context Resolved: '{text}' -> '{resolved_text}'\n")
                
            return resolved_text, True
            
        return text, False
        
    def update_memory(self, text, intent):
        """Updates the short-term memory with new entities."""
        entity = self.extract_entity(text)
        if entity:
            self.last_entity = entity
        self.last_intent = intent
        self.last_interaction_time = datetime.now()
        
    def clear_memory(self):
        """Resets the memory."""
        self.last_entity = None
        self.last_intent = None
        self.last_interaction_time = None

# Initialize memory globally
chat_memory = ChatbotMemory()

# --- 6. PREDICTION & RESPONSE GENERATION ---
def get_chatbot_response(user_input):
    if model is None or vectorizer is None:
        return "Sorry, the AI model failed to load.", 0.0
        
    # --- A. CONTEXT RESOLUTION ---
    # Try to resolve pronouns like "it" or "there" using the short-term memory
    # We do this BEFORE heavy NLP preprocessing so we can detect standard English pronouns.
    context_aware_input, used_context = chat_memory.resolve_context(user_input.lower())
    
    clean_input = preprocess_text(context_aware_input)
    
    # --- 1. NONSENSE / EMPTY INPUT HANDLING ---
    if not clean_input:
        log_low_confidence_query(user_input, 0.0, "NONSENSE")
        fallback_nonsense = [
            "I didn't quite catch any standard words there. Could you rephrase?",
            "I only understand standard English phrases. Can you try typing that again?",
            "Sorry, that doesn't look like a question about university services."
        ]
        return random.choice(fallback_nonsense), 0.0
        
    # Vectorize and Predict
    user_vec = vectorizer.transform([clean_input])
    probabilities = model.predict_proba(user_vec)[0]
    max_prob = max(probabilities)
    prediction = model.classes_[probabilities.argmax()]
    
    # --- 2. PROFESSIONAL CONFIDENCE THRESHOLD LOGIC ---
    # Why this matters: Logistic Regression with TF-IDF will ALWAYS pick a class, even if the input is "i love football".
    # By setting a threshold, we force the bot to admit ignorance instead of lying or providing irrelevant info.
    # 
    # Tuning Guide for Calibrated Models:
    # - Too low (e.g., 0.30): Bot might still guess randomly.
    # - Too high (e.g., 0.85): Bot refuses perfectly valid, but slightly uniquely phrased questions.
    # - Balanced (0.60 - 0.70): The optimal sweet spot for calibrated probabilities.
    CONFIDENCE_THRESHOLD = 0.60 
    
    if max_prob < CONFIDENCE_THRESHOLD:
        log_low_confidence_query(user_input, max_prob, prediction) # Log the failure so developers can add it to the dataset!
        
        # --- 3. MULTIPLE FALLBACK RESPONSES ---
        # Randomizing fallbacks makes the bot feel much more natural and human-like
        fallback_responses = [
            "I'm not entirely sure about that. Could you rephrase your question?",
            "I don't have enough confidence to answer that accurately. Are you asking about courses, fees, or something else?",
            "I'm still learning! Could you ask that in a slightly different way?",
            "I didn't quite get that. I specialize in university registration, schedules, locations, and admissions."
        ]
        
        # If context was used but it still failed, the memory might be stale or wrong. Clear it.
        if used_context:
            chat_memory.clear_memory()
            
        return random.choice(fallback_responses), max_prob
        
    # --- B. MEMORY UPDATE ---
    # If the bot successfully understood the question, update the short-term memory with the new context!
    chat_memory.update_memory(context_aware_input, prediction)
    
    # --- C. CONTEXT-AWARE RESPONSE GENERATION ---
    # If we know the user is talking about a specific entity (e.g. library), we provide 
    # a highly specific answer rather than the generic intent response.
    dynamic_responses = {
        ("location", "library"): "The main library is located in Block 5 near the main gate.",
        ("schedule", "library"): "The library is open from 8:00 AM to 10:00 PM on weekdays, and closes at 5:00 PM on weekends.",
        ("location", "registrar"): "The registrar office is in the Admin Building, Ground Floor.",
        ("schedule", "registrar"): "The registrar office is open from 9:00 AM to 4:00 PM.",
        ("location", "engineering"): "The engineering department is located in Block 3.",
        ("contacts", "engineering"): "The head of the engineering department can be reached at eng_head@university.edu.",
        ("fees", "transcript"): "Transcripts cost 50 Birr per copy. You can pay at the registrar.",
        ("location", "transcript"): "You can pick up your transcript from the main registrar office."
    }
    
    entity = chat_memory.last_entity
    if entity and (prediction, entity) in dynamic_responses:
        return dynamic_responses[(prediction, entity)], max_prob
        
    # --- 4. SUCCESSFUL GENERIC RESPONSE ---
    return random.choice(responses.get(prediction, ["Sorry, I don't have information on that."])), max_prob

# --- 7. GUI SECTION ---
def send_message(event=None):
    user_input = entry.get().strip()
    if user_input == "": return

    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, "You: " + user_input + "\n", "user_tag")
    
    # Unpack the response AND the confidence score
    response, confidence = get_chatbot_response(user_input)
    
    # 5. CONFIDENCE DISPLAY FOR DEBUGGING
    # Appends the confidence score in brackets for the developer to see
    debug_info = f" [Confidence: {confidence*100:.1f}%]" if confidence > 0 else " [Nonsense/Unknown]"
    
    chat_area.insert(tk.END, "Bot: " + response + debug_info + "\n\n", "bot_tag")
    chat_area.config(state=tk.DISABLED)
    chat_area.yview(tk.END) 
    entry.delete(0, tk.END)

    clean_input = preprocess_text(user_input)
    if clean_input:
        user_vec = vectorizer.transform([clean_input])
        prediction = model.classes_[model.predict_proba(user_vec)[0].argmax()]
        max_prob = max(model.predict_proba(user_vec)[0])
        # Use the updated professional threshold to decide when to quit
        if prediction == "bye" and max_prob >= 0.60:
            root.after(2000, root.destroy)

# GUI Setup
root = tk.Tk()
root.title("AI Student Chatbot - Evaluated Edition")
root.geometry("550x650")
root.config(bg="#1e1e2f")

title = tk.Label(root, text="🎓 AI Student Information Chatbot", font=("Helvetica", 16, "bold"), bg="#1e1e2f", fg="white")
title.pack(pady=15)

info = tk.Label(root, text="I can help with: Registration, Courses, Schedule, Exams, Locations,\nFees, Scholarships, Hostels, Admissions, Contacts, Results, and Holidays.", font=("Helvetica", 10), bg="#1e1e2f", fg="#cccccc", justify=tk.CENTER)
info.pack(pady=5)

chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=22, font=("Helvetica", 11), bg="#2b2b3c", fg="white", padx=10, pady=10)
chat_area.pack(padx=15, pady=10)
chat_area.tag_config("user_tag", foreground="#4CAF50") 
chat_area.tag_config("bot_tag", foreground="#00bcd4")  

chat_area.insert(tk.END, "Bot: Hello! I'm your AI campus assistant. How can I help you today? 😊\n\n", "bot_tag")
chat_area.config(state=tk.DISABLED)

input_frame = tk.Frame(root, bg="#1e1e2f")
input_frame.pack(pady=10, fill=tk.X, padx=15)

entry = tk.Entry(input_frame, width=45, font=("Helvetica", 12), bg="#3e3e50", fg="white", insertbackground="white")
entry.pack(side=tk.LEFT, padx=5, ipady=5)
entry.bind("<Return>", send_message)

send_btn = tk.Button(input_frame, text="Send", command=send_message, bg="#4CAF50", fg="white", font=("Helvetica", 11, "bold"), width=8, activebackground="#45a049")
send_btn.pack(side=tk.RIGHT, padx=5)

if __name__ == "__main__":
    if model is not None:
        print("Starting GUI...")
        root.mainloop()
    else:
        print("Chatbot could not start because the model failed to load or train.")
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 0. SETUP & INITIALIZATION ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- 1. TEXT PREPROCESSING SECTION ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and normalizes the input text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    clean_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(clean_words)

# --- 2. LOGGING SECTION ---
def log_low_confidence_query(user_input, max_prob):
    """
    Logs failed or low-confidence queries to a file for future dataset improvement.
    """
    with open("unanswered_queries_log.txt", "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Confidence: {max_prob*100:.1f}% | Query: {user_input}\n")

# --- 3. MODEL EVALUATION & TESTING SECTION ---
def test_sample_queries(model, vectorizer):
    """
    Tests the model with manual, realistic student queries to evaluate predictions.
    """
    print("\n--- 🧪 MANUAL SAMPLE QUERY TESTING ---")
    test_queries = [
        "when is the deadline to pay my tuition?",
        "where is the computer lab located?",
        "I need my transcript for graduation",
        "how do i apply for a scholarship?",
        "asdfghjkl", # Nonsense query to test fallback
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
        if max_prob < 0.15:
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
        
    print("Preprocessing text data...")
    data['clean_text'] = data['text'].apply(preprocess_text)
    
    X = data['clean_text']
    y = data['intent']
    
    # 1. Train/Test Split (Train only on training data)
    print("Splitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test) # Transform only! Prevent data leakage.
    
    # Train the Model
    print("Training MultinomialNB model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    print("\n===========================================")
    print("        📊 MODEL EVALUATION RESULTS        ")
    print("===========================================")
    
    # Predictions on test set
    y_pred = model.predict(X_test_vec)
    
    # 2. Accuracy Score: Percentage of exactly correct predictions overall
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy * 100:.2f}%\n")
    
    # 3. Cross-Validation: Evaluates model performance across 5 different subsets of the data
    # This proves the model is genuinely robust and not just lucky with the initial split.
    print("Running 5-Fold Cross Validation...")
    # Using a pipeline ensures no data leakage during cross-validation
    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB())
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
    
    # Run manual tests
    test_sample_queries(model, vectorizer)

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

def get_chatbot_response(user_input):
    if model is None or vectorizer is None:
        return "Sorry, the AI model failed to load."
        
    clean_input = preprocess_text(user_input)
    if not clean_input:
        return "I didn't quite catch that. Could you use standard words?"
        
    user_vec = vectorizer.transform([clean_input])
    probabilities = model.predict_proba(user_vec)[0]
    max_prob = max(probabilities)
    
    CONFIDENCE_THRESHOLD = 0.15 
    
    if max_prob < CONFIDENCE_THRESHOLD:
        log_low_confidence_query(user_input, max_prob) # Log the failure
        return "I'm not quite sure I understand. Could you rephrase your question about university services?"
        
    prediction = model.classes_[probabilities.argmax()]
    return random.choice(responses.get(prediction, ["Sorry, I don't have information on that."]))

# --- 6. GUI SECTION ---
def send_message(event=None):
    user_input = entry.get().strip()
    if user_input == "": return

    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, "You: " + user_input + "\n", "user_tag")
    
    response = get_chatbot_response(user_input)
    chat_area.insert(tk.END, "Bot: " + response + "\n\n", "bot_tag")
    chat_area.config(state=tk.DISABLED)
    chat_area.yview(tk.END) 
    entry.delete(0, tk.END)

    clean_input = preprocess_text(user_input)
    if clean_input:
        user_vec = vectorizer.transform([clean_input])
        prediction = model.classes_[model.predict_proba(user_vec)[0].argmax()]
        max_prob = max(model.predict_proba(user_vec)[0])
        if prediction == "bye" and max_prob >= 0.15:
            root.after(1500, root.destroy)

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
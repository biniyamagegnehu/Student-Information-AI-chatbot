import tkinter as tk
from tkinter import scrolledtext
import joblib
import os
import threading
import time
from datetime import datetime

# Local Modular Imports
from preprocess import InputValidator, preprocess_text, extract_all_entities
from responses import get_final_response
from logs.chatbot_logger import ChatbotLogger

# --- Load ML Model Artifacts ---
MODEL_PATH = "model/model.pkl"
VEC_PATH = "model/vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VEC_PATH)
    except Exception:
        model, vectorizer = None, None
else:
    model, vectorizer = None, None

def predict_intent(user_input):
    """
    Validates, preprocesses, and predicts intent using the loaded model.
    """
    if model is None or vectorizer is None:
        return "System error: Model files missing.", 0.0, "error", [], True

    # 1. Validation & Sanitization
    is_valid, sanitized_input, error_msg = InputValidator.validate_and_sanitize(user_input)
    if not is_valid:
        return error_msg, 0.0, "invalid", [], True

    # 2. Entity Extraction
    entities = extract_all_entities(sanitized_input)
    
    # 3. Preprocessing for Vectorization
    clean_text = preprocess_text(sanitized_input)
    if not clean_text:
        return "I didn't quite catch that. Could you rephrase?", 0.0, "unknown", [], True

    # 4. ML Prediction
    user_vec = vectorizer.transform([clean_text])
    probabilities = model.predict_proba(user_vec)[0]
    max_prob = max(probabilities)
    prediction = model.classes_[probabilities.argmax()]
    
    # 5. Response Mapping
    response, is_fallback = get_final_response(prediction, max_prob, entities, sanitized_input)
    
    # 6. Centralized Logging
    ChatbotLogger.log_interaction(user_input, prediction, max_prob, entities, response, is_fallback)
    
    return response, max_prob, prediction, entities, is_fallback

# --- MODERN GUI SECTION ---
class ChatBotGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("University AI Assistant")
        self.master.geometry("550x700")
        self.master.configure(bg="#0F111A")
        
        # Theme
        self.BG_COLOR = "#0F111A"
        self.TEXT_BG = "#1A1D2D"
        self.USER_BG = "#0A84FF"
        self.BOT_BG = "#2B2F42"
        self.TEXT_COLOR = "#FFFFFF"

        # Chat Window
        self.chat_area = scrolledtext.ScrolledText(master, bg=self.BG_COLOR, fg=self.TEXT_COLOR, font=("Inter", 11), wrap=tk.WORD, borderwidth=0)
        self.chat_area.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        self.chat_area.config(state=tk.DISABLED)

        # Input Frame
        self.input_frame = tk.Frame(master, bg=self.BG_COLOR)
        self.input_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=20)

        self.user_input = tk.Entry(self.input_frame, bg=self.TEXT_BG, fg=self.TEXT_COLOR, font=("Inter", 12), borderwidth=0, insertbackground="white")
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10, padx=(0, 10))
        self.user_input.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message, bg=self.USER_BG, fg="white", font=("Inter", 10, "bold"), borderwidth=0, padx=20)
        self.send_button.pack(side=tk.RIGHT)

        self.display_message("System", "Bot is ready! Type a message to begin.")
        if not model:
            self.display_message("Warning", "Model files not found. Please run train.py first!")

    def display_message(self, sender, message):
        self.chat_area.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M")
        self.chat_area.insert(tk.END, f"[{timestamp}] {sender}: {message}\n\n")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)

    def send_message(self, event=None):
        msg = self.user_input.get().strip()
        if not msg: return
        
        self.user_input.delete(0, tk.END)
        self.display_message("You", msg)
        
        # Start typing animation/threading for response
        threading.Thread(target=self.process_response, args=(msg,)).start()

    def process_response(self, msg):
        time.sleep(0.5) # Simulate thinking
        response, prob, intent, entities, is_fallback = predict_intent(msg)
        self.display_message("Assistant", response)

if __name__ == "__main__":
    root = tk.Tk()
    gui = ChatBotGUI(root)
    root.mainloop()

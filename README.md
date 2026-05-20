# Student Information Chatbot

A specialized, university-specific conversational AI chatbot designed to help students with inquiries regarding registration, courses, schedules, fees, exams, contacts, locations, and scholarships. The chatbot uses a robust machine learning pipeline and a rule-based Named Entity Recognition (NER) system to provide accurate, reliable answers.

## Features

- **High-Accuracy Intent Classification:** Utilizes a custom Hybrid TF-IDF and Logistic Regression pipeline calibrated for confidence scores.
- **Rule-Based NER:** Extracts specific campus entities (departments, buildings, services) via regex and fuzzy matching to formulate context-aware responses.
- **Context Management:** Retains topic state across conversation turns, enabling natural follow-up questions (who, what, when, where).
- **Knowledge Base Integration:** Serves verified structural facts directly from `knowledge_base.json` to prevent hallucinations.
- **Out-of-Domain Detection:** Rejects unrelated questions with fallback responses.
- **Professional Analytics & Audit Tools:** Built-in tools for measuring macro F1 scores, confidence histograms, and evaluating against a human-provided test set.

## Project Structure

- `app.py`: Main interactive Chatbot loop and context manager.
- `ner.py`: Named Entity Recognition logic and templates.
- `responses.py`: Response generator handling template rendering and KB fact fetching.
- `context_manager.py`: Handles dialogue state and topic tracking.
- `train.py`: ML pipeline to train the Logistic Regression model.
- `preprocess.py`: NLP text cleaning and OOD keyword checking.
- `config.py`: Global chatbot thresholds, settings, and constants.
- `evaluate_real_world.py`: Generates performance reports against testing datasets.
- `run_audit_tests.py`: End-to-end automated testing script simulating dialogues.
- `knowledge_base.json`: The core university data structure.
- `intents.json`: Training patterns and default responses.
- `dataset/`: Contains `.csv` resources for training and evaluation.

## Installation

```bash
# Clone the project and set up a virtual environment
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) Download required NLTK resources for tokenization & lemmatization:
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

## Usage

**1. Train the Model**
If you have modified `intents.json` or `dataset.csv`, run the training script:
```bash
python train.py
```

**2. Start the Chatbot**
To run the interactive loop:
```bash
python app.py
```

**3. Evaluate Performance**
To run the analytical dashboard and test pipeline against `human_test_data.csv`:
```bash
python evaluate_real_world.py
```
To run end-to-end simulated dialogues to catch logic regressions:
```bash
python run_audit_tests.py
```

## Settings
Adjust `config.py` to change the strictness of the chatbot, confidence thresholds for specific intents, and enable the `CHATBOT_DEBUG` mode for transparent logging.

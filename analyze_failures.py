import os
import re
from collections import Counter

FAILURE_LOG = "logs/failures.txt"
SUGGESTIONS_FILE = "logs/dataset_suggestions.txt"

def analyze_failures():
    if not os.path.exists(FAILURE_LOG):
        print("Failure log not found. Run evaluation first.")
        return

    with open(FAILURE_LOG, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to extract Expected and Predicted
    # Format in log: EXPECTED: intent\nPREDICTED: intent (Conf: 0.xx)
    matches = re.findall(r"EXPECTED: (\w+)\nPREDICTED: (\w+)", content)
    
    if not matches:
        print("No failures found to analyze.")
        return

    confusions = Counter(matches)
    
    print("\n" + "="*40)
    print(" 🔎 FAILURE CLUSTER ANALYSIS")
    print("="*40)
    
    with open(SUGGESTIONS_FILE, "w", encoding="utf-8") as out:
        out.write("--- AUTOMATED DATASET IMPROVEMENT SUGGESTIONS ---\n\n")
        
        for (expected, predicted), count in confusions.most_common(5):
            print(f"❌ {expected} is often confused with {predicted} ({count} times)")
            out.write(f"CONFUSION: {expected} -> {predicted} ({count} cases)\n")
            out.write(f"ACTION: Add more '{expected}' samples containing keywords that distinguish it from '{predicted}'.\n")
            
            # Specific suggestions based on common university chatbot confusions
            if expected == "hostel" and predicted == "location":
                out.write("SUGGESTION: Add phrases like 'living in dorm', 'hostel room number', 'proctor name'.\n")
            elif expected == "schedule" and predicted == "exam":
                out.write("SUGGESTION: Add phrases like 'class routine', 'daily timetable', 'lecture start time'.\n")
            
            out.write("-" * 30 + "\n")

    print(f"\n✅ Analysis complete. Suggestions saved to {SUGGESTIONS_FILE}")

if __name__ == "__main__":
    analyze_failures()

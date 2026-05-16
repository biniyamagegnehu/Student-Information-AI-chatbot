import pandas as pd

# Final precision fixes
precision_fixes = [
    # Strengthening 'help'
    ("help me what can you do", "help"),
    ("what can you do for me?", "help"),
    ("can you show me your features", "help"),
    ("tell me about the chatbot", "help"),
    
    # Strengthening 'exam' with course context
    ("when is the software engineering exam?", "exam"),
    ("software engineering final exam date", "exam"),
    ("what is the date for the pharmacy exam?", "exam"),
    ("exam schedule for engineering students", "exam")
]

df_p = pd.DataFrame(precision_fixes, columns=['text', 'intent'])
df_p.to_csv('dataset/dataset.csv', mode='a', header=False, index=False)
print("Applied final precision fixes.")


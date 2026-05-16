import pandas as pd

# Final Dataset Polish: Adding Help intent and improving Schedule/Hostel
final_polish_data = [
    # Help / General Query Intent
    ("what can you do?", "help"),
    ("how can you help me?", "help"),
    ("tell me about your features", "help"),
    ("what kind of questions can i ask?", "help"),
    ("help me understand how to use this bot", "help"),
    ("show me available categories", "help"),
    ("what are your capabilities?", "help"),
    ("guide me on how to use you", "help"),
    ("what services do you provide?", "help"),
    ("can you explain what you do?", "help"),
    
    # Schedule Intent (Focusing on 'timetable', 'routine', 'weekly')
    ("i need my weekly class timetable", "schedule"),
    ("what is the lecture schedule for this month?", "schedule"),
    ("show me the departmental routine", "schedule"),
    ("at what time does the morning shift start?", "schedule"),
    ("is there a specific timetable for freshmen?", "schedule"),
    ("weekly schedule of software engineering", "schedule"),
    ("what are the lecture hours for today?", "schedule"),
    ("i want to see the semester routine", "schedule"),
    ("when is the lab session occurring?", "schedule"),
    ("daily class timings for regular students", "schedule"),
    
    # Resolving Exam/Schedule confusion (Exam = Assessment, Schedule = Timing)
    ("when is the midterm assessment?", "exam"),
    ("final exam date for engineering", "exam"),
    ("routine for exams", "exam"),
    ("test dates for this semester", "exam"),
    ("when will the finals start?", "exam")
]

df_polish = pd.DataFrame(final_polish_data, columns=['text', 'intent'])
df_polish.to_csv('dataset/dataset.csv', mode='a', header=False, index=False)
print(f"Final Polish: Added {len(final_polish_data)} high-quality samples.")


import pandas as pd
import os

new_data = [
    # Schedule improvements
    ("what is the class routine for today?", "schedule"),
    ("when does the first period start?", "schedule"),
    ("what time is the lecture over?", "schedule"),
    ("is there a timetable for the lab sessions?", "schedule"),
    ("when do classes occur on weekends?", "schedule"),
    ("academic schedule for the second semester", "schedule"),
    ("is the monday routine different?", "schedule"),
    ("when does the registration window close?", "registration"), # Supporting registration too
    
    # Hostel improvements
    ("who is the proctor for block a?", "hostel"),
    ("is there laundry in the dorm?", "hostel"),
    ("how many students per room in the hostel?", "hostel"),
    ("i need to move into the dormitory", "hostel"),
    ("hostel assignment for freshmen", "hostel"),
    ("where is the student residence gate?", "hostel"),
    ("dormitory rules and regulations", "hostel"),
    ("living in campus accommodation", "hostel")
]

df_new = pd.DataFrame(new_data, columns=['text', 'intent'])
df_new.to_csv('dataset/dataset.csv', mode='a', header=False, index=False)
print(f"Added {len(new_data)} targeted samples to dataset.csv")


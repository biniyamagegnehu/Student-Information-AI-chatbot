import pandas as pd

# Targeted data to resolve specific confusion clusters
targeted_fixes = [
    # Schedule vs Holidays (Emphasize weekly/daily routine vs one-time closures)
    ("what is the weekly routine for software engineering?", "schedule"),
    ("what time do lectures usually start in the morning?", "schedule"),
    ("is the timetable the same every day of the week?", "schedule"),
    ("do we have classes every monday at 9am?", "schedule"),
    ("what is the duration of a single lecture period?", "schedule"),
    ("lecture start and end times for freshmen", "schedule"),
    ("is the university closed for the public holiday tomorrow?", "holidays"),
    ("will there be classes during the eid vacation?", "holidays"),
    ("dates for the upcoming university summer break", "holidays"),
    ("is campus closing for the national ceremony next week?", "holidays"),
    
    # Admission vs Registration (Emphasize joining vs signing up for classes)
    ("how can a new student apply for admission to the university?", "admission"),
    ("what are the entrance exam requirements for new applicants?", "admission"),
    ("when will the acceptance letters for new students be sent out?", "admission"),
    ("i want to join the masters program, how do i apply?", "admission"),
    ("is the registration portal open for current students to add courses?", "registration"),
    ("how do i register for my semester modules online?", "registration"),
    ("deadline for adding or dropping a course this semester", "registration"),
    ("i need to sign up for my electives in the portal", "registration"),
    
    # Fees vs Registration
    ("how much do i need to pay for the registration fee?", "fees"),
    ("is the tuition payment due before or after course registration?", "fees"),
    ("where can i find the bank account for my registration payment?", "fees"),
    
    # Robustness (Addressing the 'banana' / 'greeting' overlap)
    # Adding more varied greetings to strengthen the 'greeting' cluster
    ("hello there campus bot", "greeting"),
    ("hi, i need some info about the uni", "greeting"),
    ("hey assistant, how are you today?", "greeting"),
    ("good day, are you available for questions?", "greeting")
]

df_fixes = pd.DataFrame(targeted_fixes, columns=['text', 'intent'])
df_fixes.to_csv('dataset/dataset.csv', mode='a', header=False, index=False)
print(f"Successfully added {len(targeted_fixes)} targeted fixes to dataset.csv")

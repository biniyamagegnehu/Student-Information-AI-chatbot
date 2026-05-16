import sqlite3
import os

# --- UNIVERSITY DATABASE MANAGER ---
# This separates the data layer from the chatbot logic (ML pipeline).
# It allows us to scale later by replacing SQLite with a REST API or Postgres DB.

DB_NAME = "university_data.db"

def init_db():
    """Initializes the SQLite database with sample university data."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # --- TABLE CREATION (SCHEMA) ---
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS locations (
            entity_name TEXT PRIMARY KEY,
            location_desc TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fees (
            department TEXT PRIMARY KEY,
            fee_amount TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contacts (
            entity_name TEXT PRIMARY KEY,
            contact_info TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS schedules (
            entity_name TEXT PRIMARY KEY,
            schedule_info TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS general_info (
            topic TEXT PRIMARY KEY,
            info TEXT
        )
    ''')

    # --- SAMPLE DATA INJECTION (CRUD READY) ---
    # Locations
    locations_data = [
        ("library", "The main library is located in Block 5 near the main gate."),
        ("registrar", "The registrar office is in the Admin Building, Ground Floor."),
        ("reg office", "The registrar office is in the Admin Building, Ground Floor."),
        ("engineering", "The engineering department is located in Block 3."),
        ("computer science", "The computer science department is in Block 1, second floor."),
        ("cs", "The computer science department is in Block 1, second floor."),
        ("cs dept", "The computer science department is in Block 1, second floor."),
        ("block c", "Block C is located near the eastern gate, next to the sports field."),
        ("transcript", "You can pick up your transcript from the main registrar office.")
    ]
    cursor.executemany("INSERT OR REPLACE INTO locations VALUES (?, ?)", locations_data)
    
    # Fees
    fees_data = [
        ("engineering", "The tuition fee for Engineering is 15,000 Birr per semester."),
        ("computer science", "The tuition fee for Computer Science is 14,500 Birr per semester."),
        ("transcript", "Transcripts cost 50 Birr per copy. Please pay at the registrar."),
        ("default", "Standard tuition fees vary by program. Please check the student portal.")
    ]
    cursor.executemany("INSERT OR REPLACE INTO fees VALUES (?, ?)", fees_data)
    
    # Contacts
    contacts_data = [
        ("engineering", "The head of the engineering department can be reached at eng_head@university.edu or 555-0102."),
        ("computer science", "The CS department contact is cs_admin@university.edu or 555-0103."),
        ("registrar", "You can reach the registrar at info@university.edu or call the toll-free center."),
        ("professor john", "Professor John's office is Room 204. Email: john.doe@university.edu.")
    ]
    cursor.executemany("INSERT OR REPLACE INTO contacts VALUES (?, ?)", contacts_data)
    
    # Schedules
    schedules_data = [
        ("library", "The library is open from 8:00 AM to 10:00 PM on weekdays, and closes at 5:00 PM on weekends."),
        ("registrar", "The registrar office is open from 9:00 AM to 4:00 PM."),
        ("software engineering", "The Software Engineering exam is scheduled for June 15th at 9:00 AM in Block C."),
        ("registration", "The registration deadline for this semester is October 30th.")
    ]
    cursor.executemany("INSERT OR REPLACE INTO schedules VALUES (?, ?)", schedules_data)
    
    # General Info
    general_data = [
        ("hostel availability", "Hostel rooms are currently available for freshmen in Dorm Block A."),
        ("registration deadline", "The registration deadline for this semester is October 30th."),
        ("admission", "Admission requires passing the national university entrance exam. Check the MOE portal.")
    ]
    cursor.executemany("INSERT OR REPLACE INTO general_info VALUES (?, ?)", general_data)
    
    conn.commit()
    conn.close()

# --- DYNAMIC RETRIEVAL FUNCTIONS ---
def get_location(entity):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT location_desc FROM locations WHERE entity_name=?", (entity,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def get_fee(entity):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT fee_amount FROM fees WHERE department=?", (entity,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def get_contact(entity):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT contact_info FROM contacts WHERE entity_name=?", (entity,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def get_schedule(entity):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT schedule_info FROM schedules WHERE entity_name=?", (entity,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def get_general_info(topic):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT info FROM general_info WHERE topic=?", (topic,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

# Initialize DB when the module is imported
init_db()

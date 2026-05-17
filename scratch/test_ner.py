# scratch/test_ner.py
import sys
import os

# Add parent directory to sys.path to import ner
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ner import extract_entities

test_cases = [
    ("where is software engineering", [('DEPARTMENT', 'software engineering')]),
    ("wher is libary", [('BUILDING', 'library')]),
    ("when is exam this week", [('SERVICE', 'exam'), ('DATE', 'this week')])
]

print("Running STANDALONE NER Test cases...")
all_passed = True
for query, expected in test_cases:
    result = extract_entities(query)
    print(f"\nQuery: '{query}'")
    print(f"Expected: {expected}")
    print(f"Got     : {result}")
    if result == expected:
        print("Result: PASSED")
    else:
        print("Result: FAILED")
        all_passed = False

if all_passed:
    print("\nALL STANDALONE TESTS PASSED!")
else:
    print("\nSOME TESTS FAILED!")

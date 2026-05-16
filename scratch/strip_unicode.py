import os
import re

def strip_non_ascii(text):
    return re.sub(r'[^\x00-\x7f]', '', text)

for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            clean_content = strip_non_ascii(content)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(clean_content)
print("Stripped non-ASCII from all .py files.")

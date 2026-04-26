import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

load_dotenv()
client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_PATH = BASE_DIR / "/logs/."

def find_log_files(root):
    logs = []
    for dirpath, _, filenames in os.walk(root):
        for file in filenames:
            if file.endswith(".log"):
                logs.append(os.path.join(dirpath, file))
    return logs

def read_file(path):
    with open(path, "r", errors="ignore") as f:
        return f.read()
    
def llm(prompt):
    response = client.chat.completions.create(
        model = "gpt-4.1-mini",
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Example usage
log_files = find_log_files(BASE_DIR / ".")

for log in log_files:
    content = read_file(log)

    summary = llm("Summarize errors:\n" + content)
    print(f"\n {log}\n{summary}")

"""
    Summary of errors from the log:

    1. **2026-04-25 10:15:34** - Database connection failed due to timeout.
    2. **2026-04-25 10:16:03** - Authentication failed for user "admin".
    3. **2026-04-25 10:17:45** - Failed to load resource `/api/data` with a 500 Internal Server Error.
    4. **2026-04-25 10:18:30** - File not found: `config.yaml`.
"""
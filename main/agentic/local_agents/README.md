### Local Agents

Those agents are focused on building real, useful local agents.   
You can actually build your won mini Copilot/Cursor locally.  


### 1. File System Agent

A file system agent can read, search, and modify files on you computer.  
It lets an AI understand your folders, logs, or notes and act on them.  

It is the foundation for things like:
"Explain my logs"  
"Search all files for bugs" 

Log file example: 

~~~sh
2026-04-25 10:15:32 INFO Starting application
2026-04-25 10:15:33 INFO Connecting to database
2026-04-25 10:15:34 ERROR Database connection failed: timeout
2026-04-25 10:15:35 WARN Retrying connection (attempt 1)
2026-04-25 10:15:37 INFO Connection established
2026-04-25 10:16:02 INFO User login attempt: user=admin
2026-04-25 10:16:03 ERROR Authentication failed for user=admin
2026-04-25 10:16:10 INFO User login attempt: user=test_user
2026-04-25 10:16:11 INFO Authentication successful for user=test_user
2026-04-25 10:17:45 ERROR Failed to load resource: /api/data (500 Internal Server Error)
2026-04-25 10:18:01 WARN Disk usage at 85%
2026-04-25 10:18:30 ERROR File not found: config.yaml
2026-04-25 10:19:00 INFO Shutting down application
~~~

Agent code:

~~~python
import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

load_dotenv()
client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_PATH = BASE_DIR / "."

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
    3. **2026-04-25 10:17:45** - Failed to load resource `/api/data` with a 500.
    4. **2026-04-25 10:18:30** - File not found: `config.yaml`.
"""
~~~

### 2. Command Execution (Safe)

Agents can run shell commands (like git, npm, ls).  
You must restrict them to safe commands to avoid dangerous execution.  

~~~python
import os
from dotenv import load_dotenv
from openai import OpenAI
import subprocess

load_dotenv()
client = OpenAI()

ALLOWED_COMMANDS = ["ls", "pwd", "git status"]
REPO_DIR = "/var/docker/minte9/m9github/"

def llm(prompt):
    response = client.chat.completions.create(
        model = "gpt-4.1-mini",
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def run_command(cmd):
    if not any(cmd.startswith(allowed) for allowed in ALLOWED_COMMANDS):
        return "Command not allowed"
    
    try:
        result = subprocess.check_output(
            cmd, 
            shell=True, 
            text=True, 
            cwd=REPO_DIR
        )
        return result
    except Exception as e:
        return str(e)

# Example usage
user_input = "pwd"
output = run_command(user_input)
response = llm(f"Explain this output:\n{output}")
print(response)

"""
    The text you provided:
    /var/docker/minte9/m9github

    appears to be a filesystem path, likely from a Unix-like operating system .
    Here's a breakdown of what this path represents:

    - /var
    This is a standard directory in Unix-like systems. 
    This could include logs, databases, cache files. 

    - /var/docker
    This suggests a directory related to Docker, a platform for running containers. 
    Docker stores data related to containers.  
"""
~~~

### 3. Refactor Function

Given a function, AI improves readability, performance and structure.

~~~python
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

code = """
def f(x):
    return [i*2 for i in x if i%2==0]
"""

prompt = """
Refactor this code:
- improve readability
- add comments
"""

def llm(prompt):
    response = client.chat.completions.create(
        model = "gpt-4.1-mini",
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

new_code = llm(prompt + code)
print(new_code)

"""
Here's the refactored code with improved readability and comments added:

def double_even_numbers(numbers):

    # Given a list of integers, return a new list containing
    # the double of each even number in the original list.

    # :param numbers: List of integers
    # :return: List of integers (each even number doubled)
    
    doubled_evens = []
    for number in numbers:
        if number % 2 == 0:  # Check if the number is even
            doubled_evens.append(number * 2)
    return doubled_evens

**Explanation:**
- Renamed the function and parameter to more descriptive names.
- Added a docstring describing the function's purpose and parameters.
- Used a standard for-loop with explicit conditions for clarity.
- Added an inline comment inside the loop for the even check.
"""
~~~
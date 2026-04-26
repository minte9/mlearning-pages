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

appears to be a filesystem path, likely from a Unix-like operating system such as Linux.
Here's a breakdown of what this path represents:

- /var
This is a standard directory in Unix-like systems usually used to store variable data files. 
This could include logs, databases, cache files, and other files that are expected to change. 

- /var/docker
This suggests a directory related to Docker, a platform for developing, shipping, and running containers. 
Docker typically stores data related to containers, images, volumes, and other artifacts under /var/lib/docker
by default, but /var/docker could be a custom or alternative location.
"""
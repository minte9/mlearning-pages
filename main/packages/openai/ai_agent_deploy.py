""" AI Agent that orchestrates Git operations and FTP deployment based on natural language input. 
Intelligent agent that can interpret your commands like:
 - "Export all differences to FTP and GitHub"
 - "Push only python repo updates"
 - "Sync algorithms and PHP pages"
"""

import openai
import os
import subprocess
import json
import datetime

from dotenv import load_dotenv
load_dotenv()  

# Setup OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


# Define valid repositories and their paths
REPOS = {
    "python":       "/var/www/refresh.local/refresh.ro/Application/github/python-pages/",
    "algorithms":   "/var/www/refresh.local/refresh.ro/Application/github/algorithms-pages/",
    "php":          "/var/www/refresh.local/refresh.ro/Application/github/php-pages/",
    "mlearning":    "/var/www/refresh.local/refresh.ro/Application/github/mlearning-pages/",
    "java":         "/var/www/refresh.local/refresh.ro/Application/github/java-pages/"
}

# Define base FTP path and credentials (preferably load from environment)
FTP_BASE = os.getenv("FTP_BASE")
FTP_USER = os.getenv("FTP_USER")
FTP_PASS = os.getenv("FTP_PASS")


def get_action_plan(natural_language_cmd):
    """Use OpenAI to interpret the user command and return repo and ftp instructions."""

    system_prompt = f"""
        You are an AI agent that converts deployment commands into structured JSON instructions.
        
        Valid repositories:
        {',' . join(REPOS.keys())}

        Return a JSON object with:
        - "git": list of repositories to update (subset of the valid ones)
        - "ftp": list of diretories to upload via FTP (same names)

        Examples:
        User: Export only python repo differences to GitHub
        Response: {{ "git": ["python"], "ftp": [] }}

        User: Upload java and php to FTP only
        Response: {{ "git": [], "ftp": ["java", "php"] }}

        User: Export all differences to FTP and GitHub
        Response: {{ 
            "git": ["python", "algorithms", "php", "mlearning", "java"], 
            "ftp": ["python", "algorithms", "php", "mlearning", "java"] 
        }}
    """

    completion = openai.ChatCompletion.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": f"Command: {natural_language_cmd}"}
        ]
    )

    response_text = completion.choices[0].message.content.strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print("Error parsing OpenAI response:", e)
        print("Raw response:", response_text)
        return None

def perform_git(repo_name):
    """Pull, commit, and push changes in the specific repo."""
    repo_path = REPOS[repo_name]
    print(f"🌐 Updating Github repo: {repo_name}")
    os.chdir(repo_path)
    subprocess.run(["git", "pull", "origin", "main", "--force"])
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-am", f"{repo_name}-pages update"])
    subprocess.run(["git", "push", "origin", "main"])


def get_today_date():
    from datetime import datetime
    return datetime.today().strftime('%Y-%m-%d')

def get_changed_files(repo_path):
    """Return list of changed files (repo_path) from Git"""
    os.chdir(repo_path)

    try:
        result = subprocess.run(
            ["git", "diff", "--stat", f"@{{{get_today_date()}}}", "--diff-filter=ACRMRT", "--name-only"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        changed_files = result.stdout.strip().split('\n')
        return [f for f in changed_files if f.strip()]
    except subprocess.CalledProcessError as e:
        print("Git diff failed:", e.stderr)
        return []

def perform_ftp(repo_name):
    """Upload changed files to the hosting corresponding FTP Path"""
    local_repo_path = REPOS[repo_name]
    remote_path = f"{FTP_BASE}{repo_name}-pages"

    print(f"🌐 Uploding {repo_name} files to FTP ...")

    changed_files = get_changed_files(local_repo_path)

    if not changed_files:
        print("No changed files to upload")
        return

    for rel_path in changed_files:
        local_file = os.path.join(local_repo_path, rel_path)
        remote_file = f"{remote_path}/{rel_path}"

        # Skip if file doesn't exist (deleted, moved, etc.)
        if not os.path.isfile(local_file):
            continue

        print(f"- {rel_path}")
        subprocess.run(["curl",  "-T", local_file, remote_file, "--user", f"{FTP_USER}:{FTP_PASS}"])


def main():
    user_command = input("What should I do? \n> ").strip()
    #user_command = "Upload mlearning to ftp"

    """
        User commands examples: 
        "Export only python repo differences to GitHub"
        "Upload java to FTP"
        "Export all differences to FTP and GitHub"
        "Go to school"

        Action plan responses:
        {'git': ['python'], 'ftp': []}
        {'git': [], 'ftp': ['java']}
        {'git': ['python', 'algorithms', 'php', 'mlearning', 'java'], 
         'ftp': ['python', 'algorithms', 'php', 'mlearning', 'java']}
        {'git': [], 'ftp': []}

    """

    action_plan = get_action_plan(user_command)

    if not action_plan:
        print("❌ No valid action plan. Aborting.")
        return

    print(f"Action plan: {action_plan}")

    for repo in action_plan.get("git", []):
        if repo in REPOS:
            perform_git(repo)

    for repo in action_plan.get("ftp", []):
        if repo in REPOS:
            perform_ftp(repo)

    print("All tasks completed.")


if __name__ == '__main__':
    main()



"""
To run it more cleanly from anywhere:
Create a shell script or symlink:
    # Inside /usr/local/bin/deployai (or somewhere in PATH)
    #!/bin/bash
    python3 /path/to/deploy_agent.py "$@"

Make it executable:
    chmod +x /usr/local/bin/deployai
"""
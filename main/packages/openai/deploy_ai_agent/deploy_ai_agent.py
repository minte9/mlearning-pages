""" AI Agent that orchestrates Git and FTP operations based on natural language input.
The agent can interpret your commands like:

 - "Export all differences to FTP and GitHub"
 - "Push only python repo updates"
 - "Sync algorithms and PHP pages"
"""

from openai import OpenAI
import os
import subprocess
import json
import datetime
import sys
import sqlite3

from dotenv import load_dotenv
load_dotenv()

# OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define valid repositories and their paths
REPOS = {
    "python":       "/var/www/refresh.local/refresh.ro/Application/github/python-pages/",
    "algorithms":   "/var/www/refresh.local/refresh.ro/Application/github/algorithms-pages/",
    "php":          "/var/www/refresh.local/refresh.ro/Application/github/php-pages/",
    "mlearning":    "/var/www/refresh.local/refresh.ro/Application/github/mlearning-pages/",
    "java":         "/var/www/refresh.local/refresh.ro/Application/github/java/",
    "javascript":   "/var/www/refresh.local/refresh.ro/Application/github/javascript-pages/",
    "kotlin":       "/var/www/refresh.local/refresh.ro/Application/github/kotlin-pages/",
    "spring-boot":  "/var/www/refresh.local/refresh.ro/Application/github/spring-boot/"
}

# Define base FTP path and credentials (preferably load from environment)
FTP_BASE = os.getenv("FTP_BASE")
FTP_USER = os.getenv("FTP_USER")
FTP_PASS = os.getenv("FTP_PASS")

# SQLite persistence
DB_PATH = os.getenv("APP_DIR") + "prompt_cache.db"

def init_db():
    """Initialize SQLite DB and create table if not exists"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS prompt_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT UNIQUE,
            response TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_cached_response(prompt):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT response FROM prompt_cache WHERE prompt = ?", (prompt,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def store_response(prompt, response):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO prompt_cache (prompt, response) VALUES (?,?)", (prompt, response))
        conn.commit()
    except sqlite3.IntegrityError:
        # Already exists - ignore 
        pass
    finally:
        conn.close()

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

    cached = get_cached_response(natural_language_cmd)
    if cached:
        print("📦 Using cached response from SQLite")
        response_text = cached
    else:
        print("🧠 Sending to OpenAI...")
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": f"Command: {natural_language_cmd}"}
            ]
        )
        response_text = response.choices[0].message.content.strip()
        store_response(natural_language_cmd, response_text)

    # Parse json
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

    if repo_name == 'java' or repo_name == 'spring-boot':
        remote_path = f"{FTP_BASE}{repo_name}"
    else:
        remote_path = f"{FTP_BASE}{repo_name}-pages"

    print(f"🌐 Uploding {repo_name} files to FTP ...")

    changed_files = get_changed_files(local_repo_path)

    if not changed_files:
        print("No changed files to upload")
        return

    for rel_path in changed_files:
        local_file = os.path.join(local_repo_path, rel_path)
        remote_file = f"{remote_path}/{rel_path}"

        #print(local_file, remote_file); sys.exit(0)

        # Skip if file doesn't exist (deleted, moved, etc.)
        if not os.path.isfile(local_file):
            continue

        print(f"- {rel_path}")
        subprocess.run(["curl",  "-T", local_file, remote_file, "--user", f"{FTP_USER}:{FTP_PASS}"])


def main():
    init_db()

    if len(sys.argv) > 1:
        # Command passed as CLI argument (e.g. deployai "Sync my php repo")
        user_command = sys.argv[1].strip()
    else:
        # Interactive fallback
        user_command = input("What should I do? \n> ").strip()

    action_plan = get_action_plan(user_command)

    if not action_plan:
        print("No valid action plan. Aborting.")
        return

    print(f"Action plan: {action_plan}")

    for repo in action_plan.get("ftp", []):
        if repo in REPOS:
            perform_ftp(repo)

    for repo in action_plan.get("git", []):
        if repo in REPOS:
            perform_git(repo)

    print("✅ All tasks completed.")


if __name__ == '__main__':
    main()

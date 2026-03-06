# Deploy AI Agent

An AI-powered assistant that syncs Git repositories and uploads code changes via FTP.  
All controlled through natural language commands.  

No more committing to temporary folders, editing shell scripts, or uncommenting blocks.  
Just tell the agent what to do.  

---

## Features

- Understands natural language commands using OpenAI GPT
- Detects changed files per repo using `git diff`
- Uploads only changed files to FTP (no full syncs)
- Git operations: pull, add, commit, push — per project
- Clean, interactive CLI interface
- Secure: uses environment variables for secrets

---

## Installation

### 1. Clone the project and install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your environment variables

Create a .env file or export these manually:

```bash
export OPENAI_API_KEY=your_openai_api_key
export FTP_BASE=your_ftp_directory
export FTP_USER=your_ftp_username
export FTP_PASS=your_ftp_password
```

### 3. Install as a CLI tool (globally, editable)

In the project root (where setup.py is located), run:

    pip install -e .


This will install the `deployai` command system-wide (in editable mode),   
so you can run it from anywhere on your system.  

When moving directory, it should be reinstalled.  
Install it as your user, not root.  

    cd /var/newdir/
    pip uninstall deploy-ai-agent -y
    pip install -e . --user
    pip install -e . --user --config-settings editable_mode=compat



🔁 If you make changes to the Python file, they take effect immediately — no reinstall needed.

### How It Works

The agent parses your command with GPT-4 and maps it to valid repo actions:  

"git": list of repos to update with git pull/add/commit/push  
"ftp": list of repos to upload changed files via FTP  

Detects changed files in each repo using git diff.  
Uploads only those files via curl.  


### Example Interaction

```bash
$ deployai
What should I do? > Push only python and java changes to GitHub

🌐 Updating GitHub repo: python
🌐 Updating GitHub repo: java

All tasks completed.
```
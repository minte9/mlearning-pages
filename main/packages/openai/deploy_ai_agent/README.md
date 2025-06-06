# 🚀 Deploy AI Agent

An AI-powered assistant that syncs Git repositories and uploads code changes via FTP.  
All controlled through natural language commands.  

No more committing to temporary folders, editing shell scripts, or uncommenting blocks.  
Just tell the agent what to do.  

---

## ✨ Features

- 🧠 Understands natural language commands using OpenAI GPT
- 🗃️ Detects changed files per repo using `git diff`
- 📤 Uploads only changed files to FTP (no full syncs)
- 🛠 Git operations: pull, add, commit, push — per project
- 💬 Clean, interactive CLI interface
- 🔐 Secure: uses environment variables for secrets

---

## 📦 Installation

### 1. Clone the project

```bash
git clone https://github.com/yourname/deploy-ai-agent.git
cd deploy-ai-agent
```

## 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your environment variables

Create a .env file or export these manually:

```bash
export OPENAI_API_KEY=your_openai_api_key
export FTP_USER=your_ftp_username
export FTP_PASS=your_ftp_password
```

### 4. Make it globally accessible (optional)

```bash
chmod +x deploy_agent.py
ln -s "$(pwd)/deploy_agent.py" /usr/local/bin/deployai
```

Then enter commands like:

```
Export only python repo differences to GitHub
Upload java repo to FTP
Export all changes to GitHub and FTP
Sync php and mlearning only
```

### 🧠 How It Works

Parses your command with GPT-4.  

Maps it to valid repo actions:  

"git": list of repos to update with git pull/add/commit/push  
"ftp": list of repos to upload changed files via FTP  

Detects changed files in each repo using git diff.  
Uploads only those files via curl.  



### 💡 Example Interaction

```bash
$ deployai
What should I do? > Push only python and java changes to GitHub

🌐 Updating GitHub repo: python
🌐 Updating GitHub repo: java

✅ All tasks completed.
```
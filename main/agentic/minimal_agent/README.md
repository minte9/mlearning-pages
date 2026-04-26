### Minimal Agent

Memory Learning Agent.    
What this agent does:

- Load a .md file (your notes)
- Extract concepts
- Ask you questions
- Evaluates you answers (with LLM)
- Keeps weak topics in rotation

### 1. Foler Structure

~~~sh
project/
│── notes.md
│── agent.py
│── memory.json
~~~

### 2. Example notes

~~~sh
# Python - Functions
A function is a reusable block of code.

# Python - Lists
A list is a collection of items that is ordered and mutable.

# Python - Dictionary
A dictionary stores key-value pairs.
~~~

### 3. Code Agent

This is intentionally simple but actually `works` as an agent loop.  

~~~python
import json
import random
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

load_dotenv()       # loads .env into environment
client = OpenAI()   # automatically read OPENAI_API_KEY

# JSON persistence
BASE_DIR = Path(__file__).resolve().parent
MEMORY_FILE = BASE_DIR / "memory.json"
NOTES_FILE = BASE_DIR / "notes.md"


# ---------------------
# Agent Loop
# ----------------------
def run_agent():
    topics = load_notes()
    memory = load_memory()

    while True:
        topic = select_topic(topics, memory)

        print(f"\n Topic: {topic['title']}")

        # THINK
        question = generate_question(topic)

        # ACT
        print(f"\n {question}")
        user_answer = input("Your answer: ")

        # OBSERVE
        evaluation = evaluate_answer(topic, question, user_answer)
        print("\n Feedback:")
        print(evaluation)

        # UPDATE MEMORY
        try:
            score_line = [line for line in evaluation.split("\n") if "score" in line][0]
            score = float(score_line.split(":")[1].strip())
        except:
            score= 0.5

        memory[topic["title"]] = score
        save_memory(memory)

        # CONTINUE?
        cont = input("\nContinue: (y/n): ")
        if cont.lower() != "y":
            break


# ------------------------------
# Utils
# ------------------------------

# Load notes
def load_notes():
    with open(NOTES_FILE) as f:
        content = f.read()

    topics = []
    sections = content.split("# ")

    for s in sections:
        s = s.strip()
        
        if not s: continue
        if "\n" not in s: continue

        title, body = s.split("\n", 1)
        topics.append({
            "title": title.strip(),
            "content": body.strip()
        })

    return topics

# Load memory
def load_memory():
    try:
        with open(MEMORY_FILE) as f:
            return json.load(f)
    except:
        return {}

# Save memory
def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

# Select topic (focus on week ones)
def select_topic(topics, memory):
    weights = []
    for t in topics:
        score = memory.get(t["title"], 0.5)
        weight = max(0.01, 1 - score)   # prevent zero
        weights.append(weight)          # weaker = higher change

    return random.choices(topics, weights=weights)[0]


# --------------------------------------------
# Openai Prompts
# --------------------------------------------

# Ask LLM
def ask_llm(prompt):
    response = client.chat.completions.create(
        model = "gpt-4.1-mini",
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Generate question
def generate_question(topic):
    prompt = f"""
    Create a simple question to test this topic:

    {topic["title"]}

    Only return the question.
    """
    return ask_llm(prompt)

# Evaluate answer
def evaluate_answer(topic, question, user_answer):
    prompt = f"""
    Concept:
    {topic["title"]}

    Question:
    {question}

    User answer:
    {user_answer}

    Grade the answer on a scale from 0 to 1:
    - 1.0 = perfectly correct
    - 0.7-0.9 = mostly correct, minor mistake
    - 0.4-0.6 = partially correct, some understanding
    - 0.1-0.3 = mostly incorrect but relevant
    - 0.0 = completely incorrect

    IMPORTANT:
    - Do NOT use only 0 or 1 unless absolutely necessary
    - Use decimals like 0.6, 0.8, etc.

    Return EXACTLY:
    score: <number>
    feedback: <short explanation>
    """
    return ask_llm(prompt)

if __name__ == "__main__":
    run_agent()
~~~


### 4. Memory System (simple)

The memory.json will look like:

~~~json
{
  "Python - Functions": 0.9,
  "Python - Lists": 0.4,
  "Python - Dictionary": 0.5
}
~~~

### Cost Estimation

Using:
- gpt-4.1-mini
- 5-10 calls per section
- daily 30 min usage = $1-3/month
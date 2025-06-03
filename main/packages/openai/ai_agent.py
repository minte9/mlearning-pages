import openai
import os, sys

from dotenv import load_dotenv
load_dotenv()  

# Setup OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Query openai
def query_chatgpt(prompt):
    print("\n" + prompt)

    response = openai.ChatCompletion.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
    )

    return response['choices'][0]['message']['content'].strip()

# Input
topic = "function"

languages = {
    "Python": f"Explain the '{topic}' topic in Python. ",
    "PHP": f"Explain the concept similar to {topic} in PHP. ",
    "Java": f"Explain the equivalent concept of {topic} in Java. ",
}

context = "Keep the answer short, without examples."

for lang, prompt in languages.items():

    summary = query_chatgpt("--- " + prompt + context)
    
    # AI Agent part 
    # The agent decides what to do next based on results
    prompt_2 = f"How difficult is the topic '{topic}' in {lang} for beginners? Just reply with a number, 1 to 5."
    difficulty = query_chatgpt(prompt_2)

    # Output
    print("Difficulty: " + difficulty + "\n")
    print(summary)

    if int(difficulty) >= 3:
        code = query_chatgpt(f"Give a code example of {topic} in {lang}")
        qna = query_chatgpt(f"Create 3 beginner questions and answers about {topic} in {lang}.")

        # Output
        print(code)
        print(qna)
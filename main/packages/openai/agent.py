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
    "Python": f"Explain the {topic} concept in Python. ",
    "PHP": f"Explain the concept similar to {topic} in PHP. ",
    "Java": f"Explain the equivalent concept of {topic} in Java. ",
}

context = "Keep the summary short, without code or examples."

for lang, prompt in languages.items():

    summary = query_chatgpt(prompt + context)
    
    # AI agent part
    code = query_chatgpt(f"Give a code example of {topic} in {lang}")
    qna = query_chatgpt(f"Create 3 beginner questions and answers about {topic} in {lang}.")

    # Output
    print(summary)
    print(code)
    print(qna)

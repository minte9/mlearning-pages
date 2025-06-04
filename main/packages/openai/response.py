import openai
import os
from dotenv import load_dotenv

# Loads variables from .env into environment
load_dotenv()  

# Setup OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

question = "What is flask python?"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": question}],
    max_tokens=256,
    n=1,
    stop=None,
    temperature=0.7
)

answer = response['choices'][0]['message']['content']

print(question)
print(answer)

"""
What is flask python?
Flask is a lightweight web framework written in Python. 
It is used to build web applications quickly and easily. 
"""
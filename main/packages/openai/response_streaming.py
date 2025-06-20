""" Streaming with the OpenAI API allows you to get partial results 
and process them as they become available, which is more efficient and responsive. 
"""

import openai
import os
from dotenv import load_dotenv

# Loads variables from .env into environment
load_dotenv()  

# Setup OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# User question
print("Please enter your question (press Enter to submit):")
question = input()

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": question}],
    max_tokens=256,
    n=1,
    stop=None,
    temperature=0.7,
    stream=True
)

for chunk in response:
    content = chunk["choices"][0]["delta"].get("content", "")
    print(content, end="", flush=True)
    
print("\n")

"""
    Please enter your question (press Enter to submit):
    What is Flask?
    Flask is a lightweight and flexible web application framework for Python. 
    It is designed to make it easy to build web applications ... 
"""
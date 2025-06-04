""" You can add a context message at the start of the conversation 
in order to instruct the model. 
"""

import openai
import os, sys

# Setup OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Initialize conversation history
conversation_history = []

# Context for keeping answers short
context_message = {
    "role": "system",
    "content": "System: Please keep the answers short"
}
conversation_history.append(context_message)

# Question stream
questions = [
    "What is Flask? Keep the answers short.",
    "What's the current version?"
]

for question in questions:
    print("\nQuestion:", question)

    # Add user question to conversation history
    conversation_history.append({"role": "user", "content": question})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history,
        max_tokens=64,
        n=1,
        stop=None,
        temperature=0.7,
        stream=True
    )

    for chunk in response:
        content = chunk["choices"][0]["delta"].get("content", "")
        print(content, end="", flush=True)

    # Add API response to converstion history
    conversation_history.append({"role": "system", "content": content})

"""
    Question: What is Flask? Keep the answers short.
    Flask is a lightweight web framework for Python.
    Question: What's the current version?
    The current version of Flask is 1.1.2
"""
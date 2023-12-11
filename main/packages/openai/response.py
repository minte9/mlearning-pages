import openai
import os

# Setup OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

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
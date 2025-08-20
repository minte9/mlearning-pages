""" LLMChain that preprocess a text by follogin a given sequence of steps.
If a step has the value no, it shouldn't be performed.
"""

from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI   
from langchain.chains import LLMChain

import os
from dotenv import load_dotenv
load_dotenv()

prompt = PromptTemplate.from_template("""
Process the given text by following the given steps in sequence. 
Follow only the steps that have a 'yes' as value. Remove Number:{number}, 
Remove Punctuation: {punctuation}, Word stemming: {stemming}. Output just
the preprocessed text.Text:{text}
""")
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
chain = LLMChain(llm=llm, prompt=prompt)

answer = print(chain.run({
    'text': 'I have answered correctly to 7 out of 10 questions!',
    'number': 'yes',
    'punctuation': 'yes',
    'stemming': 'no'
}))
print(answer)
"""
    I have answered correctly to out of questions
"""
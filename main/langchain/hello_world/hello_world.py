from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI   
from langchain.chains import LLMChain

import os
from dotenv import load_dotenv
load_dotenv()

prompt = PromptTemplate.from_template("Suggest {number} names for a {domain} website")
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run({'number': 5, 'domain': 'machine learning'})
print(response)
"""
    1. IntelliLearn
    2. DataBrains
    3. ML Mastermind
    4. PredictiveIQ
    5. Algorithmic Academy
"""

response = chain.run({'number': 5, 'domain': 'online games'})
print(response)
"""
    1. GameSphere
    2. PlayHaven
    3. GamingZone
    4. VirtualArcade
    5. FunQuest 
"""
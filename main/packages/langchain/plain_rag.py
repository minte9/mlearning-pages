""" Ask a question about a local file and get an answer from a LLM using RAG
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1) Load data
DIR = os.path.dirname(os.path.realpath(__file__))
with open(DIR + "/sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2) Chunk data
chunk_size = 500
overlap = 50
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]

# 3) Embed chunks
def embed_texts(texts):
    res = client.embeddings.create(model='text-embedding-3-small', input=texts)
    return np.array([d.embedding for d in res.data])

chunk_embeddings = embed_texts(chunks)

# 4) Retrive top chunks
def retrieve(query, k=2):
    query_emb = embed_texts([query])[0].reshape(1, -1)
    sims = cosine_similarity(query_emb, chunk_embeddings)[0]
    top_idxs = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in top_idxs]

# 5) Generate answer
def rag_answer(question):
    context = "\n".join(retrieve(question))
    prompt = f"Answer based ONLY on this context:\n{context}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    response_text = response.choices[0].message.content.strip()
    return response_text

if __name__ == "__main__":
    question = "What is the main topic in the document?"
    print(f"\nQuestion: {question}")
    answer = rag_answer(question)
    print(f"Answer: {answer}")

    """
    Question: What is the main topic in the document?
    Answer: The main topic in the document is the discussion among a group of friends 
    about their health and the need for rest, leading to a decision to take a break, 
    potentially a trip on the River, due to feeling overworked.
    """

    question = "Give me a paragraph where the author talks about cheese, please."
    print(f"\nQuestion: {question}")
    answer = rag_answer(question)
    print(f"Answer: {answer}")

    """
    Question: Give a paragraph where the author talks about cheese, please.
    Answer: Cheese, like oil, makes too much of itself. It wants the whole boat to itself. 
    It goes through the hamper, and gives a cheesy flavour to everything else there. 
    You can’t tell whether you are eating apple-pie or German sausage, 
    or strawberries and cream. It all seems cheese. There is too much odour about cheese. 
    I remember a friend of mine, buying a couple of cheeses at Liverpool. 
    Splendid cheeses they were, ripe and mellow, and with a two hundred horse-power 
    scent about them that might have been warranted to carry three miles, and knock a man
    over at two hundred yards.
    """

    while True:
        request = input('\nQuestion: ')
        if (request == 'quit'):
            break
        print(f"Answer: {rag_answer(request)}")

    """
    Question: Who is Montmorency?
    Answer: Montmorency is a character who does not enjoy the river or the scenery and 
    finds the activities of the others, particularly their boating, to be foolishness. 
    He does not engage in smoking and prefers to create mischief and annoy the group 
    by getting in the way and being a nuisance. His ambition in life is to be a 
    perfect bother to those around him.

    Question: Is Montmorency a person?
    Answer: No, Montmorency is not a person; Montmorency is a dog.

    Question: What is the pythagorean formula?
    Answer: The context provided does not contain any information about 
    the Pythagorean formula. Therefore, I am unable to answer your question 
    based solely on that context.
    """

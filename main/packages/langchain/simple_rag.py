""" Retrieval-Augmented Generation (RAG)
A model gets relevant information from a local knowledge base.
1) Document: A small text knowledge base.
2) Chunking: Break into small pieces (per line).
3) Embedding: Turn text into numerical vectors.
4) Retrieval: Find the most similar chunks to the question using cosine similarity.
5) Generation: Give those top chunks to the LLM in a prompt, it will answer from facts in your data.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from icecream import ic

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- 1) Data ----
document = """
LangChain is a Python framework for developing applications powered by language models.
It offers tools for prompt management, chaining, and connecting to external data.

RAG stands for Retrieval-Augmented Generation, where a model uses a retriever to get relevant info
from a knowledge base before generating an answer.
This helps reduce hallucinations and keeps answers factual.
"""

# ---- 2) Split into chunks ----
chunks = document.split("\n")
chunks = [c.strip() for c in chunks if c and c.strip()]
ic(chunks)
"""
ic| chunks: ['LangChain is a Python framework for developing applications powered by '
             'language models.',
             'It offers tools for prompt management, chaining, and connecting to external '
             'data.',
             'RAG stands for Retrieval-Augmented Generation, where a model uses a '
             'retriever to get relevant info',
             'from a knowledge base before generating an answer.',
             'This helps reduce hallucinations and keeps answers factual.']
"""

# ---- 3) Embed chunks ----
def embed_texts(texts):
    res = client.embeddings.create(
        model='text-embedding-3-small',
        input=texts
    )
    return np.array([d.embedding for d in res.data])

chunk_embeddings = embed_texts(chunks)
ic(chunk_embeddings)
"""
ic| chunk_embeddings: array([[-0.02162271,  0.00888773,  0.03650993, ..., -0.05459763,
                              -0.00785622, -0.00189854],
                             [-0.02058555,  0.0135925 ,  0.06969436, ..., -0.05116868,
                              -0.02132028,  0.01723991],
                             [-0.00323994,  0.02070102, -0.00502882, ..., -0.00503547,
                               0.02446847, -0.00111243],
                             [-0.02227938, -0.00813061,  0.05838839, ...,  0.00838136,
                               0.03080512,  0.01227949],
                             [-0.01651258, -0.01769928, -0.01214926, ..., -0.00260496,
                              -0.02000033,  0.01199007]])
"""

# ---- 4) Retrive top chunks ----
def retrieve(query, k=2):
    query_emb = embed_texts([query])[0].reshape(1, -1)
    sims = cosine_similarity(query_emb, chunk_embeddings)[0]
    top_idxs = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in top_idxs]

question = "What is RAG?"
context = "\n".join(retrieve(question))
ic(context)
"""
ic| context: ('RAG stands for Retrieval-Augmented Generation, where a model uses a '
              'retriever to get relevant info'
              'It offers tools for prompt management, chaining, and connecting to external '
              'data.')
"""

# ---- 5) Generate answer ----
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
    question = "What RAG means?"
    answer = rag_answer(question)
    
    ic(answer)
    """
    ic| answer: ('RAG stands for Retrieval-Augmented Generation, which involves a model using '
             'a retriever to obtain relevant information.')
    """
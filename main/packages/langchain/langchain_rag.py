""" With LangChain we have less boilerplate, more composability.
Buit-in best practice (chunking, retrival, prompt templating). 
"""

import os
from dotenv import load_dotenv

load_dotenv() # expects OPEN_API_KEY

# ---- LangChain imports ----
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 0) Models
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# 1) Load data
DIR = os.path.dirname(os.path.realpath(__file__))
docs = TextLoader(DIR + "/sample.txt", encoding="utf-8").load()

# 2) Chunk data
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)

# 3) Build vector store (embeddings under the hood)
vs = FAISS.from_documents(splits, embeddings)
retriver = vs.as_retriever(search_kwargs={"k": 2}) # top chunks 2

# 4) Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based ONLY on this context:\n{context}"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n".join(d.page_content for d in docs)

# 5) Compose the RAG chain (retrive -> prompt -> LLM -> text)
rag_chain = (
    RunnableParallel(
        {"context": retriver | format_docs, "question": RunnablePassthrough()}
    )
    | prompt
    | llm
    | StrOutputParser()
)

def rag_answer(question: str) -> str:
    return rag_chain.invoke(question).strip()

if __name__ == "__main__":
    while True:
        request = input("\nQuestion: ")
        if request.lower().strip() == "quit":
            break
        print(f"Answer: {rag_answer(request)}")

    """
    Question: Who are the characters in the text?
    Answer: The characters in the text include:

    1. Montmorency - carrying a stick.
    2. George - carrying coats and rugs, and smoking a short pipe.
    3. Harris - trying to walk with easy grace while carrying ...
    4. Greengrocer's boy - with a basket.
    5. Baker's boy - with a basket.
    6. Boots from the hotel - carrying a hamper.
    7. Confectioner's boy - with a basket.
    8. Grocer's boy - with a basket.
    9. Long-haired dog - mentioned twice.
    10. Cheesemonger's boy - with a basket, mentioned twice.
    11. Odd man - carrying a bag.
    12. Bosom companion of odd man - with hands in pockets, smoking a short clay.
    13. Fruiterer's boy - with a basket.
    14. Narrator (myself) - carrying three hats and a pair of boots.
    15. Six small boys.
    16. Four stray dogs.
    """
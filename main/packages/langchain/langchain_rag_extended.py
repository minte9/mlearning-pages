""" LangChain's components makes it easier to extend later, you can easily 
swap in other loaders (PDF/HTML), add memory, or an API layer 
with only a few lines changed.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ---- LangChain imports ----
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# ---- New converstion memory ----
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# 0) Model + embeddings
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# 1) Load data
DIR = os.path.dirname(os.path.realpath(__file__))
docs = TextLoader(DIR + "/sample.txt", encoding="utf-8").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)

# 2) Vector store
vs = FAISS.from_documents(splits, embeddings)
retriever = vs.as_retriever(search_kwargs={"k": 2}) # top chunks 2

# ---- New memory pattern ----
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

def rag_answer(question: str) -> str:
    result = conv_chain.invoke({"question": question})
    return result["answer"]

if __name__ == "__main__":
    while True:
        request = input("\nQuestion: ")
        if request.lower().strip() == "quit":
            break
        print(f"Answer: {rag_answer(request)}")


    """
    Question: Who is Montmorency?
    Answer: Montmorency is a character described as a small fox-terrier with a personality 
    that suggests a desire to improve the world. He is portrayed as lively and prefers 
    noisy environments, indicating that he does not enjoy solitude. 
    The description implies that he has a charming and somewhat mischievous demeanor that 
    can evoke strong emotions in people.

    Question: Is it a dog?
    Answer: Yes, Montmorency is a dog, specifically a fox-terrier.
    """

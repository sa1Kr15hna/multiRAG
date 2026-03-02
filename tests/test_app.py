import os
import pickle
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import EnsembleRetriever

OLLAMA_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
OLLAMA_LLM_MODEL = "qwen3:1.7b"
OLLAMA_BASE_URL = "http://localhost:11434"
CHROMA_DB_DIR = "./chroma_db"
BM25_INDEX_FILE = "./bm25_retriever.pkl"

print("1. Setup Retriever")
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

with open(BM25_INDEX_FILE, 'rb') as f:
    bm25_retriever = pickle.load(f)
bm25_retriever.k = 2

retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)

print("2. Setup LLM")
llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)

print("3. Setup Chain using latest LangChain syntax")
system_prompt = (
    "You are an expert assistant specialized ONLY in the TV show \"Grey's Anatomy\". "
    "Use the following pieces of retrieved context to answer the user's question. "
    "If the user asks a question that is NOT related to Grey's Anatomy (the TV show, its cast, characters, or plot), "
    "you must politely decline to answer, stating that you only have information about Grey's Anatomy. "
    "Do not make up information that isn't supported by the context."
    "\n\nContext:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("4. Invoke Chain")
response = rag_chain.invoke({"input": "Who is Meredith Grey?"})

print("\n--- Answer ---")
print(response["answer"])

print("\n--- Sources ---")
for i, doc in enumerate(response["context"]):
    print(f"Source {i+1}: {doc.metadata.get('title', 'Unknown')} - {doc.page_content[:100]}")

print("\nSuccess!")

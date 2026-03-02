import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_DB_DIR = "./chroma_db"
OLLAMA_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
OLLAMA_BASE_URL = "http://localhost:11434"

embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
data = vectorstore.get()
print("Number of docs:", len(data["documents"]))
print("Sample doc:", data["documents"][0][:100] if data["documents"] else "None")

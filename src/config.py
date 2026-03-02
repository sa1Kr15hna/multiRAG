import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Models
OLLAMA_LLM_MODEL = "qwen3:1.7b"
OLLAMA_GENERATION_MODEL = OLLAMA_LLM_MODEL  # Alias for evaluation scripts
OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"
OLLAMA_EVALUATION_MODEL = "deepseek-r1:1.5b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Database paths
CHROMA_DB_DIR = os.path.join(BASE_DIR, f"chroma_db_{OLLAMA_EMBEDDING_MODEL.replace(':', '_')}")
BM25_INDEX_FILE = os.path.join(BASE_DIR, "bm25_retriever.pkl")

# Shared Prompts
SYSTEM_PROMPT = (
    "You are an expert assistant specialized ONLY in the TV show \"Grey's Anatomy\". "
    "Use the following pieces of retrieved context to answer the user's question. "
    "If the user asks a question that is NOT related to Grey's Anatomy (the TV show, its cast, characters, or plot), "
    "you must politely decline to answer, stating that you only have information about Grey's Anatomy. "
    "Do not make up information that isn't supported by the context."
    "\n\nContext:\n{context}"
)

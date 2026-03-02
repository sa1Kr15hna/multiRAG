import time
import pickle
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import EnsembleRetriever

from src.config import (
    OLLAMA_LLM_MODEL, 
    OLLAMA_BASE_URL, 
    OLLAMA_EMBEDDING_MODEL,
    CHROMA_DB_DIR,
    BM25_INDEX_FILE,
    SYSTEM_PROMPT
)

class RAGCore:
    def __init__(self, llm_model=OLLAMA_LLM_MODEL, embedding_model=OLLAMA_EMBEDDING_MODEL):
        self.embeddings = OllamaEmbeddings(model=embedding_model, base_url=OLLAMA_BASE_URL)
        
        # 1. Setup Retrievers
        self.vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=self.embeddings)
        self.chroma_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        
        with open(BM25_INDEX_FILE, 'rb') as f:
            self.bm25_retriever = pickle.load(f)
        self.bm25_retriever.k = 2
        
        self.retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.chroma_retriever], weights=[0.2, 0.8]
        )
        
        # 2. Setup LLM and Prompt
        self.llm = ChatOllama(model=llm_model, base_url=OLLAMA_BASE_URL, temperature=0.1)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ])

    def query(self, user_query: str) -> dict:
        total_start_time = time.time()
        
        # Retrieval
        retrieval_start_time = time.time()
        context = self.retriever.invoke(user_query)
        retrieval_time = time.time() - retrieval_start_time
        
        # Format context
        context_text = "\n\n".join([doc.page_content for doc in context])
        formatted_prompt = self.prompt.invoke({"input": user_query, "context": context_text})
        
        # Generation
        generation_start_time = time.time()
        ai_message = self.llm.invoke(formatted_prompt)
        generation_time = time.time() - generation_start_time
        
        total_time = time.time() - total_start_time
        
        # Token usage (if available)
        usage = ai_message.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        return {
            "answer": ai_message.content,
            "context": context,
            "metrics": {
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        }

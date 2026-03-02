import os
import pickle
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever

# Configuration
from src.config import BM25_INDEX_FILE, CHROMA_DB_DIR, OLLAMA_EMBEDDING_MODEL, OLLAMA_BASE_URL

SEARCH_QUERIES = ["Grey's Anatomy"] + [f"Grey's Anatomy season {i}" for i in range(1, 23)]

def main():
    print("Loading Wikipedia articles...")
    docs = []
    for query in SEARCH_QUERIES:
        print(f"  Fetching: '{query}'")
        # Load 1 article per query
        loader = WikipediaLoader(query=query, load_max_docs=1, doc_content_chars_max=200000)
        docs.extend(loader.load())
        
    print(f"Loaded {len(docs)} documents total.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " ", ""],
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Creating BM25 index...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    
    print(f"Saving BM25 index to {BM25_INDEX_FILE}...")
    with open(BM25_INDEX_FILE, 'wb') as f:
        pickle.dump(bm25_retriever, f)

    print(f"Initializing Ollama embeddings with model: {OLLAMA_EMBEDDING_MODEL}...")
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    print(f"Creating and persisting Chroma DB to {CHROMA_DB_DIR}...")
    # Clean up existing Chroma DB if needed (optional)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    print(f"Chroma DB saved to {CHROMA_DB_DIR}")

    print("Ingestion complete!")

if __name__ == "__main__":
    main()

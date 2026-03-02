import os
import pickle
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever

# Configuration
SEARCH_QUERIES = ["Grey's Anatomy"] + [f"Grey's Anatomy season {i}" for i in range(1, 23)]
BM25_INDEX_FILE = "./bm25_retriever.pkl"

MODELS = ["bge-m3:latest"]
OLLAMA_BASE_URL = "http://localhost:11434"

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

    for model in MODELS:
        print(f"Initializing Ollama embeddings with model: {model}...")
        embeddings = OllamaEmbeddings(
            model=model,
            base_url=OLLAMA_BASE_URL
        )

        db_dir = f"./chroma_db_{model.replace(':', '_')}"
        print(f"Creating and persisting Chroma DB to {db_dir}...")
        # Clean up existing Chroma DB if needed (optional)
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_dir
        )
        print(f"Chroma DB saved to {db_dir}")

    print("Ingestion complete!")

if __name__ == "__main__":
    main()

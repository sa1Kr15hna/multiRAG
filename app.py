import os
import streamlit as st
from src.config import (
    OLLAMA_LLM_MODEL, 
    OLLAMA_EMBEDDING_MODEL,
    CHROMA_DB_DIR, 
    BM25_INDEX_FILE
)
from src.rag_core import RAGCore

# Configure Streamlit page
st.set_page_config(page_title="Wikipedia RAG App", page_icon="🌐", layout="wide")
st.title("🌐 Wikipedia RAG Application")
st.write("Retrieve relevant Wikipedia articles and generate answers using LangChain.")

# Sidebar for Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.markdown("---")
    st.markdown(f"**LLM Model:** `{OLLAMA_LLM_MODEL}`")
    st.markdown(f"**Embedding Model:** `{OLLAMA_EMBEDDING_MODEL}`")
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. You ask a question about Grey's Anatomy.
    2. We fetch relevant chunks from your local vector store.
    3. The LLM gets the context and generates an answer.
    4. Off-topic questions are rejected.
    """)

query = st.text_input("❓ Enter your question about Grey's Anatomy:")

if query:
    # Validation of databases
    if not os.path.exists(CHROMA_DB_DIR) or not os.path.exists(BM25_INDEX_FILE):
        st.warning(f"Databases not found for {OLLAMA_EMBEDDING_MODEL}. Have you run `python ingest.py` yet?")
        st.stop()
        
    with st.spinner("Searching Wikipedia & generating answer..."):
        try:
            # Initialize RAG Logic
            rag = RAGCore()
            
            # Query
            result = rag.query(query)
            
            answer = result["answer"]
            context = result["context"]
            metrics = result["metrics"]
            retrieval_time = metrics["retrieval_time"]
            generation_time = metrics["generation_time"]
            total_time = metrics["total_time"]
            input_tokens = metrics["input_tokens"]
            output_tokens = metrics["output_tokens"]
            
            # 6. Display Answer
            st.subheader("Answer")
            st.info(answer)

            # 7. Display Metrics
            st.subheader("Metrics")
            cols = st.columns(5)
            cols[0].metric("Retrieval time", f"{retrieval_time:.2f}s")
            cols[1].metric("LLM generation time", f"{generation_time:.2f}s")
            cols[2].metric("Total response time", f"{total_time:.2f}s")
            cols[3].metric("Input tokens", input_tokens)
            cols[4].metric("Output tokens", output_tokens)
            
            # 8. Display Source Documents
            with st.expander("📚 View Source Wikipedia Articles"):
                for i, doc in enumerate(context):
                    title = doc.metadata.get('title', 'Unknown Title')
                    url = doc.metadata.get('source', '')
                    st.markdown(f"**Source {i+1}: {title}**")
                    st.write(doc.page_content + "...")
                    if url:
                        st.markdown(f"[Read full article on Wikipedia]({url})")
                    st.markdown("---")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

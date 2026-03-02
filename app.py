import os
import pickle
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import EnsembleRetriever

# Constants mapping to our specific Ollama models
OLLAMA_LLM_MODEL = "qwen3:1.7b"
OLLAMA_BASE_URL = "http://localhost:11434"
BM25_INDEX_FILE = "./bm25_retriever.pkl"

# Configure Streamlit page
st.set_page_config(page_title="Wikipedia RAG App", page_icon="🌐", layout="wide")
st.title("🌐 Wikipedia RAG Application")
st.write("Retrieve relevant Wikipedia articles and generate answers using LangChain.")

# Sidebar for Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.markdown("---")
    st.markdown(f"**LLM Model:** `{OLLAMA_LLM_MODEL}`")
    OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"
    st.markdown(f"**Embedding Model:** `{OLLAMA_EMBEDDING_MODEL}`")
    
    CHROMA_DB_DIR = f"./chroma_db_{OLLAMA_EMBEDDING_MODEL.replace(':', '_')}"
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
            # 1. Setup Retriever
            embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
            vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
            chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
            
            with open(BM25_INDEX_FILE, 'rb') as f:
                bm25_retriever = pickle.load(f)
            bm25_retriever.k = 2
            
            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, chroma_retriever], weights=[0.2, 0.8]
            )
            
            # 2. Setup LLM
            llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
            
            import time
            total_start_time = time.time()

            # 3. Retrieval
            retrieval_start_time = time.time()
            context = retriever.invoke(query)
            retrieval_time = time.time() - retrieval_start_time

            # 4. Setup Chain for Generation
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
            
            # Format context
            context_text = "\n\n".join([doc.page_content for doc in context])
            formatted_prompt = prompt.invoke({"input": query, "context": context_text})
            
            # 5. Generation
            generation_start_time = time.time()
            ai_message = llm.invoke(formatted_prompt)
            generation_time = time.time() - generation_start_time
            
            total_time = time.time() - total_start_time
            
            answer = ai_message.content
            usage = ai_message.usage_metadata or {}
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            
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

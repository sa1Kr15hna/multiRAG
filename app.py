import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configure Streamlit page
st.set_page_config(page_title="Wikipedia RAG App", page_icon="🌐", layout="wide")
st.title("🌐 Wikipedia RAG Application")
st.write("Retrieve relevant Wikipedia articles and generate answers using LangChain.")

# Sidebar for Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # We offer both OpenAI and local Ollama to truly allow "run in local"
    llm_provider = st.selectbox("Select LLM Provider", ["Ollama (Local)","OpenAI"])
    
    openai_api_key = None
    ollama_model = None
    
    if llm_provider == "OpenAI":
        openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
    else:
        ollama_model = st.text_input("Ollama Model Name", value="qwen3:1.7b")
        ollama_base_url = st.text_input("Ollama Base URL", value="http://localhost:11434", help="If deploying to Streamlit Cloud, use a tunneling service like ngrok (e.g., https://your-ngrok-url.ngrok-free.app) and paste it here.")
        st.markdown("*Note: You must have [Ollama](https://ollama.com/) running locally with this model installed (e.g., `ollama run qwen3:1.7b`).*")
        
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. You ask a question.
    2. `WikipediaRetriever` fetches relevant articles.
    3. LangChain combines the context and sends it to the LLM.
    4. The answer is displayed below.
    """)

query = st.text_input("❓ Enter your question about any topic:")

if query:
    # Validation
    if llm_provider == "OpenAI" and not openai_api_key:
        st.warning("Please provide an OpenAI API Key in the sidebar to proceed.")
        st.stop()
        
    with st.spinner("Searching Wikipedia & generating answer..."):
        try:
            # 1. Setup Retriever
            # top_k_results: number of articles to retrieve
            # doc_content_chars_max: limit the characters from each document
            retriever = WikipediaRetriever(top_k_results=1, doc_content_chars_max=1000)
            
            # 2. Setup LLM
            if llm_provider == "OpenAI":
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.3)
            else:
                llm = ChatOllama(model=ollama_model, base_url=ollama_base_url, temperature=0.3)
            
            # 3. Setup Chain using latest LangChain syntax
            system_prompt = (
                "You are a helpful assistant. Use the following pieces of retrieved Wikipedia context to answer the user's question. "
                "If the answer is not in the context, just say that you don't know based on the provided context. "
                "Do not make up information that isn't supported by the context."
                "\n\nContext:\n{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            # 4. Invoke Chain
            response = rag_chain.invoke({"input": query})
            
            # 5. Display Answer
            st.subheader("Answer")
            st.info(response["answer"])
            
            # 6. Display Source Documents
            with st.expander("📚 View Source Wikipedia Articles"):
                for i, doc in enumerate(response["context"]):
                    title = doc.metadata.get('title', 'Unknown Title')
                    url = doc.metadata.get('source', '')
                    st.markdown(f"**Source {i+1}: {title}**")
                    st.write(doc.page_content + "...")
                    if url:
                        st.markdown(f"[Read full article on Wikipedia]({url})")
                    st.markdown("---")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Wikipedia RAG App: Grey's Anatomy

A local Retrieval-Augmented Generation (RAG) application built with LangChain and Streamlit. This app retrieves Wikipedia articles about the TV show *Grey's Anatomy*, vectors them locally, and answers user questions without making any external API calls (100% privacy).

## Prerequisites
Before you begin, ensure you have installed:
1. **Python 3.9+** or higher
2. **Ollama**: Download from [ollama.com](https://ollama.com/)

## Step-by-Step Setup Instructions

### 1. Download Local AI Models via Ollama
Open your terminal or command prompt and download the local LLM and Embedding models.
```bash
# Language Model for answering questions
ollama pull qwen3:1.7b

# Embedding model for vectorizing the documents
ollama pull bge-m3:latest

# Ensure the model is running and test it in your terminal
ollama run qwen3:1.7b
```
*(You can exit the `ollama run` terminal by typing `/bye`. Ensure the Ollama application remains running in the background while you execute the rest of the steps).*

### 2. Set Up a Virtual Environment (Recommended)
It's best to run this app inside an isolated Python environment.
**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```
**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies
Install the required packages using the generated `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Ingest the Data
Before starting the Streamlit app, you must fetch the Wikipedia articles and build the local Vector and BM25 databases. This will fetch 23 articles and embed them using `bge-m3:latest`.
```bash
python ingest.py
```
*Wait for this process to complete. You will see a `chroma_db_bge-m3_latest` folder and a `bm25_retriever.pkl` file appear in your directory when it's done.*

### 5. Run the Application
Finally, start your local Streamlit server:
```bash
streamlit run app.py
```
This command will automatically open a tab in your default web browser (usually at `http://localhost:8501`) where you can chat with the application!

## Evaluating the Application (Ragas)
This project includes an automated evaluation script that uses the [Ragas framework](https://docs.ragas.io/) to measure the quality of the generated answers against a set of predefined test inputs.

### Metrics Used
The evaluation currently measures two key metrics from Ragas:
1. **Faithfulness**: Measures the factual consistency of the generated answer against the retrieved context.
2. **Answer Relevancy**: Measures how relevant the generated answer is to the user's prompt/question.

*(Note: `context_precision` and `context_recall` are also available in the script but are commented out by default).*

### Running the Evaluation
To run the evaluation script, use the following command from the root directory:
```bash
python tests/evaluate_ragas.py
```

After the evaluation completes, you will see the overall results printed in your terminal, and a detailed CSV file named `ragas_evaluation_results.csv` will be saved locally.

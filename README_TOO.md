# RAG Application & Evaluation Walkthrough

## Section 1: Project Architecture and Workflow

When a user asks a question, several interconnected scripts and modules handle the request to form the final generated response. Here is a walkthrough of the project directory and how each script contributes to the workflow:

- **`src/config.py`**: Acts as the single source of truth for the application's configuration. It stores the names of the Ollama models (`qwen3:1.7b` for text generation and `bge-m3:latest` for embeddings), the base URL for the local Ollama instance, the pre-defined system prompts, and the absolute paths to the local vector databases.

- **`ingest.py`**: The foundational script run during setup. It scrapes Wikipedia articles related to "Grey's Anatomy" (seasons 1-22), splits the text into manageable chunks, and embeds them. It creates two searchable databases: a Chroma DB vector store for semantic similarity and a BM25 index for keyword search.

- **`src/rag_core.py`**: This module contains the `RAGCore` class, which holds the core application logic. It initializes both the Chroma and BM25 retrievers and combines them into an `EnsembleRetriever`. When the `query()` method is called with the user's question, it retrieves the most relevant context chunks and passes them alongside the question to the LangChain `ChatOllama` model to generate a final answer. It also calculates backend metrics like retrieval time, generation time, and token usage.

- **`app.py`**: The Streamlit user interface. It initializes the web app, captures the user's text input, and instantiates the `RAGCore` class to process the query. Once the query is processed, `app.py` renders the generated answer, the source context documents, and performance metrics visually to the user.

- **`tests/evaluate_ragas.py`**: An automated testing script designed to grade the RAG system's response quality using the `ragas` evaluation framework. It loops through a predefined dataset of test questions grouped with expected ground-truth references. It uses `RAGCore` to generate answers, and then a stronger evaluator LLM (`deepseek-r1:1.5b`) grades those answers on various metrics.

---

## Section 2: Ragas Evaluation Metrics

The `evaluate_ragas.py` script leverages the Ragas framework to compute four distinct metrics, yielding a granular understanding of where the RAG pipeline excels and where it fails. 

### Metrics Explained
1. **Faithfulness**: Measures the factual consistency of the generated answer against the retrieved context. A high score means the LLM did not hallucinate information outside of what was retrieved.
2. **Answer Relevancy**: Assesses how thoroughly and directly the generated answer addresses the user's question. It penalizes incomplete or overly verbose/tangential answers.
3. **Context Precision**: Evaluates the signal-to-noise ratio in the retrieved chunks. A high score indicates that the retrieved contexts were highly relevant to the question and contained minimal irrelevant text.
4. **Context Recall**: Measures if the retrieved context actually contains the necessary information to construct the ground-truth reference answer. A low score implies the retriever failed to find the right documents.

### Evaluation Results
The script was run against a sub-set of 3 questions, with the raw results saved to `ragas_evaluation_results.csv`.

| Question | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
| :--- | :--- | :--- | :--- | :--- |
| *Who is Meredith Grey?* | N/A | 0.7467 | 0.9999 | 0.3333 |
| *What is the name of the hospital in the first season?* | 1.0000 | 0.8686 | 0.4999 | 1.0000 |
| *Who does Meredith Grey marry?* | 0.0000 | 0.0000 | 0.9999 | 0.0000 |

### Insights
- **Relevancy and Faithfulness**: The system performs well when the retrieval fetches explicit answers (e.g., the hospital name resulting in 1.0 Faithfulness). However, it struggles heavily when context is missing (Meredith's marriage), causing it to output that the context does not specify the answer, resulting in scores of 0.0.
- **Context Retrieval**: The retriever is generally precise but suffers from low recall on certain queries. Adjusting chunk sizes or the top-K retrieval count might help surface the proper documents for challenging questions.

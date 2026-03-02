import os
import sys
from datasets import Dataset
import pandas as pd

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama

from src.config import OLLAMA_EVALUATION_MODEL, OLLAMA_BASE_URL, OLLAMA_GENERATION_MODEL
from src.rag_core import RAGCore

def main():
    print("1. Setup RAG Core")
    rag = RAGCore(llm_model=OLLAMA_GENERATION_MODEL)

    print("2. Setup Chain (Handled by RAGCore)")

    print("3. Define Test Dataset")
    inputs = [
        {
            "user_input": "Who is Meredith Grey?", 
            "reference": "Meredith Grey is the protagonist of Grey's Anatomy, a general surgeon at Grey Sloan Memorial Hospital and the daughter of Ellis Grey."
        }
        #, {
        #     "user_input": "What is the name of the hospital in the first season?", 
        #     "reference": "The hospital is named Seattle Grace Hospital in the first season."
        # },
        # {
        #     "user_input": "Who does Meredith Grey marry?", 
        #     "reference": "Meredith Grey marries Derek Shepherd."
        # }
    ]

    print("4. Generate Responses")
    data_samples = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
        "retrieval_time_s": [],
        "generation_time_s": []
    }

    for item in inputs:
        q = item["user_input"]
        print(f"Generating answer for: {q}")
        
        # Get response and metrics from RAGCore
        result = rag.query(q)
        retrieved_docs = result["context"]
        retrieval_time = result["metrics"]["retrieval_time"]
        chain_response = result["answer"]
        generation_time = result["metrics"]["generation_time"]
        
        data_samples["user_input"].append(q)
        data_samples["response"].append(chain_response)
        data_samples["retrieved_contexts"].append([doc.page_content for doc in retrieved_docs])
        data_samples["reference"].append(item["reference"])
        data_samples["retrieval_time_s"].append(f"{retrieval_time:.4f}")
        data_samples["generation_time_s"].append(f"{generation_time:.4f}")

    dataset = Dataset.from_dict(data_samples)

    print("5. Setup Ragas Wrappers")
    eval_llm_instance = ChatOllama(model=OLLAMA_EVALUATION_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    eval_llm = LangchainLLMWrapper(eval_llm_instance)
    eval_embeddings = LangchainEmbeddingsWrapper(rag.embeddings)

    from ragas.run_config import RunConfig
    print("6. Run Ragas Evaluation")
    metrics = [faithfulness, answer_relevancy]#, context_precision, context_recall
    
    # Run evaluation
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=RunConfig(timeout=600, max_workers=1)
    )
    
    print("\n--- Evaluation Results ---")
    print(result)
    
    df = result.to_pandas()
    df.to_csv("ragas_evaluation_results.csv", index=False)
    print("\nDetailed results saved to ragas_evaluation_results.csv")

if __name__ == "__main__":
    main()

# Local Hybrid RAG & Evaluation Pipeline

This repository contains a comprehensive Jupyter Notebook implementing an end-to-end, locally hosted Hybrid Retrieval-Augmented Generation (RAG) pipeline. It is designed to evaluate and benchmark multiple retrieval strategies against complex document layouts (e.g., historical documents and tables) using local LLMs.

The primary focus of this project is the **evaluation framework**, moving beyond basic RAG implementations to rigorously quantify the performance of different retrieval and ranking methodologies.

## Key Features

*   **100% Local LLM Execution:** Utilizes `qwen2.5:3b` deployed via Ollama for privacy-preserving, local text generation and evaluation.
*   **Complex Document Parsing:** Integrates the Unstructured API to handle difficult layouts, preserving tabular data and structural integrity during chunking.
*   **Multi-Strategy Retrieval:** Implements and compares four distinct retrieval architectures:
    *   **Sparse Retrieval:** BM25 (Keyword-based)
    *   **Dense Retrieval:** ChromaDB with `BAAI/bge-small-en-v1.5` embeddings (Semantic)
    *   **Hybrid Retrieval:** Ensemble of Sparse and Dense methods.
    *   **Reranked Hybrid:** Cross-encoder reranking using `ms-marco-MiniLM-L-6-v2` for optimized context selection.
*   **Comprehensive Evaluation Framework:** 
    *   **LLM-as-a-Judge:** Uses the local Qwen model to evaluate outputs for *Correctness* and *Faithfulness* on a 1-5 scale.
    *   **Automated Metrics:** Calculates Token F1, Semantic Similarity, Exact Match, ROUGE, and BLEU without relying on external APIs.

## Experimental Results

In a batch test against a ground-truth dataset of 50 questions, the retrieval strategies yielded the following insights:
*   **Dense** and **Hybrid** retrievers demonstrated the highest correctness scores (~4.72 - 4.76/5.0).
*   **Sparse** retrieval (BM25 standalone) struggled with semantic queries, yielding the lowest performance.
*   The complete evaluation dataset, including metrics and LLM judge scores, is exported to CSV and Pickle formats in the `/results` directory.

## Prerequisites

*   Python 3.9+
*   [Ollama](https://ollama.com/) installed and running locally.
*   Jupyter Notebook or JupyterLab.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/local-hybrid-rag-eval.git
   cd local-hybrid-rag-eval
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have `langchain`, `chromadb`, `unstructured`, `sentence-transformers`, and `rank_bm25` installed).*

3. Pull the required local model via Ollama:
   ```bash
   ollama pull qwen2.5:3b
   ```

## Usage

1. Start your local Ollama server.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `hybrid_rag_evaluation.ipynb` and run the cells sequentially to ingest your PDFs, build the vector stores, and execute the evaluation pipeline.

## Roadmap (Future Work)

This notebook represents the research and prototyping phase. Planned upgrades for production readiness include:
- [ ] **Modularization:** Refactoring the notebook into a production-grade Python package with separated concerns (ingestion, retrieval, generation, evaluation).
- [ ] **Custom Parsing Engine:** Replacing the generic Unstructured API with a custom-built PDF parser optimized for high-noise financial documents.
- [ ] **API & Containerization:** Wrapping the generation and retrieval logic in a FastAPI service and providing a Dockerfile for seamless deployment.

## License
MIT License

# Hybrid RAG System with Automated Evaluation

A comprehensive Retrieval-Augmented Generation (RAG) system combining dense vector retrieval, sparse keyword-based retrieval (BM25), and Reciprocal Rank Fusion (RRF) to answer questions from Wikipedia articles.

## 📋 Overview

This system implements a hybrid retrieval approach that combines:

1. **Dense Retrieval**: Semantic search using sentence embeddings
2. **Sparse Retrieval**: Keyword-based search using BM25
3. **RRF Fusion**: Combines both methods using Reciprocal Rank Fusion
4. **LLM Generation**: Generates answers from retrieved context
5. **Comprehensive Evaluation**: Automated evaluation with multiple metrics

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│         Wikipedia Article Collection              │
│  (500 URLs: 200 fixed + 300 random per run)      │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │   Preprocessing      │
        │ (Chunking, Cleaning) │
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────────────┐
        │                             │
     ┌──┴──┐                    ┌─────┴────┐
     │Dense│                    │  Sparse  │
     │Index│                    │   Index  │
     │(FAISS) + Embeddings │     (BM25)   │
     └──┬──┘                    └─────┬────┘
        │                            │
        └──────────┬─────────────────┘
                   │
        ┌──────────┴──────────┐
        │   RRF Fusion        │
        │  (Combine scores)   │
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        │   Generation        │
        │  (LLM based)        │
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────────────┐
        │   Evaluation Framework       │
        │ (Metrics + Error Analysis)   │
        └─────────────────────────────┘
```

## 📁 Project Structure

```
hybrid_rag_system/
├── src/
│   ├── __init__.py
│   ├── data_collection.py      # Wikipedia data fetching
│   ├── preprocessing.py         # Text chunking and cleaning
│   ├── dense_retrieval.py      # FAISS-based dense retrieval
│   ├── sparse_retrieval.py     # BM25 sparse retrieval
│   ├── fusion.py               # RRF fusion strategy
│   ├── generation.py           # LLM response generation
│   └── rag_system.py           # Main RAG orchestrator
│
├── evaluation/
│   ├── __init__.py
│   ├── question_generation.py  # Q&A dataset generation
│   ├── metrics.py              # Evaluation metrics
│   └── evaluation_pipeline.py  # Automated evaluation
│
├── ui/
│   └── app.py                  # Streamlit web interface
│
├── notebooks/
│   └── demo.ipynb              # Demonstration notebook
│
├── data/
│   ├── corpus/                 # Preprocessed documents
│   ├── indices/                # Vector and BM25 indices
│   └── qa/                     # Q&A datasets
│
├── results/                    # Evaluation results
├── requirements.txt            # Python dependencies
├── setup.py                    # Project setup
├── README.md                   # This file
└── fixed_urls.json             # Fixed Wikipedia URLs (200)
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda


### Setup Steps

1. **Clone/Navigate to project directory**
```bash
cd hybrid_rag_system
```

2. **Create virtual environment** (optional but recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n rag python=3.11
conda activate rag
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

5. **Initialize project structure**
```bash
python setup.py
```

## 📖 Usage Guide

### 1. Data Collection

```python
from src.data_collection import WikipediaCollector
from src.preprocessing import prepare_rag_corpus
import json
from pathlib import Path

# Load or create fixed URLs
fixed_urls = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    # ... add 200 URLs total
]

# Save fixed URLs
collector = WikipediaCollector()
collector.save_fixed_urls(fixed_urls, "data/fixed_urls.json")

# Collect documents
documents, valid_urls = collector.collect_from_urls(fixed_urls)

# Preprocess into chunks
chunks = prepare_rag_corpus(documents, chunk_size=300, overlap=50)
```

### 2. Build Indices

```python
from src.rag_system import HybridRAGSystem

# Initialize system
rag = HybridRAGSystem(config={
    'chunk_size': 300,
    'dense_model': 'all-MiniLM-L6-v2',
    'llm_model': 'distilgpt2',
    'device': 'cuda'  # or 'cpu'
})

# Build corpus and indices
documents, valid_urls = rag.collector.collect_from_urls(fixed_urls)
chunks = rag.preprocessor.process_documents(documents)
rag.chunks = chunks

# Build indices
rag.build_indices(chunks, save_path="data/indices")

# Save system
rag.save_system(save_path="data/rag_system")
```

### 3. Query the System

```python
# Answer a single query
result = rag.answer_query("What is artificial intelligence?")

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Time: {result['timings']['total']:.3f}s")

# Display retrieved chunks
for chunk in result['retrieval']['fused'][:3]:
    print(f"\n- {chunk['title']}")
    print(f"  Score: {chunk['rrf_score']:.4f}")
    print(f"  {chunk['text'][:200]}...")
```

### 4. Generate Q&A Dataset

```python
from evaluation.question_generation import QuestionGenerator, save_qa_dataset

# Generate Q&A pairs
generator = QuestionGenerator(chunks)
qa_pairs = generator.generate_qa_dataset(total_questions=100)

# Save dataset
save_qa_dataset(qa_pairs, "data/qa/qa_dataset.json")

print(f"Generated {len(qa_pairs)} Q&A pairs")
```

### 5. Run Evaluation

```python
from evaluation.evaluation_pipeline import AutomatedEvaluationPipeline

# Initialize evaluation
eval_pipeline = AutomatedEvaluationPipeline(rag)

# Run evaluation
evaluations, summary = eval_pipeline.run_evaluation(
    num_questions=100,
    save_results=True
)

# Print summary
print(f"\nMRR (URL-level): {summary['mrr_url_mean']:.4f} ± {summary['mrr_url_std']:.4f}")
print(f"Precision@10: {summary['precision_at_10_mean']:.4f}")
print(f"Recall@10: {summary['recall_at_10_mean']:.4f}")
print(f"NDCG@10: {summary['ndcg_at_10_mean']:.4f}")
print(f"Semantic Similarity: {summary['semantic_similarity_mean']:.4f}")
```

### 6. Run Web Interface

```bash
streamlit run ui/app.py
```

Then open http://localhost:8501 in your browser.

## 📊 Evaluation Metrics

### Mandatory Metrics

**Mean Reciprocal Rank (MRR) - URL Level**
- Measures how quickly the system identifies the correct source document
- Formula: MRR = (1/n) × Σ(1/rank_i) for each question's first correct URL
- Range: [0, 1], Higher is better

### Custom Metrics

**1. Precision@K and Recall@K**
- Evaluate retrieval quality at different cutoff points
- Precision: Fraction of retrieved documents that are relevant
- Recall: Fraction of relevant documents that are retrieved

**2. NDCG@K (Normalized Discounted Cumulative Gain)**
- Evaluates ranking quality considering position
- Accounts for relevance grading
- DCG = Σ(1 / log2(rank + 1)) for relevant documents

**3. Semantic Similarity**
- Evaluates answer quality using embedding-based similarity
- Measures semantic overlap between generated and reference answers
- Uses cosine similarity of embeddings

**4. Contextual Precision & Recall**
- Precision: % of retrieved chunks relevant to question
- Recall: % of ground truth chunks successfully retrieved
- Evaluates context relevance

**5. Hit Rate**
- Binary metric: whether correct URL is in top-K results
- Hit Rate@K = 1 if found else 0

## 🔍 Advanced Features

### Ablation Studies

Compare performance across different configurations:

```python
# Dense only, Sparse only, Hybrid
ablation_results = eval_pipeline.run_ablation(
    num_questions=50,
    save_results=True
)
```

### Error Analysis

Categorize and analyze failures:

```python
error_categories = eval_pipeline.error_analyzer.categorize_errors(evaluations)
error_by_type = eval_pipeline.error_analyzer.error_summary_by_question_type(evaluations)

print(f"Retrieval failures: {len(error_categories['retrieval_failure'])}")
print(f"Generation failures: {len(error_categories['generation_failure'])}")
```

### System Configuration

All system parameters can be customized:

```python
config = {
    'chunk_size': 300,              # Tokens per chunk
    'chunk_overlap': 50,            # Overlapping tokens
    'dense_model': 'all-MiniLM-L6-v2',  # Embedding model
    'dense_top_k': 10,              # Dense retrieval top-k
    'sparse_top_k': 10,             # Sparse retrieval top-k
    'final_top_n': 5,               # Chunks for generation
    'llm_model': 'distilgpt2',      # LLM for generation
    'max_context_tokens': 300,      # Max context length
    'device': 'cpu'                 # Computing device
}

rag = HybridRAGSystem(config=config)
```

## 📚 Example Commands

### Complete Pipeline

```python
# Run complete RAG system pipeline
python pipeline.py --fixed-urls fixed_urls.json --num-questions 100

# Run only evaluation on existing system
python evaluate.py --results-dir results/latest

# Generate report (PDF/HTML)
python generate_report.py --results results/summary.json --output report.pdf
```

## 🔧 Configuration Files

### fixed_urls.json
Contains the fixed set of 200 Wikipedia URLs:

```json
{
  "fixed_urls": [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    ...
  ]
}
```

### System Configuration
Adjust via `HybridRAGSystem` initialization or Streamlit UI

## 📈 Output & Results

Results are saved in the `results/` directory:

- `summary_*.json`: Overall metrics and statistics
- `evaluations_*.json`: Detailed per-question evaluation
- `ablation_*.json`: Ablation study results
- `report_*.html`: HTML report with visualizations

## 🐛 Troubleshooting

### Out of Memory
- Reduce `chunk_size`
- Reduce `batch_size` in training
- Use `device='cpu'` if GPU memory is limited

### Slow Performance
- Enable GPU: `device='cuda'`
- Use smaller models: `all-MiniLM-L6-v2` instead of larger models
- Increase batch sizes for encoding

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### NLTK Data Missing
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 📝 Notes

- Each indexing run samples NEW 300 random URLs while keeping 200 fixed
- All 500 URLs must have minimum 200 words
- Chunks are created with 200-400 token range and 50-token overlap
- RRF uses k=60 as default constant parameter
- All URLs and metadata are preserved in chunk objects for tracking




---

**Last Updated**: February 2026  
**Python Version**: 3.8+  
**Status**: Active Development

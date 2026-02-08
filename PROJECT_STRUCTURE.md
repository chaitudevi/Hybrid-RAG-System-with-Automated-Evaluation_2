# Project Structure - Hybrid RAG System

## Complete File Listing

```
hybrid_rag_system/
│
├── src/                                    # Core RAG implementation
│   ├── __init__.py                        # Package initialization
│   ├── data_collection.py                 # Wikipedia data fetching & extraction
│   │   └── WikipediaCollector class
│   │       - extract_text_from_url()
│   │       - collect_from_urls()
│   │       - load_fixed_urls() / save_fixed_urls()
│   │       - sample_random_urls()
│   │
│   ├── preprocessing.py                   # Text chunking & cleaning
│   │   ├── TextPreprocessor class
│   │   │   - clean_text()
│   │   │   - count_tokens()
│   │   │   - split_into_sentences()
│   │   └── ChunkProcessor class
│   │       - tokenize()
│   │       - chunk_text()
│   │       - process_documents()
│   │   └── prepare_rag_corpus() function
│   │
│   ├── dense_retrieval.py                 # FAISS-based dense retrieval
│   │   ├── DenseRetriever class
│   │   │   - embed_texts()
│   │   │   - build_index()
│   │   │   - retrieve()
│   │   │   - save() / load()
│   │   └── HybridDenseRetriever class
│   │       - build_all_indices()
│   │       - retrieve_ensemble()
│   │
│   ├── sparse_retrieval.py                # BM25 sparse retrieval
│   │   ├── SparseRetriever class
│   │   │   - tokenize()
│   │   │   - build_index()
│   │   │   - retrieve()
│   │   │   - save() / load()
│   │   └── create_inverted_index() function
│   │
│   ├── fusion.py                          # RRF (Reciprocal Rank Fusion)
│   │   ├── ReciprocalRankFusion class
│   │   │   - compute_rrf_score()
│   │   │   - simple_rrf()
│   │   │   - weighted_rrf()
│   │   ├── EnsembleRetriever class
│   │   │   - register_method()
│   │   │   - retrieve_with_fusion()
│   │   └── RRFAnalyzer class
│   │       - analyze_score_contributions()
│   │
│   ├── generation.py                      # LLM-based response generation
│   │   ├── ResponseGenerator class
│   │   │   - count_tokens()
│   │   │   - build_prompt()
│   │   │   - generate()
│   │   ├── LLMAsJudge class
│   │   │   - evaluate_answer()
│   │   └── ConfidenceEstimator class
│   │       - estimate_confidence()
│   │       - estimate_hallucination_probability()
│   │
│   └── rag_system.py                      # Main RAG orchestrator
│       └── HybridRAGSystem class
│           - build_corpus()
│           - build_indices()
│           - answer_query()
│           - batch_answer_queries()
│           - save_system() / load_system()
│
├── evaluation/                            # Evaluation framework
│   ├── __init__.py                        # Package initialization
│   │
│   ├── question_generation.py             # Q&A pair generation
│   │   ├── QuestionGenerator class
│   │   │   - generate_factual_questions()
│   │   │   - generate_comparative_questions()
│   │   │   - generate_inferential_questions()
│   │   │   - generate_multihop_questions()
│   │   │   - generate_qa_dataset()
│   │   ├── save_qa_dataset() function
│   │   └── load_qa_dataset() function
│   │
│   ├── metrics.py                         # Evaluation metrics
│   │   ├── MetricsCalculator class
│   │   │   - calculate_mrr_url_level()    [MANDATORY]
│   │   │   - calculate_precision_at_k()   [CUSTOM]
│   │   │   - calculate_recall_at_k()      [CUSTOM]
│   │   │   - calculate_hit_rate()
│   │   │   - calculate_ndcg()
│   │   ├── AnswerQualityMetrics class
│   │   │   - calculate_exact_match()
│   │   │   - calculate_semantic_similarity()
│   │   │   - calculate_answer_length_score()
│   │   ├── ContextRelevanceMetrics class
│   │   │   - calculate_contextual_precision()
│   │   │   - calculate_contextual_recall()
│   │   └── EvaluationPipeline class
│   │       - evaluate_single_result()
│   │       - evaluate_batch()
│   │
│   └── evaluation_pipeline.py             # Automated evaluation
│       ├── AblationStudy class
│       │   - dense_only()
│       │   - sparse_only()
│       │   - hybrid()
│       │   - compare_k_values()
│       ├── ErrorAnalyzer class
│       │   - categorize_errors()
│       │   - error_summary_by_question_type()
│       └── AutomatedEvaluationPipeline class
│           - run_evaluation()
│           - run_ablation()
│
├── ui/                                    # Web interface
│   └── app.py                             # Streamlit web app
│       - init_session_state()
│       - load_rag_system()
│       - main()
│       - show_home()
│       - show_query_interface()
│       - show_evaluation()
│       - show_system_config()
│       - show_documentation()
│
├── notebooks/                             # Jupyter notebooks
│   └── demo.ipynb                         # Complete demonstration
│       1. Setup & imports
│       2. Initialize RAG system
│       3. Load fixed URLs
│       4. Collect documents
│       5. Preprocess chunks
│       6. Build indices
│       7. Test single query
│       8. Batch processing
│       9. Q&A generation
│       10. Run evaluation
│       11. Ablation studies
│       12. Error analysis
│       13. Visualization
│
├── data/                                  # Data directory
│   ├── corpus/
│   │   ├── chunks.json                   # Preprocessed chunks
│   │   └── documents.json                # Raw documents
│   ├── indices/
│   │   ├── dense/                        # Dense retrieval index
│   │   │   ├── dense.index              # FAISS index
│   │   │   ├── chunks.pkl               # Chunk metadata
│   │   │   └── model_info.txt
│   │   └── sparse/                       # Sparse retrieval index
│   │       ├── bm25.pkl                 # BM25 model
│   │       ├── chunks.pkl
│   │       └── tokenized_chunks.pkl
│   └── qa/
│       └── qa_dataset.json              # Q&A pairs (100)
│
├── results/                               # Evaluation results
│   ├── evaluations_*.json                # Detailed results
│   ├── summary_*.json                    # Summary metrics
│   ├── ablation_*.json                   # Ablation study
│   ├── metric_distributions.png          # Visualization
│   └── report_*.html                     # HTML report
│
├── fixed_urls.json                        # Fixed 200 Wikipedia URLs
├── requirements.txt                       # Python dependencies
├── setup.py                               # Project initialization
├── main.py                                # Main pipeline script
├── README.md                              # Documentation
└── .gitignore                             # Git ignore file


## Key Components

### 1. Data Pipeline
- **Input**: Wikipedia URLs (200 fixed + 300 random)
- **Processing**: Text extraction, cleaning, chunking with overlap
- **Output**: Structured chunks with metadata

### 2. Indexing
- **Dense Index**: FAISS vector database with sentence embeddings
- **Sparse Index**: BM25 keyword index
- **Both**: Queryable and saveable

### 3. Retrieval
- **Dense Retriever**: Semantic similarity search
- **Sparse Retriever**: Keyword-based search
- **RRF Fusion**: Combines both methods with rank fusion

### 4. Generation
- **LLM Models**: DistilGPT2, Flan-T5, or custom models
- **Context Building**: Chunk selection and prompt engineering
- **Confidence Estimation**: Answer quality assessment

### 5. Evaluation
- **Metric Categories**:
  - Retrieval: MRR, Precision@K, Recall@K, NDCG, Hit Rate
  - Answer Quality: Semantic Similarity, Answer Length Score
  - Context: Contextual Precision, Contextual Recall
- **Ablation Studies**: Dense-only, Sparse-only, Hybrid comparison
- **Error Analysis**: Categorization by failure type and question type

### 6. User Interface
- **Streamlit App**: Interactive web interface
- **Features**: Query interface, evaluation dashboard, configuration, documentation

## Metrics Implemented

### Mandatory Metrics
- **Mean Reciprocal Rank (MRR) - URL Level**: Measures retrieval effectiveness

### Custom Metrics
1. **Precision@K & Recall@K**: Retrieval quality at different cutoffs
2. **NDCG@K**: Ranking quality with relevance grading
3. **Semantic Similarity**: Answer quality via embedding similarity
4. **Contextual Precision/Recall**: Context relevance assessment
5. **Hit Rate**: Binary success metric

## Usage Modes

### Build Mode
```bash
python main.py --mode build --fixed-urls fixed_urls.json --num-urls 10 --device cpu
```

### Evaluate Mode
```bash
python main.py --mode evaluate --num-questions 20 --device cpu
```

### Query Mode
```bash
python main.py --mode query --query "What is machine learning?" --device cpu
```

### Full Pipeline
```bash
python main.py --mode full --fixed-urls fixed_urls.json --num-questions 100 --device cpu
```

### Web Interface
```bash
streamlit run ui/app.py
```

## Dependencies
- **Core**: transformers, sentence-transformers, faiss-cpu, rank-bm25
- **NLP**: nltk, spacy
- **Data**: numpy, pandas, scikit-learn
- **Web**: streamlit, plotly
- **APIs**: requests, beautifulsoup4
- **Utilities**: python-dotenv, tqdm

## File Statistics
- **Python modules**: 10 core modules
- **Evaluation modules**: 3 modules
- **Total LOC**: ~3000+ lines of code
- **Functions/Classes**: 100+ public interfaces
- **Documentation**: Comprehensive docstrings and comments

## Notes
- All URLs and metadata preserved for traceability
- Automated evaluation with 100 Q&A pairs
- Results saved in JSON/CSV/HTML formats
- Fully configurable system parameters
- GPU support available (CUDA)
- Models auto-downloaded on first use

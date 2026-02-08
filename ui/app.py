"""
Streamlit UI: Web interface for Hybrid RAG System
"""

import streamlit as st
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import sys
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Hybrid RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-left: 4px solid #0066cc;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []


def load_rag_system():
    """Load or initialize RAG system"""
    try:
        if st.session_state.rag_system is None:
            from src.rag_system import HybridRAGSystem
            st.session_state.rag_system = HybridRAGSystem()
            st.session_state.indices_loaded = False
            
            # Try to load indices if they exist
            indices_path = Path("data/indices")
            if indices_path.exists() and (indices_path / "dense").exists() and (indices_path / "sparse").exists():
                dense_loaded = st.session_state.rag_system.dense_retriever.load(f"{indices_path}/dense")
                sparse_loaded = st.session_state.rag_system.sparse_retriever.load(f"{indices_path}/sparse")
                
                if dense_loaded and sparse_loaded:
                    st.session_state.indices_loaded = True
                    st.success("✓ Loaded existing indices")
                else:
                    st.warning("⚠ Could not load indices. Build corpus first.")
            else:
                st.warning("⚠ No indices found. Please build corpus first using CLI.")
        
        return st.session_state.rag_system
    
    except Exception as e:
        st.error(f"Error loading RAG system: {e}")
        return None


def main():
    """Main Streamlit application"""
    init_session_state()
    
    # Sidebar navigation
    st.sidebar.title("🔍 Hybrid RAG System")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "📊 Query Interface", "📈 Evaluation", 
         "⚙️ System Config", "📚 Documentation"]
    )
    
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Query Interface":
        show_query_interface()
    elif page == "📈 Evaluation":
        show_evaluation()
    elif page == "⚙️ System Config":
        show_system_config()
    elif page == "📚 Documentation":
        show_documentation()


def show_home():
    """Home page"""
    st.title("🔍 Hybrid RAG System")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Dense Retrieval** 
        • Sentence embeddings
        • FAISS indexing
        • Semantic search
        """)
    
    with col2:
        st.info("""
        **Sparse Retrieval**
        • BM25 algorithm
        • Keyword search
        • Lexical matching
        """)
    
    with col3:
        st.info("""
        **RRF Fusion**
        • Reciprocal Rank Fusion
        • Score combination
        • Hybrid retrieval
        """)
    
    st.markdown("---")
    
    st.subheader("📊 System Status")
    
    rag_system = load_rag_system()
    
    if rag_system:
        # Check if indices are loaded
        has_dense = rag_system.dense_retriever.index is not None
        has_sparse = rag_system.sparse_retriever.bm25 is not None
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if has_dense:
                st.success(f"✓ Dense Index")
                st.caption(f"{len(rag_system.dense_retriever.chunks)} chunks")
            else:
                st.error("✗ Dense Index")
                st.caption("Not loaded")
        
        with col2:
            if has_sparse:
                st.success(f"✓ Sparse Index")
                st.caption(f"{len(rag_system.sparse_retriever.chunks)} chunks")
            else:
                st.error("✗ Sparse Index")
                st.caption("Not loaded")
        
        with col3:
            st.info("✓ Models Ready")
            st.caption(f"{rag_system.config.get('dense_model', 'N/A')}")
        
        # Check if build is needed
        if not has_dense or not has_sparse:
            st.markdown("---")
            st.info("⚡ **Build the Corpus**")
            st.markdown("""
            Run this command in terminal to build indices:
            ```bash
            python main.py --mode build --num-urls 10
            ```
            
            This will:
            - Fetch 10 Wikipedia articles
            - Create dense embeddings (384-dim)
            - Build BM25 sparse index
            - Save to `data/indices/`
            
            **Time:** ~10-15 minutes (first run downloads models)
            """)
    
    st.markdown("---")
    
    st.subheader("🚀 Next Steps")
    st.markdown("""
    1. ✅ **Models Loaded**: All transformers ready (all-MiniLM-L6-v2, DistilGPT2)
    2. ⏳ **Build Corpus**: Run build command above to create indices
    3. 🔍 **Query**: Use the Query Interface tab to ask questions
    4. 📊 **Evaluate**: Run evaluation suite for metrics
    """)


def show_query_interface():
    """Query interface page"""
    st.title("📊 Query Interface")
    st.markdown("---")
    
    rag_system = load_rag_system()
    
    if not rag_system:
        st.error("RAG system not loaded")
        return
    
    # Check if indices are loaded
    if rag_system.dense_retriever.index is None:
        st.error("Dense index not loaded. Please build corpus first.")
        return
    
    # Query input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Enter your query:",
            placeholder="What is artificial intelligence?",
            key="query_input"
        )
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    if search_button and query:
        with st.spinner("Processing query..."):
            start_time = time.time()
            result = rag_system.answer_query(query)
            elapsed_time = time.time() - start_time
        
        # Debug: Check result structure
        if 'error' in result:
            st.error(f"Query error: {result.get('answer', 'Unknown error')}")
            return
        
        # Store in history
        st.session_state.query_history.append({
            'query': query,
            'timestamp': datetime.now(),
            'time': elapsed_time
        })
        
        # Display results
        st.markdown("---")
        st.subheader("📝 Answer")
        
        answer_col, metric_col = st.columns([2, 1])
        
        with answer_col:
            answer_text = result.get('answer', 'No answer generated')
            st.info(answer_text)
        
        with metric_col:
            confidence = result.get('confidence', 0.0)
            st.metric("Confidence", f"{confidence:.1%}")
            st.metric("Time", f"{elapsed_time:.2f}s")
        
        # Display retrieval results
        st.markdown("---")
        st.subheader("📚 Retrieved Chunks")
        
        # Get retrieval data
        retrieval_data = result.get('retrieval', {})
        dense_chunks = retrieval_data.get('dense', [])
        sparse_chunks = retrieval_data.get('sparse', [])
        fused_chunks = retrieval_data.get('fused', [])
        
        # Create tabs for different retrieval methods
        tabs = st.tabs(["Dense", "Sparse", "Fused"])
        
        with tabs[0]:
            display_chunks(dense_chunks, "dense_score")
        
        with tabs[1]:
            display_chunks(sparse_chunks, "sparse_score")
        
        with tabs[2]:
            display_chunks(fused_chunks, "rrf_score")
        
        # Display score comparison
        st.markdown("---")
        st.subheader("📊 Score Comparison")
        
        if fused_chunks and len(fused_chunks) > 0:
            scores_data = []
            for chunk in fused_chunks[:5]:
                chunk_title = chunk.get('title', 'Unknown')[:30]
                scores_data.append({
                    'Chunk': chunk_title,
                    'Dense': round(chunk.get('dense_score', 0), 4),
                    'Sparse': round(chunk.get('sparse_score', 0), 4),
                    'RRF': round(chunk.get('rrf_score', 0), 4)
                })
            
            if scores_data:
                scores_df = pd.DataFrame(scores_data)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=scores_df['Chunk'], y=scores_df['Dense'], name='Dense'))
                fig.add_trace(go.Bar(x=scores_df['Chunk'], y=scores_df['Sparse'], name='Sparse'))
                fig.add_trace(go.Bar(x=scores_df['Chunk'], y=scores_df['RRF'], name='RRF'))
                
                fig.update_layout(
                    title="Retrieval Score Comparison",
                    xaxis_title="Chunk",
                    yaxis_title="Score",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No scores to display")
        else:
            st.info("No fused results to compare")
    
    # Query history
    if st.session_state.query_history:
        st.markdown("---")
        st.subheader("📜 Query History")
        
        history_data = []
        for item in st.session_state.query_history[-10:]:  # Last 10
            history_data.append({
                'Query': item['query'][:50],
                'Time': item['timestamp'].strftime("%H:%M:%S"),
                'Duration': f"{item['time']:.3f}s"
            })
        
        st.dataframe(pd.DataFrame(history_data), use_container_width=True)


def show_evaluation():
    """Evaluation page"""
    st.title("📈 Evaluation Dashboard")
    st.markdown("---")
    
    # Load evaluation results if available
    results_dir = Path("results")
    
    if results_dir.exists():
        result_files = sorted(list(results_dir.glob("summary_*.json")), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if result_files:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_file = st.selectbox(
                    "Select evaluation results:",
                    result_files,
                    format_func=lambda x: x.stem
                )
            
            with col2:
                if st.button("🔄 Refresh"):
                    st.rerun()
            
            if selected_file:
                with open(selected_file, 'r') as f:
                    summary = json.load(f)
                
                display_evaluation_results(summary)
        else:
            st.info("No evaluation results found. Run evaluation first.")
    else:
        st.info("No evaluation results directory found. Create one to save results.")
    
    # Evaluation controls
    st.markdown("---")
    st.subheader("⚙️ Run Evaluation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_questions = st.slider("Number of questions", 10, 100, 20)
    
    with col2:
        include_ablation = st.checkbox("Include ablation studies", value=False)
    
    with col3:
        if st.button("▶️ Start Evaluation", key="eval_button"):
            run_evaluation(num_questions, include_ablation)


def display_chunks(chunks: list, score_key: str):
    """Display chunk results"""
    if not chunks:
        st.info("No chunks retrieved")
        return
    
    for i, chunk in enumerate(chunks[:5], 1):
        with st.expander(f"Chunk {i}: {chunk.get('title', 'Unknown')}"):
            st.markdown(f"**Score:** {chunk.get(score_key, 0):.4f}")
            st.markdown(f"**URL:** {chunk.get('url', 'N/A')}")
            st.markdown(f"**Tokens:** {chunk.get('token_count', 0)}")
            st.text(chunk.get('text', 'No text')[:500] + "...")


def run_evaluation(num_questions: int, include_ablation: bool):
    """Run evaluation pipeline with progress tracking"""
    from evaluation.question_generation import QuestionGenerator
    from evaluation.metrics import EvaluationPipeline
    from evaluation.evaluation_pipeline import AblationStudy
    from src.preprocessing import ChunkProcessor
    import pickle
    
    # Load RAG system
    rag_system = load_rag_system()
    if not rag_system:
        st.error("Failed to load RAG system")
        return
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Load or generate Q&A pairs
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Load chunks from corpus
        chunks_path = Path("data/corpus/chunks.json")
        if not chunks_path.exists():
            st.error("No corpus found. Please build corpus first.")
            return
        
        with open(chunks_path, 'r') as f:
            chunks = json.load(f)
        
        status_placeholder.info(f"Generating {num_questions} Q&A pairs...")
        progress_bar.progress(10)
        
        # Generate Q&A pairs
        qg = QuestionGenerator(chunks)
        qa_pairs = qg.generate_factual_questions(num_questions)
        
        if not qa_pairs:
            st.error("Failed to generate Q&A pairs")
            return
        
        status_placeholder.info(f"Evaluating {len(qa_pairs)} questions...")
        progress_bar.progress(20)
        
        # Initialize evaluation pipeline
        eval_pipeline = EvaluationPipeline()
        detailed_results = []
        
        # Evaluate each Q&A pair
        for i, qa_pair in enumerate(qa_pairs):
            progress = 20 + int((i / len(qa_pairs)) * 60)
            progress_bar.progress(progress)
            status_placeholder.info(f"Evaluating question {i+1}/{len(qa_pairs)}...")
            
            try:
                result = rag_system.answer_query(qa_pair['question'])
                evaluation = eval_pipeline.evaluate_single_result(result, qa_pair)
                detailed_results.append(evaluation)
            except Exception as e:
                logger.warning(f"Error evaluating question {i+1}: {e}")
                continue
        
        if not detailed_results:
            st.error("No evaluation results generated")
            return
        
        # Calculate summary statistics
        status_placeholder.info("Computing summary statistics...")
        progress_bar.progress(85)
        
        summary = compute_evaluation_summary(detailed_results)
        
        # Run ablation studies if requested
        ablation_results = {}
        if include_ablation:
            status_placeholder.info("Running ablation studies...")
            progress_bar.progress(90)
            
            ablation = AblationStudy(rag_system)
            test_queries = [qa['question'] for qa in qa_pairs[:5]]
            
            ablation_results = {
                'dense_only': ablation.dense_only(test_queries),
                'sparse_only': ablation.sparse_only(test_queries),
                'hybrid': ablation.hybrid(test_queries)
            }
        
        # Save results
        status_placeholder.info("Saving results...")
        progress_bar.progress(95)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary['timestamp'] = timestamp
        summary['num_questions'] = num_questions
        summary['ablation_included'] = include_ablation
        
        # Save summary
        summary_file = results_dir / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed results
        detailed_file = results_dir / f"detailed_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Save ablation results if available
        if ablation_results:
            ablation_file = results_dir / f"ablation_{timestamp}.json"
            with open(ablation_file, 'w') as f:
                json.dump(ablation_results, f, indent=2, default=str)
        
        progress_bar.progress(100)
        status_placeholder.success(f"✓ Evaluation complete! Results saved to {results_dir}")
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Evaluation Results")
        display_evaluation_results(summary)
        
    except Exception as e:
        st.error(f"Error during evaluation: {e}")
        logger.error(f"Evaluation error: {e}", exc_info=True)


def compute_evaluation_summary(detailed_results: List[Dict]) -> Dict:
    """Compute summary statistics from detailed evaluation results"""
    summary = {}
    
    metric_keys = [
        'mrr_url', 'precision_at_5', 'precision_at_10', 'recall_at_10',
        'hit_rate_at_10', 'ndcg_at_10', 'semantic_similarity', 'answer_length_score'
    ]
    
    for metric in metric_keys:
        values = [r.get(metric, 0) for r in detailed_results if metric in r]
        if values:
            summary[f'{metric}_mean'] = float(np.mean(values))
            summary[f'{metric}_std'] = float(np.std(values))
            summary[f'{metric}_min'] = float(np.min(values))
            summary[f'{metric}_max'] = float(np.max(values))
    
    return summary


def display_evaluation_results(summary: dict):
    """Display evaluation summary"""
    st.subheader("📊 Evaluation Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "MRR (URL)",
            f"{summary.get('mrr_url_mean', 0):.3f}",
            f"±{summary.get('mrr_url_std', 0):.3f}"
        )
    
    with col2:
        st.metric(
            "Precision@10",
            f"{summary.get('precision_at_10_mean', 0):.3f}",
            f"±{summary.get('precision_at_10_std', 0):.3f}"
        )
    
    with col3:
        st.metric(
            "Recall@10",
            f"{summary.get('recall_at_10_mean', 0):.3f}",
            f"±{summary.get('recall_at_10_std', 0):.3f}"
        )
    
    with col4:
        st.metric(
            "NDCG@10",
            f"{summary.get('ndcg_at_10_mean', 0):.3f}",
            f"±{summary.get('ndcg_at_10_std', 0):.3f}"
        )
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Hit Rate@10",
            f"{summary.get('hit_rate_at_10_mean', 0):.3f}",
            f"±{summary.get('hit_rate_at_10_std', 0):.3f}"
        )
    
    with col2:
        st.metric(
            "Semantic Similarity",
            f"{summary.get('semantic_similarity_mean', 0):.3f}",
            f"±{summary.get('semantic_similarity_std', 0):.3f}"
        )
    
    with col3:
        st.metric(
            "Answer Length Score",
            f"{summary.get('answer_length_score_mean', 0):.3f}",
            f"±{summary.get('answer_length_score_std', 0):.3f}"
        )
    
    # Metric distributions
    st.subheader("📈 Metric Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        metrics_to_plot = ['mrr_url', 'precision_at_10', 'recall_at_10']
        metric_data = {}
        
        for metric in metrics_to_plot:
            if f'{metric}_mean' in summary:
                metric_data[metric.replace('_', ' ').title()] = summary[f'{metric}_mean']
        
        if metric_data:
            fig = go.Figure(data=[
                go.Bar(x=list(metric_data.keys()), y=list(metric_data.values()), marker_color='#1f77b4')
            ])
            fig.update_layout(title="Retrieval Metrics", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        quality_metrics = {}
        for metric in ['semantic_similarity', 'answer_length_score']:
            if f'{metric}_mean' in summary:
                quality_metrics[metric.replace('_', ' ').title()] = summary[f'{metric}_mean']
        
        if quality_metrics:
            fig = go.Figure(data=[
                go.Bar(x=list(quality_metrics.keys()), y=list(quality_metrics.values()), marker_color='#2ca02c')
            ])
            fig.update_layout(title="Answer Quality Metrics", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown
    st.subheader("📋 Detailed Metrics")
    metrics_df = pd.DataFrame([
        {'Metric': k.replace('_mean', '').replace('_', ' ').title(), 
         'Mean': f"{v:.4f}",
         'Std Dev': f"{summary.get(k.replace('_mean', '_std'), 0):.4f}"}
        for k, v in summary.items() if k.endswith('_mean')
    ])
    st.dataframe(metrics_df, use_container_width=True)
    
    with col2:
        quality_metrics = ['semantic_similarity', 'contextual_precision']
        quality_data = {}
        
        for metric in quality_metrics:
            if f'{metric}_mean' in summary:
                quality_data[metric] = summary[f'{metric}_mean']
        
        if quality_data:
            fig = go.Figure(data=[
                go.Bar(x=list(quality_data.keys()), y=list(quality_data.values()))
            ])
            fig.update_layout(title="Answer Quality Metrics", height=400)
            st.plotly_chart(fig, use_container_width=True)


def show_system_config():
    """System configuration page"""
    st.title("⚙️ System Configuration")
    st.markdown("---")
    
    rag_system = load_rag_system()
    
    if rag_system:
        config = rag_system.config
        
        st.subheader("📋 Current Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Retrieval Settings")
            
            config['chunk_size'] = st.slider(
                "Chunk Size (tokens)",
                200, 500, config.get('chunk_size', 300)
            )
            
            config['chunk_overlap'] = st.slider(
                "Chunk Overlap (tokens)",
                20, 100, config.get('chunk_overlap', 50)
            )
            
            config['dense_top_k'] = st.slider(
                "Dense Top-K",
                5, 20, config.get('dense_top_k', 10)
            )
            
            config['sparse_top_k'] = st.slider(
                "Sparse Top-K",
                5, 20, config.get('sparse_top_k', 10)
            )
            
            config['final_top_n'] = st.slider(
                "Final Context Chunks",
                1, 10, config.get('final_top_n', 5)
            )
        
        with col2:
            st.subheader("Model Settings")
            
            dense_model = st.selectbox(
                "Dense Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
                index=0 if config.get('dense_model') == 'all-MiniLM-L6-v2' else 1
            )
            config['dense_model'] = dense_model
            
            llm_model = st.selectbox(
                "LLM Model",
                ["distilgpt2", "gpt2", "google/flan-t5-base"],
                index=0 if config.get('llm_model') == 'distilgpt2' else 1
            )
            config['llm_model'] = llm_model
            
            config['device'] = st.selectbox(
                "Device",
                ["cpu", "cuda"],
                index=0 if config.get('device', 'cpu') == 'cpu' else 1
            )
            
            config['max_context_tokens'] = st.slider(
                "Max Context Tokens",
                100, 500, config.get('max_context_tokens', 300)
            )
        
        if st.button("💾 Save Configuration"):
            st.success("Configuration updated!")


def show_documentation():
    """Documentation page"""
    st.title("📚 System Documentation")
    st.markdown("---")
    
    st.markdown("""
    ## Architecture Overview
    
    ### 1. **Dense Retrieval**
    - Uses sentence transformers to embed text chunks
    - Creates FAISS index for efficient similarity search
    - Retrieves documents based on semantic similarity
    
    ### 2. **Sparse Retrieval**
    - Implements BM25 algorithm for keyword matching
    - Uses TF-IDF based ranking
    - Efficient keyword-based search
    
    ### 3. **Reciprocal Rank Fusion (RRF)**
    - Combines dense and sparse results
    - Formula: RRF(d) = Σ 1/(k + rank(d))
    - k = 60 (default constant)
    
    ### 4. **Response Generation**
    - Uses open-source LLMs (DistilGPT2, Flan-T5, etc.)
    - Generates answers from retrieved context
    - Supports various generation configurations
    
    ### 5. **Evaluation Metrics**
    - **MRR (URL-level)**: Measures retrieval effectiveness
    - **Precision@K**: Fraction of relevant documents
    - **Recall@K**: Coverage of relevant documents
    - **Semantic Similarity**: Answer quality assessment
    - **Custom Metrics**: Additional domain-specific metrics
    
    ## Usage Guide
    
    1. **Build Corpus**
       - Fetch Wikipedia articles
       - Preprocess into chunks
       - Store with metadata
    
    2. **Build Indices**
       - Create dense embeddings
       - Build FAISS index
       - Create BM25 index
    
    3. **Query**
       - Enter your question
       - System retrieves and fuses results
       - LLM generates response
       - Returns confidence scores
    
    4. **Evaluate**
       - Run on Q&A dataset
       - Calculate metrics
       - Generate reports
    """)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Main Pipeline Script: Run complete RAG system with evaluation
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='Hybrid RAG System - Complete Pipeline'
    )
    
    parser.add_argument(
        '--mode',
        choices=['build', 'evaluate', 'query', 'full'],
        default='full',
        help='Mode of operation'
    )
    
    parser.add_argument(
        '--fixed-urls',
        default='fixed_urls.json',
        help='Path to fixed URLs JSON file'
    )
    
    parser.add_argument(
        '--num-urls',
        type=int,
        default=10,
        help='Number of URLs to process (for demo)'
    )
    
    parser.add_argument(
        '--num-questions',
        type=int,
        default=20,
        help='Number of Q&A pairs to generate'
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Computing device'
    )
    
    parser.add_argument(
        '--query',
        default=None,
        help='Query to answer (if mode=query)'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting Hybrid RAG System in '{args.mode}' mode")
    
    try:
        if args.mode == 'build' or args.mode == 'full':
            build_system(args)
        
        if args.mode == 'evaluate' or args.mode == 'full':
            evaluate_system(args)
        
        if args.mode == 'query' and args.query:
            query_system(args)
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("✓ Pipeline execution complete")


def build_system(args):
    """Build the RAG system"""
    
    logger.info("=" * 60)
    logger.info("BUILDING RAG SYSTEM")
    logger.info("=" * 60)
    
    from src.rag_system import HybridRAGSystem
    from src.preprocessing import prepare_rag_corpus
    
    # Initialize system
    config = {
        'chunk_size': 300,
        'chunk_overlap': 50,
        'dense_model': 'all-MiniLM-L6-v2',
        'dense_top_k': 10,
        'sparse_top_k': 10,
        'final_top_n': 5,
        'llm_model': 'distilgpt2',
        'max_context_tokens': 300,
        'device': args.device
    }
    
    rag_system = HybridRAGSystem(config=config)
    
    # Load URLs
    logger.info(f"Loading URLs from {args.fixed_urls}")
    fixed_urls = rag_system.collector.load_fixed_urls(args.fixed_urls)
    
    # Determine URL mix
    target_count = args.num_urls
    
    if target_count <= len(fixed_urls):
        # Demo mode / Small scale: Use subset of fixed URLs
        urls_to_process = fixed_urls[:target_count]
        logger.info(f"Processing {len(urls_to_process)} Fixed URLs (Subset mode)")
    else:
        # Full mode / Large scale: Use all fixed + random
        num_fixed = len(fixed_urls)
        num_random = target_count - num_fixed
        
        logger.info(f"Using {num_fixed} Fixed URLs")
        logger.info(f"Fetching {num_random} Random URLs to reach total {target_count}...")
        
        random_urls = rag_system.collector.fetch_random_urls(
            count=num_random,
            fixed_urls=fixed_urls
        )
        
        urls_to_process = fixed_urls + random_urls
        logger.info(f"Total URLs to process: {len(urls_to_process)} ({len(fixed_urls)} Fixed + {len(random_urls)} Random)")
    
    
    # Collect documents
    logger.info("Collecting documents...")
    documents, valid_urls = rag_system.collector.collect_from_urls(urls_to_process)
    logger.info(f"✓ Collected {len(documents)} documents")
    
    # Preprocess
    logger.info("Preprocessing documents into chunks...")
    chunks = prepare_rag_corpus(
        documents,
        chunk_size=config['chunk_size'],
        overlap=config['chunk_overlap']
    )
    logger.info(f"✓ Created {len(chunks)} chunks")
    
    rag_system.chunks = chunks
    
    # Build indices
    logger.info("Building indices...")
    Path('data/indices').mkdir(parents=True, exist_ok=True)
    success = rag_system.build_indices(chunks, save_path='data/indices')
    
    if success:
        logger.info("✓ Indices built successfully")
    else:
        logger.error("✗ Failed to build indices")
        return
    
    # Save system
    rag_system.save_system('data/rag_system')
    logger.info("✓ System saved to data/rag_system")
    
    # Save chunks for later
    Path('data/corpus').mkdir(parents=True, exist_ok=True)
    with open('data/corpus/chunks.json', 'w') as f:
        json.dump(chunks, f, indent=2)
    
    logger.info(f"✓ RAG system build complete")
    

def evaluate_system(args):
    """Evaluate the RAG system"""
    
    logger.info("=" * 60)
    logger.info("EVALUATING RAG SYSTEM")
    logger.info("=" * 60)
    
    from src.rag_system import HybridRAGSystem
    from evaluation.question_generation import QuestionGenerator, save_qa_dataset
    from evaluation.evaluation_pipeline import AutomatedEvaluationPipeline
    
    # Load system
    logger.info("Loading RAG system...")
    rag_system = HybridRAGSystem()
    
    try:
        rag_system.dense_retriever.load('data/indices/dense')
        rag_system.sparse_retriever.load('data/indices/sparse')
        logger.info("✓ Indices loaded")
    except Exception as e:
        logger.error(f"Failed to load indices: {e}")
        logger.info("Please run build mode first")
        return
    
    # Load chunks
    try:
        with open('data/corpus/chunks.json', 'r') as f:
            chunks = json.load(f)
        rag_system.chunks = chunks
        logger.info(f"✓ Loaded {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Failed to load chunks: {e}")
        return
    
    # Generate Q&A pairs
    logger.info(f"Generating Q&A pairs...")
    generator = QuestionGenerator(chunks)
    qa_pairs = generator.generate_qa_dataset(total_questions=args.num_questions)
    
    Path('data/qa').mkdir(parents=True, exist_ok=True)
    save_qa_dataset(qa_pairs, 'data/qa/qa_dataset.json')
    logger.info(f"✓ Generated {len(qa_pairs)} Q&A pairs")
    
    # Run evaluation
    logger.info("Running evaluation pipeline...")
    eval_pipeline = AutomatedEvaluationPipeline(rag_system, 'data/qa/qa_dataset.json')
    
    evaluations, summary = eval_pipeline.run_evaluation(
        num_questions=min(args.num_questions, len(qa_pairs)),
        save_results=True
    )
    
    logger.info(f"✓ Evaluation complete")
    
    # Print summary
    print_evaluation_summary(summary)


def query_system(args):
    """Query the RAG system"""
    
    logger.info("=" * 60)
    logger.info("QUERYING RAG SYSTEM")
    logger.info("=" * 60)
    
    from src.rag_system import HybridRAGSystem
    
    # Load system
    logger.info("Loading RAG system...")
    rag_system = HybridRAGSystem()
    
    try:
        rag_system.dense_retriever.load('data/indices/dense')
        rag_system.sparse_retriever.load('data/indices/sparse')
        logger.info("✓ System loaded")
    except Exception as e:
        logger.error(f"Failed to load system: {e}")
        return
    
    # Answer query
    logger.info(f"Query: {args.query}")
    result = rag_system.answer_query(args.query)
    
    # Display result
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result['answer'])
    print("\n" + "=" * 60)
    print("METADATA")
    print("=" * 60)
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Time: {result['timings']['total']:.3f}s")
    print(f"Context tokens: {result['timings']['context_tokens']}")
    
    print("\n" + "=" * 60)
    print("TOP RETRIEVED CHUNKS")
    print("=" * 60)
    for i, chunk in enumerate(result['retrieval']['fused'][:3], 1):
        print(f"\n{i}. {chunk['title']}")
        print(f"   Score: {chunk['rrf_score']:.4f}")
        print(f"   URL: {chunk['url']}")
        print(f"   {chunk['text'][:200]}...")


def print_evaluation_summary(summary):
    """Print evaluation summary"""
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\nQuestions: {summary.get('total_questions', 0)}")
    print(f"Avg Time: {summary.get('avg_time', 0):.3f}s")
    
    print(f"\nRetrieval Metrics:")
    print(f"  MRR (URL-level): {summary.get('mrr_url_mean', 0):.4f} ± {summary.get('mrr_url_std', 0):.4f}")
    print(f"  Precision@10: {summary.get('precision_at_10_mean', 0):.4f}")
    print(f"  Recall@10: {summary.get('recall_at_10_mean', 0):.4f}")
    print(f"  NDCG@10: {summary.get('ndcg_at_10_mean', 0):.4f}")
    print(f"  Hit Rate@10: {summary.get('hit_rate_at_10_mean', 0):.4f}")
    
    print(f"\nAnswer Quality Metrics:")
    print(f"  Semantic Similarity: {summary.get('semantic_similarity_mean', 0):.4f}")
    print(f"  Context Precision: {summary.get('contextual_precision_mean', 0):.4f}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()

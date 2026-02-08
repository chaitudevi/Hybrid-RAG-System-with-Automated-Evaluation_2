"""
Main RAG System: Orchestrates all components
"""

import json
import time
from typing import List, Dict, Tuple
import logging
from pathlib import Path

from .data_collection import WikipediaCollector
from .preprocessing import ChunkProcessor, prepare_rag_corpus
from .dense_retrieval import DenseRetriever
from .sparse_retrieval import SparseRetriever
from .fusion import ReciprocalRankFusion
from .generation import ResponseGenerator, ConfidenceEstimator

logger = logging.getLogger(__name__)


class HybridRAGSystem:
    """Complete Hybrid RAG System combining all components"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize RAG system with configuration
        
        Args:
            config: Configuration dictionary
        """
        # Default configuration
        self.config = config or {
            'chunk_size': 300,
            'chunk_overlap': 50,
            'dense_model': 'all-MiniLM-L6-v2',
            'dense_top_k': 10,
            'sparse_top_k': 10,
            'final_top_n': 5,
            'llm_model': 'distilgpt2',
            'max_context_tokens': 300,
            'device': 'cpu'
        }
        
        # Initialize components
        self.collector = WikipediaCollector()
        self.preprocessor = ChunkProcessor(
            chunk_size=self.config['chunk_size'],
            overlap=self.config['chunk_overlap']
        )
        self.dense_retriever = DenseRetriever(
            model_name=self.config['dense_model'],
            device=self.config['device']
        )
        self.sparse_retriever = SparseRetriever()
        self.fusion = ReciprocalRankFusion()
        self.generator = ResponseGenerator(
            model_name=self.config['llm_model'],
            device=self.config['device']
        )
        self.confidence_estimator = ConfidenceEstimator()
        
        self.chunks = []
        self.documents = []
        
        # Try to load existing indices
        indices_path = Path("data/indices")
        if indices_path.exists():
            try:
                dense_path = indices_path / "dense"
                sparse_path = indices_path / "sparse"
                if dense_path.exists() and sparse_path.exists():
                    self.dense_retriever.load(str(dense_path))
                    self.sparse_retriever.load(str(sparse_path))
                    logger.info("Loaded existing indices from disk")
            except Exception as e:
                logger.warning(f"Could not load existing indices: {e}")
        
        logger.info("HybridRAGSystem initialized")
    
    def build_corpus(self, fixed_urls: List[str], 
                    random_urls: List[str] = None,
                    save_path: str = "data/corpus") -> List[Dict]:
        """
        Build RAG corpus from URLs
        
        Args:
            fixed_urls: Fixed set of URLs
            random_urls: Random set of URLs for this run
            save_path: Path to save preprocessed data
            
        Returns:
            Preprocessed chunks
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        all_urls = fixed_urls.copy()
        if random_urls:
            all_urls.extend(random_urls)
        
        logger.info(f"Building corpus from {len(all_urls)} URLs...")
        
        # Collect documents
        documents, valid_urls = self.collector.collect_from_urls(all_urls)
        self.documents = documents
        
        logger.info(f"Collected {len(documents)} valid documents")
        
        # Preprocess into chunks
        chunks = prepare_rag_corpus(
            documents,
            chunk_size=self.config['chunk_size'],
            overlap=self.config['chunk_overlap']
        )
        self.chunks = chunks
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Save chunks
        with open(f"{save_path}/chunks.json", 'w') as f:
            json.dump(chunks, f, indent=2)
        
        with open(f"{save_path}/documents.json", 'w') as f:
            json.dump(documents, f, indent=2)
        
        return chunks
    
    def build_indices(self, chunks: List[Dict] = None,
                     save_path: str = "data/indices") -> bool:
        """
        Build dense and sparse indices
        
        Args:
            chunks: Chunks to index (uses self.chunks if not provided)
            save_path: Path to save indices
            
        Returns:
            Success status
        """
        if chunks is None:
            chunks = self.chunks
        
        if not chunks:
            logger.error("No chunks to index")
            return False
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        try:
            # Build dense index
            logger.info("Building dense retrieval index...")
            self.dense_retriever.build_index(chunks, f"{save_path}/dense")
            
            # Build sparse index
            logger.info("Building sparse retrieval index...")
            self.sparse_retriever.build_index(chunks, f"{save_path}/sparse")
            
            logger.info("Indices built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Index building failed: {e}")
            return False
    
    def answer_query(self, query: str) -> Dict:
        """
        Answer a query using hybrid retrieval and generation
        
        Args:
            query: User query
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        try:
            # Dense retrieval
            dense_results = self.dense_retriever.retrieve(
                query, 
                top_k=self.config['dense_top_k']
            )
            
            # Sparse retrieval
            sparse_results = self.sparse_retriever.retrieve(
                query,
                top_k=self.config['sparse_top_k']
            )
            
            # RRF fusion
            fused_results = self.fusion.compute_rrf_score(
                dense_results,
                sparse_results,
                alpha=0.5
            )
            
            # Select top-N for generation context
            context_chunks = fused_results[:self.config['final_top_n']]
            
            # Generate response
            generation_result = self.generator.generate(
                query,
                context_chunks,
                max_length=256
            )
            
            # Estimate confidence
            context_relevance = sum([c.get('rrf_score', 0) 
                                    for c in context_chunks]) / len(context_chunks)
            confidence = self.confidence_estimator.estimate_confidence(
                generation_result['answer'],
                context_relevance=context_relevance,
                context_coverage=0.7
            )
            
            # Compile result
            result = {
                'query': query,
                'answer': generation_result['answer'],
                'confidence': confidence,
                'retrieval': {
                    'dense': dense_results[:3],
                    'sparse': sparse_results[:3],
                    'fused': context_chunks
                },
                'timings': {
                    'total': time.time() - start_time,
                    'context_tokens': generation_result.get('context_tokens_used', 0)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Query answering failed: {e}")
            return {
                'query': query,
                'answer': f"Error: {e}",
                'error': str(e)
            }
    
    def batch_answer_queries(self, queries: List[str]) -> List[Dict]:
        """
        Answer multiple queries
        
        Args:
            queries: List of queries
            
        Returns:
            List of results
        """
        results = []
        for i, query in enumerate(queries):
            logger.info(f"Answering query {i+1}/{len(queries)}...")
            result = self.answer_query(query)
            results.append(result)
        
        return results
    
    def save_system(self, save_path: str = "data/rag_system"):
        """
        Save system state including indices and configuration
        
        Args:
            save_path: Path to save system
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(f"{save_path}/config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"RAG system saved to {save_path}")
    
    def load_system(self, load_path: str = "data/rag_system"):
        """
        Load system state from disk
        
        Args:
            load_path: Path to load from
        """
        # Load configuration
        with open(f"{load_path}/config.json", 'r') as f:
            self.config = json.load(f)
        
        # Load indices
        self.dense_retriever.load(f"{load_path}/indices/dense")
        self.sparse_retriever.load(f"{load_path}/indices/sparse")
        
        logger.info(f"RAG system loaded from {load_path}")


if __name__ == "__main__":
    logger.info("Hybrid RAG System module initialized")

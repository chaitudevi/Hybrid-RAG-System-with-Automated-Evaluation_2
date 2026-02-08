"""
Fusion Module: Reciprocal Rank Fusion (RRF) for combining retrieval results
"""

from typing import List, Dict
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ReciprocalRankFusion:
    """Combine dense and sparse retrieval results using RRF"""
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF
        
        Args:
            k: Constant parameter for RRF (typically 60)
        """
        self.k = k
    
    def compute_rrf_score(self, dense_results: List[Dict], 
                         sparse_results: List[Dict],
                         alpha: float = 0.5) -> List[Dict]:
        """
        Combine dense and sparse results using RRF
        
        Formula: RRF_score(d) = Σ 1/(k + rank_i(d))
        
        Args:
            dense_results: Results from dense retriever with 'rank' key
            sparse_results: Results from sparse retriever with 'rank' key
            alpha: Weight for dense results (1-alpha for sparse)
            
        Returns:
            List of fused results sorted by RRF score
        """
        # Create score dictionary
        chunk_scores = defaultdict(lambda: {'dense': 0, 'sparse': 0, 'chunk': None})
        
        # Process dense results
        for result in dense_results:
            chunk_id = result['chunk_id']
            rank = result.get('rank', len(dense_results) + 1)
            score = 1.0 / (self.k + rank)
            chunk_scores[chunk_id]['dense'] = score
            chunk_scores[chunk_id]['chunk'] = result
        
        # Process sparse results
        for result in sparse_results:
            chunk_id = result['chunk_id']
            rank = result.get('rank', len(sparse_results) + 1)
            score = 1.0 / (self.k + rank)
            chunk_scores[chunk_id]['sparse'] = score
            if chunk_scores[chunk_id]['chunk'] is None:
                chunk_scores[chunk_id]['chunk'] = result
        
        # Compute combined RRF scores
        fused_results = []
        for chunk_id, scores in chunk_scores.items():
            rrf_score = (alpha * scores['dense'] + 
                        (1 - alpha) * scores['sparse'])
            
            chunk = scores['chunk'].copy()
            chunk['rrf_score'] = rrf_score
            chunk['dense_component'] = scores['dense']
            chunk['sparse_component'] = scores['sparse']
            
            fused_results.append(chunk)
        
        # Sort by RRF score
        fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        # Add final rank
        for idx, result in enumerate(fused_results):
            result['final_rank'] = idx + 1
        
        return fused_results
    
    def simple_rrf(self, dense_results: List[Dict], 
                  sparse_results: List[Dict]) -> List[Dict]:
        """
        Simple RRF without weighting (equal weight to dense and sparse)
        
        Args:
            dense_results: Results from dense retriever
            sparse_results: Results from sparse retriever
            
        Returns:
            Fused results
        """
        return self.compute_rrf_score(dense_results, sparse_results, alpha=0.5)
    
    def weighted_rrf(self, dense_results: List[Dict],
                    sparse_results: List[Dict],
                    dense_weight: float = 0.6) -> List[Dict]:
        """
        Weighted RRF with custom weights
        
        Args:
            dense_results: Results from dense retriever
            sparse_results: Results from sparse retriever
            dense_weight: Weight for dense results [0, 1]
            
        Returns:
            Fused results
        """
        if not (0 <= dense_weight <= 1):
            raise ValueError("dense_weight must be between 0 and 1")
        
        return self.compute_rrf_score(dense_results, sparse_results, 
                                     alpha=dense_weight)


class EnsembleRetriever:
    """Ensemble multiple retrieval methods"""
    
    def __init__(self, k_rrf: int = 60):
        """
        Initialize ensemble retriever
        
        Args:
            k_rrf: RRF constant parameter
        """
        self.rrf = ReciprocalRankFusion(k=k_rrf)
        self.methods = {}
    
    def register_method(self, name: str, retriever):
        """
        Register a retrieval method
        
        Args:
            name: Method name
            retriever: Retriever instance with retrieve() method
        """
        self.methods[name] = retriever
        logger.info(f"Registered retrieval method: {name}")
    
    def retrieve_with_fusion(self, query: str, top_k: int = 10) -> Dict:
        """
        Retrieve and fuse results from all methods
        
        Args:
            query: Query string
            top_k: Number of final results
            
        Returns:
            Dictionary with individual and fused results
        """
        all_results = {}
        method_results = {}
        
        # Retrieve from each method
        for name, retriever in self.methods.items():
            results = retriever.retrieve(query, top_k=top_k)
            all_results[name] = results
            method_results[name] = results
        
        # Fuse results (if at least 2 methods available)
        fused = None
        if len(all_results) >= 2:
            # Get first two results for RRF
            method_names = list(all_results.keys())
            dense_results = all_results[method_names[0]]
            sparse_results = all_results[method_names[1]]
            
            fused = self.rrf.simple_rrf(dense_results, sparse_results)
            fused = fused[:top_k]  # Keep only top_k
        
        return {
            'individual': method_results,
            'fused': fused if fused else all_results.get(list(all_results.keys())[0], [])
        }


class RRFAnalyzer:
    """Analyze RRF scores and contributions"""
    
    @staticmethod
    def analyze_score_contributions(fused_results: List[Dict]) -> List[Dict]:
        """
        Analyze how much dense vs sparse contribute to each result
        
        Args:
            fused_results: Results from RRF fusion
            
        Returns:
            Analysis of contributions
        """
        analysis = []
        
        for result in fused_results:
            rrf_score = result.get('rrf_score', 0)
            dense_component = result.get('dense_component', 0)
            sparse_component = result.get('sparse_component', 0)
            
            total = dense_component + sparse_component
            dense_pct = (dense_component / total * 100) if total > 0 else 0
            sparse_pct = (sparse_component / total * 100) if total > 0 else 0
            
            analysis.append({
                'chunk_id': result.get('chunk_id'),
                'rrf_score': rrf_score,
                'dense_contribution': dense_pct,
                'sparse_contribution': sparse_pct,
                'rank': result.get('final_rank')
            })
        
        return analysis


if __name__ == "__main__":
    logger.info("Reciprocal Rank Fusion module initialized")

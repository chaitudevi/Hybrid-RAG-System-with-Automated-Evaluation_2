"""
Metrics Module: Evaluation metrics for RAG system
"""

import json
import logging
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate evaluation metrics for RAG system"""
    
    @staticmethod
    def calculate_mrr_url_level(results: List[Dict], 
                               ground_truth_urls: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank at URL level
        
        Mandatory Metric: Find rank of first correct URL in results
        
        Args:
            results: Retrieved chunks with 'url' keys
            ground_truth_urls: URLs that contain correct answer
            
        Returns:
            Reciprocal rank (0-1)
        """
        retrieved_urls = [r['url'] for r in results]
        
        for rank, url in enumerate(retrieved_urls, 1):
            if url in ground_truth_urls:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def calculate_precision_at_k(results: List[Dict],
                                ground_truth_urls: List[str],
                                k: int = 10) -> float:
        """
        Calculate Precision@K - retrieval quality
        
        Args:
            results: Retrieved chunks
            ground_truth_urls: Correct URLs
            k: Number of top results to consider
            
        Returns:
            Precision score [0, 1]
        """
        top_k_results = results[:k]
        retrieved_urls = [r['url'] for r in top_k_results]
        
        correct = sum(1 for url in retrieved_urls if url in ground_truth_urls)
        
        return correct / k if k > 0 else 0.0
    
    @staticmethod
    def calculate_recall_at_k(results: List[Dict],
                             ground_truth_urls: List[str],
                             k: int = 10) -> float:
        """
        Calculate Recall@K - retrieval quality
        
        Args:
            results: Retrieved chunks
            ground_truth_urls: Correct URLs
            k: Number of top results to consider
            
        Returns:
            Recall score [0, 1]
        """
        top_k_results = results[:k]
        retrieved_urls = set(r['url'] for r in top_k_results)
        
        correct = sum(1 for url in ground_truth_urls if url in retrieved_urls)
        
        return correct / len(ground_truth_urls) if len(ground_truth_urls) > 0 else 0.0
    
    @staticmethod
    def calculate_hit_rate(results: List[Dict],
                          ground_truth_urls: List[str],
                          k: int = 10) -> float:
        """
        Calculate Hit Rate - whether any correct URL is in top-k
        
        Args:
            results: Retrieved chunks
            ground_truth_urls: Correct URLs
            k: Number of top results
            
        Returns:
            Hit rate [0, 1]
        """
        top_k_results = results[:k]
        retrieved_urls = set(r['url'] for r in top_k_results)
        
        has_correct = any(url in retrieved_urls for url in ground_truth_urls)
        
        return 1.0 if has_correct else 0.0
    
    @staticmethod
    def calculate_ndcg(results: List[Dict],
                      ground_truth_urls: List[str],
                      k: int = 10) -> float:
        """
        Calculate NDCG (Normalized Discounted Cumulative Gain)
        
        Args:
            results: Retrieved chunks
            ground_truth_urls: Correct URLs
            k: Number of top results
            
        Returns:
            NDCG score [0, 1]
        """
        # DCG calculation
        dcg = 0.0
        top_k_results = results[:k]
        
        for rank, result in enumerate(top_k_results, 1):
            if result['url'] in ground_truth_urls:
                dcg += 1.0 / np.log2(rank + 1)
        
        # IDCG calculation (perfect ranking)
        idcg = 0.0
        for rank in range(1, min(len(ground_truth_urls), k) + 1):
            idcg += 1.0 / np.log2(rank + 1)
        
        return (dcg / idcg) if idcg > 0 else 0.0


class AnswerQualityMetrics:
    """Metrics for evaluating answer quality"""
    
    @staticmethod
    def calculate_exact_match(generated: str, reference: str,
                            case_sensitive: bool = False) -> float:
        """
        Exact Match (EM) - simple string match
        
        Args:
            generated: Generated answer
            reference: Reference answer
            case_sensitive: Whether to use case-sensitive matching
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        if not case_sensitive:
            generated = generated.lower()
            reference = reference.lower()
        
        return 1.0 if generated == reference else 0.0
    
    @staticmethod
    def calculate_semantic_similarity(generated: str, reference: str,
                                     embeddings_fn=None) -> float:
        """
        Semantic Similarity - using embeddings
        
        Args:
            generated: Generated answer
            reference: Reference answer
            embeddings_fn: Optional function to get embeddings
            
        Returns:
            Cosine similarity score [0, 1]
        """
        if embeddings_fn is None:
            # Simple token-based similarity as fallback
            gen_words = set(generated.lower().split())
            ref_words = set(reference.lower().split())
            
            if len(gen_words | ref_words) == 0:
                return 0.0
            
            intersection = len(gen_words & ref_words)
            union = len(gen_words | ref_words)
            
            return intersection / union  # Jaccard similarity
        else:
            # Use embedding-based similarity
            gen_emb = embeddings_fn(generated)
            ref_emb = embeddings_fn(reference)
            
            similarity = cosine_similarity(
                gen_emb.reshape(1, -1),
                ref_emb.reshape(1, -1)
            )[0, 0]
            
            return float(similarity)
    
    @staticmethod
    def calculate_answer_length_score(generated: str,
                                     reference: str) -> float:
        """
        Answer Length Score - how well answer length matches reference
        
        Args:
            generated: Generated answer
            reference: Reference answer
            
        Returns:
            Score [0, 1]
        """
        gen_len = len(generated.split())
        ref_len = len(reference.split())
        
        if ref_len == 0:
            return 0.0
        
        ratio = min(gen_len, ref_len) / max(gen_len, ref_len)
        return ratio


class ContextRelevanceMetrics:
    """Metrics for evaluating context relevance"""
    
    @staticmethod
    def calculate_contextual_precision(retrieved_chunks: List[Dict],
                                      ground_truth_urls: List[str]) -> float:
        """
        Contextual Precision - fraction of relevant chunks in retrieved set
        
        Args:
            retrieved_chunks: Retrieved chunks
            ground_truth_urls: URLs with correct answers
            
        Returns:
            Precision score [0, 1]
        """
        if not retrieved_chunks:
            return 0.0
        
        relevant = sum(1 for chunk in retrieved_chunks 
                      if chunk.get('url') in ground_truth_urls)
        
        return relevant / len(retrieved_chunks)
    
    @staticmethod
    def calculate_contextual_recall(retrieved_chunks: List[Dict],
                                   ground_truth_chunks: List[str]) -> float:
        """
        Contextual Recall - fraction of ground truth chunks retrieved
        
        Args:
            retrieved_chunks: Retrieved chunks
            ground_truth_chunks: Ground truth chunk IDs
            
        Returns:
            Recall score [0, 1]
        """
        if not ground_truth_chunks:
            return 0.0
        
        retrieved_ids = {chunk.get('chunk_id') for chunk in retrieved_chunks}
        relevant = sum(1 for chunk_id in ground_truth_chunks 
                      if chunk_id in retrieved_ids)
        
        return relevant / len(ground_truth_chunks)


class EvaluationPipeline:
    """Complete evaluation pipeline"""
    
    def __init__(self):
        """Initialize evaluation pipeline"""
        self.metrics_calc = MetricsCalculator()
        self.answer_quality = AnswerQualityMetrics()
        self.context_relevance = ContextRelevanceMetrics()
    
    def evaluate_single_result(self, result: Dict, qa_pair: Dict) -> Dict:
        """
        Evaluate a single RAG result
        
        Args:
            result: RAG system result
            qa_pair: Q&A pair with ground truth
            
        Returns:
            Evaluation metrics dictionary
        """
        retrieved_chunks = result.get('retrieval', {}).get('fused', [])
        generated_answer = result.get('answer', '')
        
        ground_truth_urls = [qa_pair.get('source_url')]
        if 'source_urls' in qa_pair:
            ground_truth_urls = qa_pair['source_urls']
        
        ground_truth_urls = [u for u in ground_truth_urls if u]
        
        # Calculate metrics
        evaluation = {
            'question_id': qa_pair.get('question_id'),
            'question': qa_pair.get('question'),
            'question_type': qa_pair.get('question_type'),
            
            # Retrieval metrics
            'mrr_url': self.metrics_calc.calculate_mrr_url_level(
                retrieved_chunks, ground_truth_urls
            ),
            'precision_at_5': self.metrics_calc.calculate_precision_at_k(
                retrieved_chunks, ground_truth_urls, k=5
            ),
            'precision_at_10': self.metrics_calc.calculate_precision_at_k(
                retrieved_chunks, ground_truth_urls, k=10
            ),
            'recall_at_10': self.metrics_calc.calculate_recall_at_k(
                retrieved_chunks, ground_truth_urls, k=10
            ),
            'hit_rate_at_10': self.metrics_calc.calculate_hit_rate(
                retrieved_chunks, ground_truth_urls, k=10
            ),
            'ndcg_at_10': self.metrics_calc.calculate_ndcg(
                retrieved_chunks, ground_truth_urls, k=10
            ),
            
            # Answer quality metrics
            'semantic_similarity': self.answer_quality.calculate_semantic_similarity(
                generated_answer,
                qa_pair.get('ground_truth_answer', '')
            ),
            'answer_length_score': self.answer_quality.calculate_answer_length_score(
                generated_answer,
                qa_pair.get('ground_truth_answer', '')
            ),
            
            # Context metrics
            'contextual_precision': self.context_relevance.calculate_contextual_precision(
                retrieved_chunks, ground_truth_urls
            ),
            
            # Response metadata
            'time_taken': result.get('timings', {}).get('total', 0),
            'confidence': result.get('confidence', 0),
        }
        
        return evaluation
    
    def evaluate_batch(self, results: List[Dict],
                       qa_pairs: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Evaluate batch of results
        
        Args:
            results: List of RAG results
            qa_pairs: List of Q&A pairs
            
        Returns:
            Tuple of (detailed_evaluations, summary_metrics)
        """
        evaluations = []
        
        for result, qa_pair in zip(results, qa_pairs):
            evaluation = self.evaluate_single_result(result, qa_pair)
            evaluations.append(evaluation)
        
        # Calculate summary metrics
        summary = self._calculate_summary(evaluations)
        
        return evaluations, summary
    
    @staticmethod
    def _calculate_summary(evaluations: List[Dict]) -> Dict:
        """Calculate summary statistics"""
        summary = {}
        
        metrics = [
            'mrr_url', 'precision_at_5', 'precision_at_10',
            'recall_at_10', 'hit_rate_at_10', 'ndcg_at_10',
            'semantic_similarity', 'answer_length_score',
            'contextual_precision', 'confidence'
        ]
        
        for metric in metrics:
            values = [e[metric] for e in evaluations if metric in e]
            
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_median'] = np.median(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
        
        # Overall metrics
        summary['total_questions'] = len(evaluations)
        summary['avg_time'] = np.mean([e.get('time_taken', 0) for e in evaluations])
        
        return summary


if __name__ == "__main__":
    logger.info("Metrics Calculator initialized")

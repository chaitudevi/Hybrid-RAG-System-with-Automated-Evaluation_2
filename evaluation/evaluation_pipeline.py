"""
Evaluation Pipeline: Automated evaluation with visualization and reporting
"""

import json
import logging
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.question_generation import load_qa_dataset
from evaluation.metrics import EvaluationPipeline
from src.rag_system import HybridRAGSystem

logger = logging.getLogger(__name__)


class AblationStudy:
    """Perform ablation studies on RAG components"""
    
    def __init__(self, rag_system: HybridRAGSystem):
        """
        Initialize ablation study
        
        Args:
            rag_system: RAG system instance
        """
        self.rag_system = rag_system
        self.results = {}
    
    def dense_only(self, queries: List[str]) -> List[Dict]:
        """
        Test with dense retrieval only
        
        Args:
            queries: List of queries
            
        Returns:
            Results using only dense retrieval
        """
        results = []
        
        for query in queries:
            dense_results = self.rag_system.dense_retriever.retrieve(
                query, 
                top_k=self.rag_system.config.get('final_top_n', 5)
            )
            
            generation_result = self.rag_system.generator.generate(
                query,
                dense_results
            )
            
            result = {
                'query': query,
                'answer': generation_result['answer'],
                'retrieval_method': 'dense_only',
                'chunks_used': len(dense_results)
            }
            
            results.append(result)
        
        self.results['dense_only'] = results
        return results
    
    def sparse_only(self, queries: List[str]) -> List[Dict]:
        """
        Test with sparse retrieval only
        
        Args:
            queries: List of queries
            
        Returns:
            Results using only sparse retrieval
        """
        results = []
        
        for query in queries:
            sparse_results = self.rag_system.sparse_retriever.retrieve(
                query,
                top_k=self.rag_system.config.get('final_top_n', 5)
            )
            
            generation_result = self.rag_system.generator.generate(
                query,
                sparse_results
            )
            
            result = {
                'query': query,
                'answer': generation_result['answer'],
                'retrieval_method': 'sparse_only',
                'chunks_used': len(sparse_results)
            }
            
            results.append(result)
        
        self.results['sparse_only'] = results
        return results
    
    def hybrid(self, queries: List[str]) -> List[Dict]:
        """
        Test with hybrid RRF approach
        
        Args:
            queries: List of queries
            
        Returns:
            Results using hybrid approach
        """
        results = [self.rag_system.answer_query(q) for q in queries]
        self.results['hybrid'] = results
        return results
    
    def compare_k_values(self, queries: List[str], 
                        k_values: List[int] = [5, 10, 15]) -> Dict:
        """
        Compare performance with different K values
        
        Args:
            queries: List of queries
            k_values: Different K values to test
            
        Returns:
            Dictionary of results for each K
        """
        comparison = {}
        
        for k in k_values:
            original_k = self.rag_system.config['dense_top_k']
            self.rag_system.config['dense_top_k'] = k
            self.rag_system.config['sparse_top_k'] = k
            
            results = self.hybrid(queries)
            comparison[f'k_{k}'] = results
            
            # Restore
            self.rag_system.config['dense_top_k'] = original_k
            self.rag_system.config['sparse_top_k'] = original_k
        
        return comparison


class ErrorAnalyzer:
    """Analyze and categorize failures"""
    
    @staticmethod
    def categorize_errors(evaluations: List[Dict]) -> Dict:
        """
        Categorize evaluation errors
        
        Args:
            evaluations: List of evaluation results
            
        Returns:
            Dictionary with error categories
        """
        categories = {
            'retrieval_failure': [],
            'generation_failure': [],
            'confidence_mismatch': [],
            'semantic_gap': [],
            'context_insufficient': []
        }
        
        for eval_result in evaluations:
            # Retrieval failure: low MRR
            if eval_result.get('mrr_url', 0) == 0:
                categories['retrieval_failure'].append(eval_result)
            
            # Generation failure: low answer quality
            if eval_result.get('semantic_similarity', 1) < 0.3:
                categories['generation_failure'].append(eval_result)
            
            # Confidence mismatch
            if (eval_result.get('confidence', 0) > 0.7 and 
                eval_result.get('semantic_similarity', 0) < 0.4):
                categories['confidence_mismatch'].append(eval_result)
            
            # Semantic gap
            if (eval_result.get('contextual_precision', 1) > 0.7 and
                eval_result.get('semantic_similarity', 0) < 0.5):
                categories['semantic_gap'].append(eval_result)
            
            # Context insufficient
            if eval_result.get('recall_at_10', 0) < 0.3:
                categories['context_insufficient'].append(eval_result)
        
        return categories
    
    @staticmethod
    def error_summary_by_question_type(evaluations: List[Dict]) -> Dict:
        """
        Summarize errors by question type
        
        Args:
            evaluations: List of evaluations
            
        Returns:
            Summary grouped by question type
        """
        summary = {}
        
        for eval_result in evaluations:
            q_type = eval_result.get('question_type', 'unknown')
            
            if q_type not in summary:
                summary[q_type] = {
                    'total': 0,
                    'avg_mrr': [],
                    'avg_similarity': [],
                    'avg_time': []
                }
            
            summary[q_type]['total'] += 1
            summary[q_type]['avg_mrr'].append(eval_result.get('mrr_url', 0))
            summary[q_type]['avg_similarity'].append(
                eval_result.get('semantic_similarity', 0)
            )
            summary[q_type]['avg_time'].append(eval_result.get('time_taken', 0))
        
        # Calculate averages
        for q_type in summary:
            summary[q_type]['avg_mrr'] = np.mean(summary[q_type]['avg_mrr'])
            summary[q_type]['avg_similarity'] = np.mean(summary[q_type]['avg_similarity'])
            summary[q_type]['avg_time'] = np.mean(summary[q_type]['avg_time'])
        
        return summary


class AutomatedEvaluationPipeline:
    """Complete automated evaluation pipeline"""
    
    def __init__(self, rag_system: HybridRAGSystem, 
                 qa_dataset_path: str = "data/qa_dataset.json"):
        """
        Initialize evaluation pipeline
        
        Args:
            rag_system: RAG system instance
            qa_dataset_path: Path to Q&A dataset
        """
        self.rag_system = rag_system
        self.qa_dataset = load_qa_dataset(qa_dataset_path)
        self.evaluation_pipeline = EvaluationPipeline()
        self.ablation_study = AblationStudy(rag_system)
        self.error_analyzer = ErrorAnalyzer()
    
    def run_evaluation(self, num_questions: int = None,
                      save_results: bool = True) -> Tuple[List[Dict], Dict]:
        """
        Run complete evaluation
        
        Args:
            num_questions: Number of questions to evaluate (None = all)
            save_results: Whether to save results to disk
            
        Returns:
            Tuple of (detailed results, summary)
        """
        qa_pairs = self.qa_dataset[:num_questions] if num_questions else self.qa_dataset
        
        logger.info(f"Running evaluation on {len(qa_pairs)} questions...")
        
        # Generate answers
        results = self.rag_system.batch_answer_queries(
            [q['question'] for q in qa_pairs]
        )
        
        # Evaluate
        evaluations, summary = self.evaluation_pipeline.evaluate_batch(
            results, qa_pairs
        )
        
        # Error analysis
        error_categories = self.error_analyzer.categorize_errors(evaluations)
        error_by_type = self.error_analyzer.error_summary_by_question_type(evaluations)
        
        summary['error_analysis'] = {
            'categories': {k: len(v) for k, v in error_categories.items()},
            'by_question_type': error_by_type
        }
        
        if save_results:
            self._save_results(evaluations, summary)
        
        return evaluations, summary
    
    def run_ablation(self, num_questions: int = 20,
                    save_results: bool = True) -> Dict:
        """
        Run ablation studies
        
        Args:
            num_questions: Number of questions to test
            save_results: Whether to save results
            
        Returns:
            Ablation study results
        """
        qa_pairs = self.qa_dataset[:num_questions]
        queries = [q['question'] for q in qa_pairs]
        
        logger.info("Running ablation studies...")
        
        ablation_results = {
            'dense_only': self.ablation_study.dense_only(queries),
            'sparse_only': self.ablation_study.sparse_only(queries),
            'hybrid': self.ablation_study.hybrid(queries)
        }
        
        if save_results:
            self._save_ablation_results(ablation_results)
        
        return ablation_results
    
    def _save_results(self, evaluations: List[Dict], summary: Dict):
        """Save evaluation results"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        with open(f"{results_dir}/evaluations_{timestamp}.json", 'w') as f:
            json.dump(evaluations, f, indent=2)
        
        # Save summary
        with open(f"{results_dir}/summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {results_dir}")
    
    def _save_ablation_results(self, results: Dict):
        """Save ablation study results"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(f"{results_dir}/ablation_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Ablation results saved to {results_dir}")


if __name__ == "__main__":
    logger.info("Evaluation Pipeline initialized")

"""
Generation Module: LLM-based response generation for RAG system
"""

from typing import List, Dict, Tuple
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate responses using open-source LLMs"""
    
    def __init__(self, model_name: str = "distilgpt2", 
                 device: str = 'cpu', task: str = 'text-generation'):
        """
        Initialize response generator
        
        Args:
            model_name: Hugging Face model name
            device: Device to use ('cpu' or 'cuda')
            task: Pipeline task type
        """
        self.model_name = model_name
        self.device = device
        self.task = task
        
        # Load pipeline
        self.pipeline = pipeline(
            task=task,
            model=model_name,
            device=0 if device == 'cuda' else -1
        )
        
        # Load tokenizer for token count
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logger.info(f"ResponseGenerator initialized with {model_name}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        
        Args:
            text: Input text
            
        Returns:
            Token count
        """
        return len(self.tokenizer.encode(text))
    
    def build_prompt(self, query: str, context_chunks: List[Dict],
                    max_context_tokens: int = 512) -> Tuple[str, int]:
        """
        Build prompt from query and context chunks
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            max_context_tokens: Maximum tokens for context
            
        Returns:
            Tuple of (prompt, context_tokens_used)
        """
        context_text = ""
        context_tokens = 0
        logger.info(f"Building prompt with {len(context_chunks)} chunks")
        
        # Add context incrementally
        for i, chunk in enumerate(context_chunks):
            if 'text' not in chunk:
                logger.warning(f"Chunk {i} missing 'text' key")
                continue
            chunk_text = f"[{chunk.get('title', 'Unknown')}] {chunk['text']}\n\n"
            chunk_tokens = self.count_tokens(chunk_text)
            logger.info(f"Chunk {i}: {chunk_tokens} tokens")
            
            if context_tokens + chunk_tokens <= max_context_tokens:
                context_text += chunk_text
                context_tokens += chunk_tokens
                logger.info(f"  Added, total={context_tokens}")
            else:
                logger.info(f"  Skipped (limit)")
                break
        
        logger.info(f"Final context: {context_tokens} tokens")
        # Build prompt
        prompt = f"""Given the following context:
{context_text}

Question: {query}

Answer: """
        
        return prompt, context_tokens
    
    def generate(self, query: str, context_chunks: List[Dict],
                    max_length: int = 256,
                temperature: float = 0.7,
                top_p: float = 0.9) -> Dict:
        """
        Generate response
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Dictionary with response and metadata
        """
        # Build prompt
        prompt, context_tokens = self.build_prompt(query, context_chunks)
        logger.info(f"Prompt length: {len(prompt)} chars, {self.count_tokens(prompt)} tokens")
        
        # Generate
        try:
            outputs = self.pipeline(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1
            )
            
            generated_text = outputs[0]['generated_text']
            logger.info(f"Generated text length: {len(generated_text)} chars")
            logger.info(f"Prompt is first {len(prompt)} chars of generated_text: {generated_text[:len(prompt)] == prompt}")
            
            # Extract answer (remove prompt)
            answer = generated_text[len(prompt):].strip()
            logger.info(f"Extracted answer length: {len(answer)} chars")
            logger.info(f"Answer preview: {answer[:100]}")
            
            result = {
                'answer': answer,
                'full_response': generated_text,
                'context_tokens_used': context_tokens,
                'chunks_used': len(context_chunks),
                'model': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            result = {
                'answer': "Error generating response",
                'error': str(e),
                'context_tokens_used': context_tokens
            }
        
        return result


class LLMAsJudge:
    """Use LLM to evaluate answer quality"""
    
    def __init__(self, model_name: str = "distilgpt2", 
                 device: str = 'cpu'):
        """
        Initialize LLM-as-Judge
        
        Args:
            model_name: Hugging Face model name
            device: Device type
        """
        self.generator = ResponseGenerator(model_name, device)
        self.tokenizer = self.generator.tokenizer
    
    def evaluate_answer(self, question: str, answer: str, 
                       context: str) -> Dict:
        """
        Evaluate answer using LLM
        
        Args:
            question: Original question
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Evaluation results
        """
        prompt = f"""Evaluate the following answer based on the context provided.

Question: {question}

Context: {context}

Answer: {answer}

Evaluate on:
1. Relevance (1-5): How well does it answer the question?
2. Factuality (1-5): Is the answer factually correct based on context?
3. Completeness (1-5): Does it cover all aspects?
4. Coherence (1-5): Is it well-written and clear?

Provide scores and brief explanation."""
        
        try:
            outputs = self.generator.pipeline(
                prompt,
                max_length=150,
                temperature=0.7
            )
            
            evaluation_text = outputs[0]['generated_text']
            
            return {
                'evaluation': evaluation_text,
                'prompt': prompt
            }
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return {'evaluation': 'Error evaluating', 'error': str(e)}


class ConfidenceEstimator:
    """Estimate confidence scores for generated answers"""
    
    @staticmethod
    def estimate_confidence(answer: str, context_relevance: float,
                           context_coverage: float) -> float:
        """
        Estimate answer confidence
        
        Args:
            answer: Generated answer
            context_relevance: Relevance score of context (0-1)
            context_coverage: Coverage score of context (0-1)
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple heuristic: combination of answer length and context quality
        answer_length_score = min(len(answer.split()) / 50, 1.0)
        
        confidence = (0.4 * context_relevance + 
                     0.4 * context_coverage + 
                     0.2 * answer_length_score)
        
        return min(max(confidence, 0), 1)
    
    @staticmethod
    def estimate_hallucination_probability(answer: str, context: str) -> float:
        """
        Rough estimate of hallucination probability
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Hallucination probability [0, 1]
        """
        # Simple approach: check word overlap
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        if len(answer_words) == 0:
            return 0
        
        overlap = len(answer_words & context_words)
        coverage = overlap / len(answer_words)
        
        # High coverage = low hallucination
        hallucination_prob = 1 - coverage
        
        return min(max(hallucination_prob, 0), 1)


if __name__ == "__main__":
    logger.info("Response Generator initialized")

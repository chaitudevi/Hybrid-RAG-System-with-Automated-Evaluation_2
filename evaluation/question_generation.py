"""
Evaluation Module: Question generation and metrics computation
"""

import json
import logging
from typing import List, Dict, Tuple
import random
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Generate Q&A pairs from Wikipedia corpus"""
    
    def __init__(self, chunks: List[Dict]):
        """
        Initialize question generator
        
        Args:
            chunks: Chunks from preprocessing
        """
        self.chunks = chunks
    
    def generate_factual_questions(self, num_questions: int = 20) -> List[Dict]:
        """
        Generate factual questions (who, what, when, where)
        
        Args:
            num_questions: Number to generate
            
        Returns:
            List of Q&A pairs
        """
        qa_pairs = []
        sampled_chunks = random.sample(self.chunks, min(num_questions, len(self.chunks)))
        
        question_templates = [
            "What is {}?",
            "Who is {}?",
            "When did {} happen?",
            "Where is {}?",
            "What does {} mean?",
            "How does {} work?"
        ]
        
        for chunk in sampled_chunks:
            # Extract key noun phrase (simplified)
            words = chunk['text'].split()[:5]
            key_phrase = ' '.join(words)
            
            template = random.choice(question_templates)
            question = template.format(key_phrase)
            
            qa_pair = {
                'question_id': f"Q_{len(qa_pairs):03d}",
                'question': question,
                'question_type': 'factual',
                'ground_truth_answer': chunk['text'][:100],
                'source_url': chunk['url'],
                'source_chunk_id': chunk['chunk_id'],
                'source_title': chunk.get('title', 'Unknown')
            }
            
            qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def generate_comparative_questions(self, num_questions: int = 20) -> List[Dict]:
        """
        Generate comparative questions
        
        Args:
            num_questions: Number to generate
            
        Returns:
            List of Q&A pairs
        """
        qa_pairs = []
        
        if len(self.chunks) < 2:
            return qa_pairs
        
        templates = [
            "What is the difference between {} and {}?",
            "Compare {} and {}",
            "How do {} and {} differ?",
            "What do {} and {} have in common?"
        ]
        
        for i in range(min(num_questions, len(self.chunks) - 1)):
            chunk1 = self.chunks[i]
            chunk2 = self.chunks[i + 1]
            
            phrase1 = ' '.join(chunk1['text'].split()[:3])
            phrase2 = ' '.join(chunk2['text'].split()[:3])
            
            template = random.choice(templates)
            question = template.format(phrase1, phrase2)
            
            qa_pair = {
                'question_id': f"Q_{len(qa_pairs) + 100:03d}",
                'question': question,
                'question_type': 'comparative',
                'ground_truth_answer': f"Compare: {chunk1['text'][:50]} vs {chunk2['text'][:50]}",
                'source_urls': [chunk1['url'], chunk2['url']],
                'source_chunk_ids': [chunk1['chunk_id'], chunk2['chunk_id']],
                'source_titles': [chunk1.get('title'), chunk2.get('title')]
            }
            
            qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def generate_inferential_questions(self, num_questions: int = 20) -> List[Dict]:
        """
        Generate inferential/reasoning questions
        
        Args:
            num_questions: Number to generate
            
        Returns:
            List of Q&A pairs
        """
        qa_pairs = []
        sampled_chunks = random.sample(self.chunks, min(num_questions, len(self.chunks)))
        
        templates = [
            "Based on the context, what can be inferred about {}?",
            "Why might {} be important?",
            "What are the implications of {}?",
            "How does {} relate to {}?"
        ]
        
        for chunk in sampled_chunks:
            key_phrase = ' '.join(chunk['text'].split()[:5])
            
            template = random.choice(templates)
            if "{}" in template:
                if template.count("{}") == 2:
                    question = template.format(key_phrase, key_phrase)
                else:
                    question = template.format(key_phrase)
            else:
                question = template
            
            qa_pair = {
                'question_id': f"Q_{len(qa_pairs) + 200:03d}",
                'question': question,
                'question_type': 'inferential',
                'ground_truth_answer': chunk['text'][:150],
                'source_url': chunk['url'],
                'source_chunk_id': chunk['chunk_id'],
                'source_title': chunk.get('title')
            }
            
            qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def generate_multihop_questions(self, num_questions: int = 20) -> List[Dict]:
        """
        Generate multi-hop questions requiring multiple chunks
        
        Args:
            num_questions: Number to generate
            
        Returns:
            List of Q&A pairs
        """
        qa_pairs = []
        
        if len(self.chunks) < 3:
            return qa_pairs
        
        for i in range(min(num_questions, len(self.chunks) - 2)):
            chunks = random.sample(self.chunks, 3)
            
            phrases = [' '.join(c['text'].split()[:4]) for c in chunks]
            
            templates = [
                f"How does {phrases[0]} relate to {phrases[1]} and {phrases[2]}?",
                f"What is the relationship between {phrases[0]} and {phrases[1]}?",
                f"Given information about {phrases[0]}, what can we say about {phrases[1]}?"
            ]
            
            question = random.choice(templates)
            
            answer = " ".join([c['text'][:80] for c in chunks])
            
            qa_pair = {
                'question_id': f"Q_{len(qa_pairs) + 300:03d}",
                'question': question,
                'question_type': 'multi-hop',
                'ground_truth_answer': answer,
                'source_urls': [c['url'] for c in chunks],
                'source_chunk_ids': [c['chunk_id'] for c in chunks],
                'source_titles': [c.get('title') for c in chunks]
            }
            
            qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def generate_qa_dataset(self, total_questions: int = 100) -> List[Dict]:
        """
        Generate balanced Q&A dataset with multiple question types
        
        Args:
            total_questions: Total questions to generate
            
        Returns:
            List of Q&A pairs
        """
        per_type = total_questions // 4
        
        qa_pairs = []
        qa_pairs.extend(self.generate_factual_questions(per_type))
        qa_pairs.extend(self.generate_comparative_questions(per_type))
        qa_pairs.extend(self.generate_inferential_questions(per_type))
        qa_pairs.extend(self.generate_multihop_questions(per_type))
        
        # Shuffle
        random.shuffle(qa_pairs)
        
        logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
        return qa_pairs[:total_questions]


def save_qa_dataset(qa_pairs: List[Dict], filepath: str):
    """Save Q&A dataset to JSON"""
    with open(filepath, 'w') as f:
        json.dump(qa_pairs, f, indent=2)
    logger.info(f"Saved {len(qa_pairs)} Q&A pairs to {filepath}")


def load_qa_dataset(filepath: str) -> List[Dict]:
    """Load Q&A dataset from JSON"""
    with open(filepath, 'r') as f:
        qa_pairs = json.load(f)
    logger.info(f"Loaded {len(qa_pairs)} Q&A pairs from {filepath}")
    return qa_pairs


if __name__ == "__main__":
    logger.info("Question Generator initialized")

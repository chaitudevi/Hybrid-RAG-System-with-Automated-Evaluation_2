#!/usr/bin/env python
"""Test generation output"""

from src.rag_system import HybridRAGSystem

rag = HybridRAGSystem()
result = rag.answer_query('What is machine learning?')

print("=" * 80)
print("FULL ANSWER:")
print("=" * 80)
print(result['answer'])
print("\n" + "=" * 80)
print(f"Total length: {len(result['answer'])} characters")
print(f"Chunks used: {result['chunks_used']}")
print("=" * 80)


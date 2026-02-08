#!/usr/bin/env python
"""Debug generation issue"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation import ResponseGenerator
from src.rag_system import HybridRAGSystem

# Load system
print("Loading RAG system...")
rag_system = HybridRAGSystem()

try:
    dense_ok = rag_system.dense_retriever.load('data/indices/dense')
    sparse_ok = rag_system.sparse_retriever.load('data/indices/sparse')
    print(f"Dense loaded: {dense_ok}")
    print(f"Sparse loaded: {sparse_ok}")
except Exception as e:
    print(f"Error loading indices: {e}")
    sys.exit(1)

# Test simple retrieval
query = "what is machine learning"
print(f"\nTesting query: {query}")

dense_results = rag_system.dense_retriever.retrieve(query, top_k=3)
print(f"Dense results: {len(dense_results)} chunks")
for i, chunk in enumerate(dense_results[:2]):
    print(f"  Chunk {i+1}: {chunk.get('title', 'Unknown')[:50]}")

sparse_results = rag_system.sparse_retriever.retrieve(query, top_k=3)
print(f"Sparse results: {len(sparse_results)} chunks")

# Fuse results
fused_results = rag_system.fusion.compute_rrf_score(dense_results, sparse_results, alpha=0.5)
print(f"Fused results: {len(fused_results)} chunks")

# Test generation
context_chunks = fused_results[:3]
print(f"\nContext chunks for generation: {len(context_chunks)}")
print("Chunk structure:")
for i, chunk in enumerate(context_chunks[:1]):
    print(f"  Keys: {chunk.keys()}")
    print(f"  Text: {chunk.get('text', 'MISSING')[:100]}")

print("\nTesting generator.generate()...")
gen_result = rag_system.generator.generate(query, context_chunks, max_length=100)
print(f"Generation result:")
print(f"  Keys: {gen_result.keys()}")
print(f"  Answer: {gen_result.get('answer', 'EMPTY')[:100]}")
print(f"  Context tokens: {gen_result.get('context_tokens_used', 0)}")

# Test full answer_query
print(f"\nTesting full answer_query()...")
result = rag_system.answer_query(query)
print(f"Result keys: {result.keys()}")
print(f"Answer: {result.get('answer', 'EMPTY')[:100]}")
print(f"Confidence: {result.get('confidence', 0):.2%}")
print(f"Has error: {'error' in result}")

print("\n✓ Debug complete")

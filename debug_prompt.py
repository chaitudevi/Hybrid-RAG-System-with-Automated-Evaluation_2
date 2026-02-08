#!/usr/bin/env python
"""Debug prompt building"""

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
except Exception as e:
    print(f"Error loading indices: {e}")
    sys.exit(1)

# Test retrieval
query = "what is machine learning"
dense_results = rag_system.dense_retriever.retrieve(query, top_k=3)
sparse_results = rag_system.sparse_retriever.retrieve(query, top_k=3)
fused_results = rag_system.fusion.compute_rrf_score(dense_results, sparse_results, alpha=0.5)
context_chunks = fused_results[:3]

print(f"\n=== CHUNK INSPECTION ===")
print(f"Number of chunks: {len(context_chunks)}")
if context_chunks:
    chunk = context_chunks[0]
    print(f"First chunk keys: {chunk.keys()}")
    print(f"Title: {chunk.get('title', 'N/A')}")
    print(f"Text length: {len(chunk.get('text', ''))}")
    print(f"Text preview: {chunk.get('text', 'EMPTY')[:100]}")

print(f"\n=== PROMPT BUILDING ===")
prompt, context_tokens = rag_system.generator.build_prompt(query, context_chunks)

print(f"Context tokens used: {context_tokens}")
print(f"Prompt length: {len(prompt)}")
print(f"\n--- FULL PROMPT ---")
print(prompt)
print(f"--- END PROMPT ---\n")

# Count tokens in prompt
tokenizer = rag_system.generator.tokenizer
prompt_tokens = len(tokenizer.encode(prompt))
print(f"Prompt token count: {prompt_tokens}")

# Now test generation with the prompt
print(f"\n=== GENERATION ===")
gen_outputs = rag_system.generator.pipeline(
    prompt,
    max_length=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=1
)

print(f"Generated text:\n{gen_outputs[0]['generated_text']}")

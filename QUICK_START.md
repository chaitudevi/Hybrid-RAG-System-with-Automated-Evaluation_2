# QUICK START GUIDE - Hybrid RAG System

## 🚀 Installation (2 minutes)

```bash
# 1. Navigate to project
cd hybrid_rag_system

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize project
python setup.py
```

## 🎯 Quick Commands

### Option 1: Web Interface (Easiest)
```bash
streamlit run ui/app.py
```
Then open http://localhost:8501 in your browser.

### Option 2: Demo Notebook
```bash
jupyter notebook notebooks/demo.ipynb
```
Run cells sequentially to see the complete workflow.

### Option 3: Command Line Pipeline
```bash
# Build system
python main.py --mode build --num-urls 5

# Evaluate
python main.py --mode evaluate --num-questions 20

# Query
python main.py --mode query --query "What is machine learning?"

# Full pipeline
python main.py --mode full --num-urls 10 --num-questions 50
```

## 📝 Sample Queries

```python
from src.rag_system import HybridRAGSystem

# Initialize
rag = HybridRAGSystem()

# Would need to build indices first (see above)

# Query
result = rag.answer_query("What is artificial intelligence?")
print(result['answer'])
print(f"Confidence: {result['confidence']:.2%}")
```

## 📊 Understanding the Output

### Query Result Structure
```python
result = {
    'query': "Your question",
    'answer': "Generated answer",
    'confidence': 0.75,  # 0-1 scale
    'retrieval': {
        'dense': [...chunks from dense search...],
        'sparse': [...chunks from sparse search...],
        'fused': [...final RRF-fused chunks...]
    },
    'timings': {
        'total': 2.34,  # seconds
        'context_tokens': 256
    }
}
```

### Evaluation Metrics
- **MRR (URL-level)**: How quickly system finds correct source (0-1)
- **Precision@10**: % of top-10 results that are relevant
- **Recall@10**: % of relevant docs found in top-10
- **NDCG@10**: Ranking quality score
- **Semantic Similarity**: Answer quality (0-1)

## 🔍 File Structure Overview

```
├── src/                  # Core RAG implementation
├── evaluation/          # Evaluation & metrics
├── ui/app.py           # Web interface
├── notebooks/demo.ipynb # Jupyter demo
├── fixed_urls.json     # 200 Wikipedia URLs
├── README.md           # Full documentation
├── main.py             # Main pipeline
└── requirements.txt    # Dependencies
```

## ⚙️ Configuration

Edit config in code:
```python
config = {
    'chunk_size': 300,           # Tokens per chunk
    'dense_model': 'all-MiniLM-L6-v2',  # Embedding model
    'llm_model': 'distilgpt2',   # LLM for generation
    'device': 'cpu',             # or 'cuda' for GPU
    'final_top_n': 5             # Chunks for context
}

rag = HybridRAGSystem(config=config)
```

Or adjust via Streamlit UI → System Config page.

## 🐛 Troubleshooting

### "Module not found" error
```bash
pip install --upgrade -r requirements.txt
python setup.py
```

### NLTK data missing
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Slow performance
- Use GPU: Change `device='cuda'`
- Reduce chunk size
- Use smaller model: `distilgpt2` instead of larger models

### Out of memory
- Reduce `chunk_size` to 200
- Reduce batch size
- Use CPU mode

## 📈 Next Steps

1. **Explore Notebook**: Run `notebooks/demo.ipynb` for complete walkthrough
2. **Web Interface**: Try `streamlit run ui/app.py` for interactive experience
3. **Full Evaluation**: Run `python main.py --mode full` for complete pipeline
4. **Read Docs**: Check `README.md` for detailed documentation

## 🎓 Learning Path

1. Start with demo notebook to understand flow
2. Explore web interface to see components
3. Try different queries to understand behavior
4. Run evaluation to see metrics
5. Modify configuration and re-evaluate
6. Check error analysis to understand failures

## ✅ Verification Checklist

After installation, verify:
- [ ] Dependencies installed: `pip list | grep -E "transformers|faiss|streamlit"`
- [ ] NLTK data: Run any Python file (should download on first use)
- [ ] Model access: First query will download models (~500MB total)
- [ ] Web app runs: `streamlit run ui/app.py` opens browser

## 📚 Key Resources

- **Dense Retrieval**: FAISS (Facebook AI Similarity Search)
- **Sparse Retrieval**: BM25 (Okapi)
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **LLM**: DistilGPT2 or Flan-T5
- **Web UI**: Streamlit

## 🔗 Important Files

| File | Purpose |
|------|---------|
| `main.py` | Command-line entry point |
| `ui/app.py` | Web interface entry point |
| `notebooks/demo.ipynb` | Interactive demonstration |
| `fixed_urls.json` | 200 Wikipedia URLs |
| `README.md` | Full documentation |
| `ASSIGNMENT_CHECKLIST.md` | Assignment compliance |

## 💡 Tips & Tricks

1. **Use demo notebook first** - Easiest way to understand system
2. **Start with small dataset** - Use `--num-urls 5` for quick testing
3. **Monitor console output** - Shows detailed progress
4. **Save results** - All evaluations auto-saved to `results/` directory
5. **Modify queries** - Try different question types to see behavior

---

**Ready to start?** → Run: `streamlit run ui/app.py`

For detailed info → Read: `README.md`

For assignment details → Check: `ASSIGNMENT_CHECKLIST.md`

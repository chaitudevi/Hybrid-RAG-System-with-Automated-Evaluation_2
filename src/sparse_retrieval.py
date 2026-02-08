"""
Sparse Retrieval Module: BM25 keyword-based retrieval
"""

from typing import List, Dict
import logging
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)


class SparseRetriever:
    """Sparse keyword-based retrieval using BM25"""
    
    def __init__(self, remove_stopwords: bool = True, 
                 lowercase: bool = True):
        """
        Initialize sparse retriever
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lowercase: Whether to lowercase tokens
        """
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.bm25 = None
        self.chunks = []
        self.tokenized_chunks = []
        self.stopwords = set(stopwords.words('english')) if remove_stopwords else set()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize and clean text
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.lowercase:
            text = text.lower()
        
        tokens = word_tokenize(text)
        
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords and t.isalnum()]
        
        return tokens
    
    def build_index(self, chunks: List[Dict], 
                   save_path: str = None) -> BM25Okapi:
        """
        Build BM25 index from chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            save_path: Optional path to save index
            
        Returns:
            BM25Okapi index
        """
        self.chunks = chunks
        
        logger.info(f"Building BM25 index for {len(chunks)} chunks...")
        
        # Tokenize all chunks
        self.tokenized_chunks = [
            self.tokenize(chunk['text']) 
            for chunk in chunks
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        
        logger.info("BM25 index built successfully")
        
        if save_path:
            self.save(save_path)
        
        return self.bm25
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve top-k chunks using BM25
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks with scores
        """
        if self.bm25 is None:
            raise ValueError("Index not built yet. Call build_index first.")
        
        # Tokenize query
        query_tokens = self.tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Sort by score and get top-k
        ranked_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:top_k]
        
        results = []
        for rank, idx in enumerate(ranked_indices):
            chunk = self.chunks[idx].copy()
            chunk['sparse_score'] = float(scores[idx])
            chunk['rank'] = rank + 1
            results.append(chunk)
        
        logger.info(f"Retrieved {len(results)} chunks using BM25")
        return results
    
    def save(self, save_path: str):
        """
        Save BM25 index and chunks
        
        Args:
            save_path: Directory to save files
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Save BM25 index
        with open(f"{save_path}/bm25.pkl", 'wb') as f:
            pickle.dump(self.bm25, f)
        
        # Save chunks metadata
        with open(f"{save_path}/chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Save tokenized chunks
        with open(f"{save_path}/tokenized_chunks.pkl", 'wb') as f:
            pickle.dump(self.tokenized_chunks, f)
        
        logger.info(f"Sparse (BM25) index saved to {save_path}")
    
    def load(self, load_path: str):
        """
        Load BM25 index and chunks
        
        Args:
            load_path: Directory containing saved files
        """
        bm25_file = f"{load_path}/bm25.pkl"
        chunks_file = f"{load_path}/chunks.pkl"
        
        if not Path(bm25_file).exists() or not Path(chunks_file).exists():
            logger.warning(f"BM25 index files not found at {load_path}. Skipping load.")
            return False
        
        try:
            # Load BM25 index
            with open(bm25_file, 'rb') as f:
                self.bm25 = pickle.load(f)
            
            # Load chunks
            with open(chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
            
            logger.info(f"Sparse index loaded from {load_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load sparse index: {e}")
            return False


def create_inverted_index(chunks: List[Dict]) -> Dict[str, List[int]]:
    """
    Create inverted index for keyword search
    
    Args:
        chunks: List of chunks
        
    Returns:
        Dictionary mapping tokens to chunk indices
    """
    inverted_index = {}
    
    for idx, chunk in enumerate(chunks):
        tokens = set(word_tokenize(chunk['text'].lower()))
        
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append(idx)
    
    return inverted_index


if __name__ == "__main__":
    logger.info("Sparse Retriever (BM25) initialized")

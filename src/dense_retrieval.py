"""
Dense Retrieval Module: Sentence embeddings with FAISS vector index
"""

import numpy as np
from typing import List, Dict, Tuple
import logging
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense vector retrieval using sentence embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 device: str = 'cpu'):
        """
        Initialize dense retriever with sentence transformer
        
        Args:
            model_name: HuggingFace model name for embeddings
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.chunks = []
        self.embeddings = None
        logger.info(f"DenseRetriever initialized with {model_name}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed list of texts using sentence transformer
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def build_index(self, chunks: List[Dict], 
                   save_path: str = None) -> faiss.Index:
        """
        Build FAISS index from chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            save_path: Optional path to save index and chunks
            
        Returns:
            FAISS index
        """
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.embed_texts(texts)
        self.embeddings = embeddings
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index built: {len(texts)} chunks, {dimension} dimensions")
        
        if save_path:
            self.save(save_path)
        
        return self.index
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve top-k chunks for a query
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks with scores
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index first.")
        
        # Embed query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            min(top_k, len(self.chunks))
        )
        
        # Convert distances to similarity scores (cosine similarity)
        # FAISS returns L2 distances, convert to similarity
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.chunks[idx].copy()
            # L2 distance to cosine similarity approximation
            similarity = 1 / (1 + distance)
            chunk['dense_score'] = float(similarity)
            chunk['rank'] = len(results) + 1
            results.append(chunk)
        
        logger.info(f"Retrieved {len(results)} chunks for query")
        return results
    
    def save(self, save_path: str):
        """
        Save index and chunks to disk
        
        Args:
            save_path: Directory to save files
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{save_path}/dense.index")
        
        # Save chunks metadata
        with open(f"{save_path}/chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Save model info
        with open(f"{save_path}/model_info.txt", 'w') as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Total chunks: {len(self.chunks)}\n")
        
        logger.info(f"Dense index saved to {save_path}")
    
    def load(self, load_path: str):
        """
        Load index and chunks from disk
        
        Args:
            load_path: Directory containing saved files
        """
        index_file = f"{load_path}/dense.index"
        chunks_file = f"{load_path}/chunks.pkl"
        
        if not Path(index_file).exists() or not Path(chunks_file).exists():
            logger.warning(f"Index files not found at {load_path}. Skipping load.")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_file)
            
            # Load chunks
            with open(chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
            
            logger.info(f"Dense index loaded from {load_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load dense index: {e}")
            return False


class HybridDenseRetriever:
    """Extended dense retriever with multiple models for comparison"""
    
    def __init__(self, models: List[str] = None):
        """
        Initialize with multiple models
        
        Args:
            models: List of model names
        """
        self.models = models or ["all-MiniLM-L6-v2"]
        self.retrievers = {}
        
        for model_name in self.models:
            self.retrievers[model_name] = DenseRetriever(model_name)
    
    def build_all_indices(self, chunks: List[Dict], save_path: str = None):
        """
        Build indices for all models
        
        Args:
            chunks: List of chunks
            save_path: Path to save indices
        """
        for model_name, retriever in self.retrievers.items():
            path = f"{save_path}/{model_name}" if save_path else None
            retriever.build_index(chunks, path)
    
    def retrieve_ensemble(self, query: str, top_k: int = 10) -> Dict[str, List]:
        """
        Retrieve using all models
        
        Args:
            query: Query string
            top_k: Number of results per model
            
        Returns:
            Dictionary with results from each model
        """
        results = {}
        for model_name, retriever in self.retrievers.items():
            results[model_name] = retriever.retrieve(query, top_k)
        
        return results


if __name__ == "__main__":
    logger.info("Dense Retriever initialized")

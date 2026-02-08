"""
Preprocessing Module: Text cleaning and chunking for RAG system
"""

import re
import string
from typing import List, Dict, Tuple
import logging
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Clean and preprocess text for chunking"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing special characters, extra whitespace, etc.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\-\:\;]', '', text)
        
        return text.strip()
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Approximate token count (simple word-based approximation)
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple approximation: ~1.3 tokens per word on average
        words = len(text.split())
        return int(words * 1.3)
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences using NLTK
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback to simple split
            sentences = re.split(r'[.!?]+', text)
        
        return [s.strip() for s in sentences if s.strip()]


class ChunkProcessor:
    """Chunk text with sliding window approach"""
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50, 
                 tokenizer_type: str = 'word'):
        """
        Initialize chunk processor
        
        Args:
            chunk_size: Target tokens per chunk
            overlap: Overlapping tokens between chunks
            tokenizer_type: 'word' or 'sentence'
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer_type = tokenizer_type
        self.preprocessor = TextPreprocessor()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into units
        
        Args:
            text: Input text
            
        Returns:
            List of tokens/units
        """
        if self.tokenizer_type == 'sentence':
            return self.preprocessor.split_into_sentences(text)
        else:  # word-based
            return text.split()
    
    def chunk_text(self, text: str, doc_id: str = None, 
                   url: str = None) -> List[Dict]:
        """
        Chunk text with sliding window and overlap
        
        Args:
            text: Input text to chunk
            doc_id: Document identifier
            url: Source URL
            
        Returns:
            List of chunks with metadata
        """
        # Clean text
        text = self.preprocessor.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        chunks = []
        chunk_id = 0
        
        # Create chunks with overlap
        i = 0
        while i < len(tokens):
            # Get chunk
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_tokens)
            
            # Validate chunk size
            if self.preprocessor.count_tokens(chunk_text) < 50:
                i += 1
                continue
            
            chunk_dict = {
                'chunk_id': f"{doc_id}_chunk_{chunk_id}" if doc_id else f"chunk_{chunk_id}",
                'text': chunk_text,
                'token_count': self.preprocessor.count_tokens(chunk_text),
                'url': url,
                'sequence': chunk_id
            }
            
            chunks.append(chunk_dict)
            chunk_id += 1
            
            # Move by overlap
            i += self.chunk_size - self.overlap
        
        return chunks
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Process multiple documents and return all chunks
        
        Args:
            documents: List of documents with 'text', 'url', 'title' keys
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc.get('text', '')
            url = doc.get('url', '')
            title = doc.get('title', '')
            
            chunks = self.chunk_text(
                text=text,
                doc_id=f"doc_{doc_idx}",
                url=url
            )
            
            # Add metadata
            for chunk in chunks:
                chunk['title'] = title
            
            all_chunks.extend(chunks)
            
            logger.info(f"Processed {title}: {len(chunks)} chunks")
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


def prepare_rag_corpus(documents: List[Dict], chunk_size: int = 300, 
                       overlap: int = 50) -> List[Dict]:
    """
    Main function to prepare documents for RAG system
    
    Args:
        documents: Raw documents
        chunk_size: Chunk token size
        overlap: Overlap tokens
        
    Returns:
        Processed chunks ready for indexing
    """
    processor = ChunkProcessor(chunk_size=chunk_size, overlap=overlap)
    chunks = processor.process_documents(documents)
    
    return chunks


if __name__ == "__main__":
    logger.info("Text Preprocessor initialized")

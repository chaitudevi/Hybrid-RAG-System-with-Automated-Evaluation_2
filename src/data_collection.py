"""
Data Collection Module: Fetch and extract text from Wikipedia articles
"""

import json
import random
import re
from typing import List, Dict, Tuple
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WikipediaCollector:
    """Collect and extract text from Wikipedia URLs"""
    
    def __init__(self, min_words: int = 200, delay: float = 1.0):
        """
        Initialize Wikipedia collector
        
        Args:
            min_words: Minimum words required per page
            delay: Delay between requests (seconds)
        """
        self.min_words = min_words
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_text_from_url(self, url: str) -> Tuple[str, str, int]:
        """
        Extract main text content from Wikipedia URL
        
        Args:
            url: Wikipedia article URL
            
        Returns:
            Tuple of (text, title, word_count)
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1', class_='firstHeading')
            title = title_elem.get_text() if title_elem else 'Unknown'
            
            # Extract main content
            content_div = soup.find('div', id='mw-content-text')
            if not content_div:
                return "", title, 0
            
            # Remove unwanted elements
            for element in content_div.find_all(['script', 'style', 'sup', '[document]']):
                element.decompose()
            
            # Extract paragraphs
            paragraphs = []
            for p in content_div.find_all('p'):
                text = p.get_text().strip()
                if text:
                    paragraphs.append(text)
            
            text = ' '.join(paragraphs)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            word_count = len(text.split())
            
            return text, title, word_count
            
        except Exception as e:
            logger.warning(f"Failed to extract from {url}: {e}")
            return "", "", 0
    
    def collect_from_urls(self, urls: List[str]) -> List[Dict]:
        """
        Collect data from list of URLs with word count validation
        
        Args:
            urls: List of Wikipedia URLs
            
        Returns:
            List of documents with text, title, URL, and word count
        """
        documents = []
        valid_urls = []
        
        for i, url in enumerate(urls):
            logger.info(f"Processing {i+1}/{len(urls)}: {url}")
            
            text, title, word_count = self.extract_text_from_url(url)
            
            if word_count >= self.min_words:
                doc = {
                    'url': url,
                    'title': title,
                    'text': text,
                    'word_count': word_count
                }
                documents.append(doc)
                valid_urls.append(url)
                logger.info(f"✓ Extracted ({word_count} words)")
            else:
                logger.warning(f"✗ Insufficient words ({word_count} < {self.min_words})")
            
            time.sleep(self.delay)
        
        return documents, valid_urls
    
    @staticmethod
    def load_fixed_urls(json_path: str) -> List[str]:
        """Load fixed URLs from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
            return data.get('fixed_urls', [])
    
    @staticmethod
    def save_fixed_urls(urls: List[str], json_path: str):
        """Save fixed URLs to JSON file"""
        data = {'fixed_urls': urls}
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def sample_random_urls(all_urls: List[str], fixed_urls: List[str], 
                          count: int = 300) -> List[str]:
        """
        Sample random URLs excluding fixed URLs
        
        Args:
            all_urls: Pool of all URLs
            fixed_urls: URLs to exclude
            count: Number of URLs to sample
            
        Returns:
            List of sampled URLs
        """
        available_urls = [u for u in all_urls if u not in fixed_urls]
        return random.sample(available_urls, min(count, len(available_urls)))

    def fetch_random_urls(self, count: int = 1, fixed_urls: List[str] = None) -> List[str]:
        """
        Fetch random Wikipedia URLs using Special:Random
        
        Args:
            count: Number of random URLs to fetch
            fixed_urls: Optional list of URLs to exclude
            
        Returns:
            List of unique random URLs
        """
        random_urls = set()
        exclude_urls = set(fixed_urls) if fixed_urls else set()
        
        logger.info(f"Fetching {count} random URLs...")
        
        max_attempts = count * 3
        attempts = 0
        
        while len(random_urls) < count and attempts < max_attempts:
            try:
                response = self.session.get("https://en.wikipedia.org/wiki/Special:Random", allow_redirects=True)
                url = response.url
                
                # Check if valid article (not a special page or category)
                if "Category:" in url or "Special:" in url or "File:" in url or "Help:" in url:
                    continue
                    
                if url not in exclude_urls and url not in random_urls:
                    random_urls.add(url)
                    logger.info(f"  + Found random URL: {url}")
                
                time.sleep(0.5)  # Be polite
                
            except Exception as e:
                logger.warning(f"Error fetching random URL: {e}")
            
            attempts += 1
            
        return list(random_urls)


def generate_sample_urls() -> Tuple[List[str], List[str]]:
    """
    Generate sample Wikipedia URLs for demonstration
    Returns: (fixed_urls, sample_random_urls)
    """
    # Example fixed URLs (200 URLs covering diverse topics)
    fixed_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://en.wikipedia.org/wiki/Deep_learning",
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://en.wikipedia.org/wiki/Computer_vision",
        "https://en.wikipedia.org/wiki/Quantum_computing",
        "https://en.wikipedia.org/wiki/Data_science",
        "https://en.wikipedia.org/wiki/Neural_network",
        "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)",
        # Add more URLs as needed to reach 200
    ]
    
    # For demonstration, we'll use a smaller set
    # In actual implementation, collect 200 diverse URLs
    sample_random = [
        "https://en.wikipedia.org/wiki/History_of_artificial_intelligence",
        "https://en.wikipedia.org/wiki/Knowledge_representation",
    ]
    
    return fixed_urls, sample_random


if __name__ == "__main__":
    # Example usage
    logger.info("Wikipedia Data Collector initialized")
    # streamlit run ui/app.py

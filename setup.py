"""
Main setup and initialization file
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def setup_directories():
    """Create necessary directories"""
    directories = [
        'data/corpus',
        'data/indices',
        'data/qa',
        'results',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory: {directory}")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        'transformers': 'transformers',
        'sentence_transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu',
        'rank_bm25': 'rank-bm25',
        'nltk': 'nltk',
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'streamlit': 'streamlit',
        'plotly': 'plotly'
    }
    
    missing = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        logger.warning(f"Missing packages: {', '.join(missing)}")
        logger.warning("Install with: pip install " + " ".join(missing))
    else:
        logger.info("All required packages are installed")

if __name__ == "__main__":
    setup_directories()
    check_dependencies()
    logger.info("Setup complete!")

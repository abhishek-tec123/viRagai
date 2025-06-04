from sentence_transformers import SentenceTransformer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Global variable to store the model
embedding_model = None

def initialize_model():
    global embedding_model
    log.info("Loading sentence transformer model...")
    try:
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        log.info("Sentence transformer model loaded successfully")
        return embedding_model
    except Exception as e:
        log.error(f"Error loading sentence transformer model: {str(e)}")
        raise e

def get_model():
    global embedding_model
    if embedding_model is None:
        return initialize_model()
    return embedding_model 
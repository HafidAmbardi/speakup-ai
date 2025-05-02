import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Qdrant configuration
QDRANT_HOST = os.getenv('QDRANT_HOST')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

def test_qdrant_connection():
    try:
        # Initialize client
        client = QdrantClient(
            host=QDRANT_HOST,
            api_key=QDRANT_API_KEY,
        )
        
        # Test connection by getting collections
        collections = client.get_collections()
        logger.info("Successfully connected to Qdrant!")
        logger.info(f"Available collections: {[col.name for col in collections.collections]}")
        
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {str(e)}")
        raise

if __name__ == '__main__':
    test_qdrant_connection() 
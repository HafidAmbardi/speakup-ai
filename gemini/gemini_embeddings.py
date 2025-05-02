import json
from typing import List, Dict, Any
import os
import google.generativeai as genai
import time
from dotenv import load_dotenv
import logging
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG level
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
logger.debug("Loading environment variables...")
load_dotenv()
logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f"Environment variables loaded: {os.environ.get('QDRANT_HOST') is not None}")

def initialize_qdrant_client(host: str, api_key: str) -> QdrantClient:
    """Initialize Qdrant client with proper configuration."""
    try:
        client = QdrantClient(
            host=host,
            api_key=api_key,
        )
        logger.info("Qdrant client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {str(e)}")
        raise

def create_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 768  # Default for text-embedding-004
) -> None:
    """Create a new Qdrant collection with proper configuration."""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name in collection_names:
            logger.info(f"Collection {collection_name} already exists")
            return
        
        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        
        logger.info(f"Created Qdrant collection: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to create Qdrant collection: {str(e)}")
        raise

def load_materials(json_path: str) -> List[Dict[str, Any]]:
    """Load and process materials from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_materials = []
        
        for item in data:
            # Extract required fields
            section_id = item.get('_sectionId', '')
            section_title = item.get('_sectionTitle', '')
            title = item.get('title', '')
            plain_text_content = item.get('plainTextContent', '')
            sequence = item.get('sequence', 0)
            
            # Combine section title, title, and content for embedding
            combined_text = f"{section_title} {title} {plain_text_content}"
            
            processed_materials.append({
                'id': item.get('_id', ''),
                'title': title,
                'text': combined_text,
                'section_id': section_id,
                'section_title': section_title,
                'sequence': sequence,
                'plainTextContent': plain_text_content
            })
        
        logger.info(f"Successfully processed {len(processed_materials)} materials")
        return processed_materials
    except Exception as e:
        logger.error(f"Error loading materials: {str(e)}")
        raise

def reset_collection(client: QdrantClient, collection_name: str) -> None:
    """Reset the collection by deleting and recreating it."""
    try:
        # Delete existing collection if it exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name in collection_names:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        
        # Create new collection
        create_qdrant_collection(client, collection_name)
        logger.info(f"Created new collection: {collection_name}")
    except Exception as e:
        logger.error(f"Error resetting collection: {str(e)}")
        raise

def create_and_upload_embeddings(
    materials: List[Dict[str, Any]], 
    client: QdrantClient,
    collection_name: str
) -> None:
    """Create embeddings and store them in Qdrant."""
    try:
        # Initialize Generative AI
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        # Process in batches to avoid memory issues
        batch_size = 100
        points = []
        
        for i in range(0, len(materials), batch_size):
            batch = materials[i:i + batch_size]
            batch_points = []
            
            for idx, material in enumerate(batch):
                try:
                    # Create embedding for the text
                    response = genai.embed_content(
                        model="text-embedding-004",
                        content=material['text'],
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                    
                    # Generate a unique ID if material_id is empty or invalid
                    point_id = material.get('id')
                    if not point_id:
                        point_id = f"{i + idx}"  # Use batch index + local index as ID
                    
                    point = models.PointStruct(
                        id=point_id,
                        vector=response['embedding'],
                        payload={
                            "_id": material['id'],
                            "title": material['title'],
                            "_sectionTitle": material['section_title'],
                            "_sectionId": material['section_id'],
                            "sequence": material['sequence'],
                            "plainTextContent": material['plainTextContent']
                        }
                    )
                    batch_points.append(point)
                    
                    logger.info(f"Generated embedding for material: {material['title']}")
                except Exception as e:
                    logger.error(f"Error processing material {material['title']}: {str(e)}")
                    continue
            
            if batch_points:
                # Upload batch to Qdrant
                client.upsert(
                    collection_name=collection_name,
                    points=batch_points
                )
                
                points.extend(batch_points)
                logger.info(f"Uploaded batch {i//batch_size + 1} of embeddings")
            
            # Add a small delay between batches
            time.sleep(1)
        
        logger.info(f"Successfully created and stored {len(points)} embeddings")
            
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise

def main():
    # Configuration
    QDRANT_HOST = os.getenv('QDRANT_HOST')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', "educational-materials")
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    JSON_PATH = 'query-result.json'
    
    # Debug logging for environment variables
    logger.debug(f"QDRANT_HOST: {'Set' if QDRANT_HOST else 'Not set'}")
    logger.debug(f"QDRANT_API_KEY: {'Set' if QDRANT_API_KEY else 'Not set'}")
    logger.debug(f"GOOGLE_API_KEY: {'Set' if GOOGLE_API_KEY else 'Not set'}")
    
    # Validate environment variables
    if not all([QDRANT_HOST, QDRANT_API_KEY, GOOGLE_API_KEY]):
        logger.error("Missing required environment variables")
        logger.info("Please create a .env file with the following variables:")
        logger.info("QDRANT_HOST=your-qdrant-host")
        logger.info("QDRANT_API_KEY=your-qdrant-api-key")
        logger.info("COLLECTION_NAME=your-collection-name (optional)")
        logger.info("GOOGLE_API_KEY=your-google-api-key")
        return
    
    try:
        # Initialize Qdrant client
        client = initialize_qdrant_client(QDRANT_HOST, QDRANT_API_KEY)
        
        # Reset collection
        logger.info("Resetting collection...")
        reset_collection(client, COLLECTION_NAME)
        
        # Load and process materials
        logger.info("Loading materials...")
        materials = load_materials(JSON_PATH)
        logger.info(f"Processing {len(materials)} materials")
        
        # Create and upload embeddings
        logger.info("Creating and uploading embeddings...")
        create_and_upload_embeddings(materials, client, COLLECTION_NAME)
        
        logger.info("Process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")

if __name__ == '__main__':
    main() 
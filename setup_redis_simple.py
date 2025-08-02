#!/usr/bin/env python3
"""
Simple Redis Index Setup Script for Semantic Cache
This version uses direct Redis commands instead of search field imports.
"""

import os
import redis
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_redis_index():
    """Create Redis Search index using direct commands"""
    try:
        # Connect to Redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(redis_url)
        
        # Test connection
        redis_client.ping()
        logger.info("✓ Connected to Redis successfully")
        
        # Drop existing index if it exists
        try:
            redis_client.execute_command("FT.DROPINDEX", "llm_cache_idx", "DD")
            logger.info("⚠ Dropped existing index 'llm_cache_idx'")
        except:
            logger.info("Index 'llm_cache_idx' doesn't exist - creating new one")
        
        # Create index using direct FT.CREATE command
        # Amazon Titan embeddings are 1536 dimensions (same as OpenAI)
        create_command = [
            "FT.CREATE",
            "llm_cache_idx",
            "ON", "JSON",
            "PREFIX", "1", "cache:",
            "SCHEMA",
            "$.query", "AS", "query", "TEXT",
            "$.embedding", "AS", "embedding", 
            "VECTOR", "HNSW", "6", 
            "TYPE", "FLOAT32", 
            "DIM", "1536", 
            "DISTANCE_METRIC", "COSINE",
            "$.response", "AS", "response", "TEXT",
            "$.timestamp", "AS", "timestamp", "TEXT"
        ]
        
        redis_client.execute_command(*create_command)
        
        logger.info("✓ Successfully created Redis Search index 'llm_cache_idx'")
        logger.info("Index configuration:")
        logger.info("  - Type: JSON documents")
        logger.info("  - Prefix: cache:")
        logger.info("  - Vector field: embedding (1536 dimensions, COSINE distance)")
        logger.info("  - HNSW vector search enabled")
        
        # Verify index creation
        try:
            info = redis_client.execute_command("FT.INFO", "llm_cache_idx")
            logger.info(f"✓ Index verification successful")
        except Exception as e:
            logger.warning(f"Could not verify index: {e}")
        
        return True
        
    except redis.ConnectionError as e:
        logger.error(f"✗ Redis connection failed: {e}")
        logger.error("Make sure Redis Stack is running with RediSearch module")
        return False
        
    except Exception as e:
        logger.error(f"✗ Index creation failed: {e}")
        logger.error(f"Error details: {str(e)}")
        return False


def verify_redis_setup():
    """Verify that Redis is properly configured"""
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(redis_url)
        
        # Test basic connection
        redis_client.ping()
        
        # Check RedisJSON
        try:
            redis_client.execute_command("JSON.SET", "test_json", "$", '{"test": "value"}')
            redis_client.execute_command("JSON.DEL", "test_json")
            logger.info("✓ RedisJSON module is available")
        except Exception as e:
            logger.error(f"✗ RedisJSON module not available: {e}")
            return False
        
        # Check RediSearch
        try:
            redis_client.execute_command("FT.INFO", "llm_cache_idx")
            logger.info("✓ RediSearch index exists and is accessible")
        except Exception as e:
            logger.error(f"✗ RediSearch index not found: {e}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Redis setup verification failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Setting up Redis Search index for Semantic Cache...")
    print()
    
    # Check environment
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    print(f"Redis URL: {redis_url}")
    print()
    
    # Setup index
    if setup_redis_index():
        print()
        print("✅ Redis index setup completed successfully!")
        print()
        print("Next steps:")
        print("1. Your OpenAI API key is already set")
        print("2. Start the FastAPI app: uvicorn main:app --reload")
        print("3. Test with: curl -X POST localhost:8000/ask -d '{\"query\":\"Hello\",\"api_key\":\"your-key\"}'")
        
        # Verify setup
        print()
        print("🔍 Verifying Redis setup...")
        if verify_redis_setup():
            print("✅ All Redis components verified and working!")
        else:
            print("❌ Redis setup verification failed")
            exit(1)
    else:
        print()
        print("❌ Redis index setup failed!")
        print()
        print("Troubleshooting:")
        print("1. Make sure Redis Stack is running (not regular Redis)")
        print("2. Check if Redis Stack container is running: docker ps")
        print("3. Try restarting Redis: docker restart $(docker ps -q --filter ancestor=redis/redis-stack)")
        exit(1)
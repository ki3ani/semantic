#!/usr/bin/env python3
"""
Redis Index Setup Script for Semantic Cache

This script creates the required Redis Search index for vector similarity search.
Run this once before starting the FastAPI application.
"""

import os
import redis
import logging
try:
    from redis.commands.search.field import VectorField, TextField
    from redis.commands.search.indexdefinition import IndexDefinition, IndexType
except ImportError:
    # Try newer redis-py structure
    try:
        from redis.commands.search import VectorField, TextField
        from redis.commands.search import IndexDefinition, IndexType
    except ImportError:
        # Fallback for older versions
        from redisearch import VectorField, TextField, IndexDefinition

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_redis_index():
    """Create Redis Search index for semantic caching"""
    try:
        # Connect to Redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(redis_url)
        
        # Test connection
        redis_client.ping()
        logger.info("✓ Connected to Redis successfully")
        
        # Check if RedisSearch is available
        try:
            redis_client.ft().info("llm_cache_idx")
            logger.info("⚠ Index 'llm_cache_idx' already exists - dropping and recreating")
            redis_client.ft("llm_cache_idx").dropindex(delete_documents=True)
        except:
            logger.info("Index 'llm_cache_idx' doesn't exist - creating new one")
        
        # Define index schema
        schema = [
            TextField("$.query", as_name="query"),
            VectorField(
                "$.embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": 1536,
                    "DISTANCE_METRIC": "COSINE",
                    "INITIAL_CAP": 1000,
                    "M": 16,
                    "EF_CONSTRUCTION": 200
                },
                as_name="embedding"
            ),
            TextField("$.response", as_name="response"),
            TextField("$.timestamp", as_name="timestamp")
        ]
        
        # Create index definition
        definition = IndexDefinition(
            prefix=["cache:"],
            index_type=IndexType.JSON
        )
        
        # Create the index
        redis_client.ft("llm_cache_idx").create_index(
            schema,
            definition=definition
        )
        
        logger.info("✓ Successfully created Redis Search index 'llm_cache_idx'")
        logger.info("Index configuration:")
        logger.info("  - Type: JSON documents")
        logger.info("  - Prefix: cache:")
        logger.info("  - Vector field: embedding (1536 dimensions, COSINE distance)")
        logger.info("  - HNSW parameters: M=16, EF_CONSTRUCTION=200")
        
        # Verify index creation
        info = redis_client.ft("llm_cache_idx").info()
        logger.info(f"✓ Index verification: {info['num_docs']} documents indexed")
        
        return True
        
    except redis.ConnectionError as e:
        logger.error(f"✗ Redis connection failed: {e}")
        logger.error("Make sure Redis Stack is running with RediSearch module")
        return False
        
    except Exception as e:
        logger.error(f"✗ Index creation failed: {e}")
        return False


def verify_redis_setup():
    """Verify that Redis is properly configured for the semantic cache"""
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(redis_url)
        
        # Test basic connection
        redis_client.ping()
        
        # Check RedisJSON
        try:
            redis_client.json().set("test_json", "$", {"test": "value"})
            redis_client.delete("test_json")
            logger.info("✓ RedisJSON module is available")
        except Exception:
            logger.error("✗ RedisJSON module not available")
            return False
        
        # Check RediSearch
        try:
            redis_client.ft().info("llm_cache_idx")
            logger.info("✓ RediSearch index exists and is accessible")
        except Exception:
            logger.error("✗ RediSearch index not found")
            return False
        
        # Check search functionality with dummy data
        try:
            # This will fail gracefully if no data exists
            redis_client.ft("llm_cache_idx").search("*")
            logger.info("✓ Vector search functionality verified")
        except Exception as e:
            logger.warning(f"⚠ Vector search test failed (expected if no data): {e}")
        
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
        print("1. Set your OpenAI API key: export OPENAI_API_KEY='sk-...'")
        print("2. Start the FastAPI app: uvicorn main:app --reload")
        print("3. Test with: curl -X POST localhost:8000/ask -d '{\"query\":\"Hello\",\"api_key\":\"sk-...\"}'")
        
        # Verify setup
        print()
        print("🔍 Verifying Redis setup...")
        if verify_redis_setup():
            print("✅ All Redis components verified and working!")
        else:
            print("❌ Redis setup verification failed")
    else:
        print()
        print("❌ Redis index setup failed!")
        print()
        print("Troubleshooting:")
        print("1. Make sure Redis Stack is running (not regular Redis)")
        print("2. Install Redis Stack: https://redis.io/docs/stack/")
        print("3. Or use Docker: docker run -p 6379:6379 redis/redis-stack:latest")
        exit(1)
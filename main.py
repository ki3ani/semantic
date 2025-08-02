import os
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, Field
import redis
from contextlib import asynccontextmanager
import boto3
from botocore.exceptions import ClientError
import time
from collections import defaultdict
import re
import asyncio


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The query to process")
    aws_access_key_id: str = Field(..., min_length=16, description="AWS Access Key ID")
    aws_secret_access_key: str = Field(..., min_length=20, description="AWS Secret Access Key")
    aws_region: str = Field(default="us-east-1", description="AWS Region")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        # Basic sanitization - remove control characters
        v = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', v)
        return v.strip()
    
    @validator('aws_access_key_id')
    def validate_access_key(cls, v):
        if not v or len(v) < 16:
            raise ValueError('Invalid AWS Access Key ID format')
        return v
    
    @validator('aws_secret_access_key')
    def validate_secret_key(cls, v):
        if not v or len(v) < 20:
            raise ValueError('Invalid AWS Secret Access Key format')
        return v


class QueryResponse(BaseModel):
    response: str
    from_cache: bool
    cached_at: Optional[str] = None
    processing_time_ms: Optional[float] = None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('semantic_cache.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rate limiting storage
rate_limit_storage = defaultdict(list)
RATE_LIMIT_REQUESTS = 100  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

def check_rate_limit(client_ip: str) -> bool:
    """Check if client is within rate limits"""
    now = time.time()
    # Clean old requests
    rate_limit_storage[client_ip] = [
        req_time for req_time in rate_limit_storage[client_ip] 
        if now - req_time < RATE_LIMIT_WINDOW
    ]
    
    # Check if under limit
    if len(rate_limit_storage[client_ip]) >= RATE_LIMIT_REQUESTS:
        return False
    
    # Add current request
    rate_limit_storage[client_ip].append(now)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    try:
        redis_client = redis.from_url(redis_url)
        redis_client.ping()
        logger.info("✓ Redis connection successful")
        
        # Test RedisJSON availability
        try:
            redis_client.json().get("test_key")
        except Exception:
            pass  # Key doesn't exist, that's fine
            
        app.state.redis = redis_client
        app.state.start_time = time.time()
        
        yield
        
    except redis.ConnectionError as e:
        logger.error(f"✗ Redis connection failed: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        raise
    finally:
        if hasattr(app.state, 'redis'):
            app.state.redis.close()
            logger.info("Redis connection closed")


app = FastAPI(
    title="Semantic Cache API",
    description="Smart caching layer for AI/LLM responses using vector similarity",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_embedding(text: str, aws_access_key_id: str, aws_secret_access_key: str, aws_region: str) -> List[float]:
    """Get embedding from Amazon Bedrock"""
    try:
        bedrock = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        
        # Use Amazon Titan Embeddings G1 - Text
        body = json.dumps({
            "inputText": text
        })
        
        response = bedrock.invoke_model(
            modelId='amazon.titan-embed-text-v1',
            body=body,
            contentType='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        embedding = response_body['embedding']
        
        logger.info(f"Generated embedding for query length: {len(text)}")
        return embedding
        
    except ClientError as e:
        logger.error(f"AWS Bedrock embedding error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"AWS Bedrock API error: {str(e)}")
    except Exception as e:
        logger.error(f"Bedrock embedding error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Embedding generation error: {str(e)}")


async def generate_llm_response(query: str, aws_access_key_id: str, aws_secret_access_key: str, aws_region: str) -> str:
    """Generate response using Amazon Bedrock (Claude)"""
    try:
        bedrock = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        
        # Use Amazon Titan Text Express (fast and cost-effective)
        body = json.dumps({
            "inputText": f"You are a helpful assistant. Provide a concise, accurate response to this query: {query}",
            "textGenerationConfig": {
                "maxTokenCount": 150,
                "temperature": 0.7,
                "topP": 0.9
            }
        })
        
        response = bedrock.invoke_model(
            modelId='amazon.titan-text-express-v1',
            body=body,
            contentType='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        llm_response = response_body['results'][0]['outputText'].strip()
        
        logger.info(f"Generated LLM response for query: {query[:50]}...")
        return llm_response
        
    except ClientError as e:
        logger.error(f"AWS Bedrock LLM error: {str(e)}")
        # Fallback to a generic response
        return "I apologize, but I'm unable to process your request at the moment. Please try again later."
    except Exception as e:
        logger.error(f"Bedrock LLM error: {str(e)}")
        # Fallback to a generic response
        return "I apologize, but I'm unable to process your request at the moment. Please try again later."


async def search_similar_cache(redis_client: redis.Redis, embedding: List[float], threshold: float = 0.85) -> Optional[dict]:
    """Search for similar cached results using Redis vector search"""
    try:
        # Convert embedding to bytes for Redis
        embedding_bytes = b''.join([float(x).hex().encode() for x in embedding])
        
        # Perform vector search using Redis Search
        query = f"*=>[KNN 10 @embedding $vec AS score]"
        params = {"vec": embedding_bytes}
        
        results = redis_client.ft("llm_cache_idx").search(
            query,
            query_params=params
        )
        
        # Check if any result meets the similarity threshold
        for doc in results.docs:
            if hasattr(doc, 'score') and float(doc.score) >= threshold:
                # Parse the cached document
                cached_data = json.loads(doc.json)
                
                # Check if cache entry is still valid (not expired)
                timestamp_str = cached_data.get("timestamp")
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    cache_age = datetime.now(timezone.utc) - timestamp
                    cache_ttl_hours = int(os.getenv("CACHE_TTL_HOURS", "24"))
                    
                    if cache_age > timedelta(hours=cache_ttl_hours):
                        logger.info(f"Cache entry expired: {cache_age.total_seconds()/3600:.1f} hours old")
                        continue
                
                logger.info(f"Cache hit with similarity: {float(doc.score):.3f}")
                return {
                    "response": cached_data.get("response"),
                    "similarity": float(doc.score),
                    "cached_at": cached_data.get("timestamp")
                }
        
        logger.info("No cache hit found")
        return None
        
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return None


async def save_to_cache(redis_client: redis.Redis, query: str, embedding: List[float], response: str, ttl_hours: int = 24) -> None:
    """Save query, embedding, and response to Redis cache with TTL"""
    try:
        # Generate UUID for cache key
        cache_key = f"cache:{uuid.uuid4()}"
        
        # Create cache document
        cache_doc = {
            "query": query,
            "embedding": embedding,
            "response": response,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Store using RedisJSON with TTL
        redis_client.json().set(cache_key, "$", cache_doc)
        redis_client.expire(cache_key, ttl_hours * 3600)  # TTL in seconds
        
        logger.info(f"Saved to cache with key: {cache_key}, TTL: {ttl_hours}h")
        
    except Exception as e:
        logger.error(f"Cache save error: {e}")


def get_client_ip(request: Request) -> str:
    """Get client IP address for rate limiting"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest, http_request: Request):
    start_time = time.time()
    client_ip = get_client_ip(http_request)
    
    # Rate limiting check
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Please try again later."
        )
    
    try:
        logger.info(f"Processing query from {client_ip}: {request.query[:100]}...")
        
        # Get embedding for the query
        embedding = await get_embedding(
            request.query, 
            request.aws_access_key_id, 
            request.aws_secret_access_key, 
            request.aws_region
        )
        
        # Search for similar cached results
        similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
        cached_result = await search_similar_cache(app.state.redis, embedding, similarity_threshold)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        if cached_result:
            # Cache hit - return cached response
            logger.info(f"Cache hit for query: {request.query[:50]}...")
            return QueryResponse(
                response=cached_result["response"],
                from_cache=True,
                cached_at=cached_result.get("cached_at"),
                processing_time_ms=processing_time
            )
        else:
            # No cache hit - generate real LLM response
            logger.info(f"Cache miss - generating new response for: {request.query[:50]}...")
            
            # Generate actual LLM response using Bedrock
            llm_response = await generate_llm_response(
                request.query,
                request.aws_access_key_id,
                request.aws_secret_access_key, 
                request.aws_region
            )
            
            # Save to cache for future use
            ttl_hours = int(os.getenv("CACHE_TTL_HOURS", "24"))
            await save_to_cache(app.state.redis, request.query, embedding, llm_response, ttl_hours)
            
            processing_time = (time.time() - start_time) * 1000
            
            return QueryResponse(
                response=llm_response,
                from_cache=False,
                processing_time_ms=processing_time
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error occurred")


# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Check Redis connection
        app.state.redis.ping()
        uptime = time.time() - app.state.start_time
        
        return {
            "status": "healthy",
            "uptime_seconds": uptime,
            "redis_connected": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


# Cache statistics endpoint
@app.get("/stats")
async def get_cache_stats():
    try:
        # Get Redis info
        redis_info = app.state.redis.info()
        
        return {
            "redis_memory_used": redis_info.get("used_memory_human", "unknown"),
            "redis_connected_clients": redis_info.get("connected_clients", 0),
            "uptime_seconds": time.time() - app.state.start_time,
            "rate_limit_active_ips": len(rate_limit_storage),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve stats")
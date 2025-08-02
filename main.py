import os
import json
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
from contextlib import asynccontextmanager
from openai import OpenAI


class QueryRequest(BaseModel):
    query: str
    api_key: str


class QueryResponse(BaseModel):
    response: str
    from_cache: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url)
    
    try:
        redis_client.ping()
        print("✓ Redis connection successful")
    except redis.ConnectionError:
        print("✗ Redis connection failed")
        raise
    
    app.state.redis = redis_client
    yield
    redis_client.close()


app = FastAPI(lifespan=lifespan)


async def get_embedding(text: str, api_key: str) -> List[float]:
    """Get embedding from OpenAI API"""
    try:
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OpenAI API error: {str(e)}")


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
                return {
                    "response": cached_data.get("response"),
                    "similarity": float(doc.score)
                }
        
        return None
    except Exception as e:
        print(f"Vector search error: {e}")
        return None


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    try:
        # Get embedding for the query
        embedding = await get_embedding(request.query, request.api_key)
        
        # Search for similar cached results
        cached_result = await search_similar_cache(app.state.redis, embedding, 0.85)
        
        if cached_result:
            return QueryResponse(
                response=cached_result["response"],
                from_cache=True
            )
        else:
            # No cache hit, return dummy response
            return QueryResponse(
                response="dummy response",
                from_cache=False
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
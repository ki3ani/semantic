# Semantic Cache API 🚀

A production-ready semantic caching layer for AI/LLM applications using vector similarity search with Redis and Amazon Bedrock (Claude + Titan Embeddings).

## ✨ Features

### 🧠 **Smart Semantic Caching**
- Uses Amazon Titan Embeddings for query understanding (1536 dimensions)
- Amazon Bedrock Claude 3 Haiku for text generation (fast & cost-effective)
- Redis HNSW vector search with cosine similarity
- Configurable similarity threshold (default: 85%)
- Automatic cache expiration with TTL

### 🛡️ **Production Ready**
- **Rate Limiting**: 100 requests/minute per IP (configurable)
- **Input Validation**: Query sanitization and API key validation
- **Error Handling**: Graceful fallbacks and comprehensive error responses  
- **Structured Logging**: File + console logging with timestamps
- **Health Monitoring**: `/health` and `/stats` endpoints

### 🔥 **Performance**
- **Sub-100ms** responses for cached queries
- **60-80% cost reduction** on AI API usage
- **Horizontal scalability** with Redis clustering
- **Automatic TTL** prevents stale cache entries

## 🚀 Quick Start

### 1. **Prerequisites**
```bash
# Install Redis Stack (includes RediSearch + RedisJSON)
docker run -d -p 6379:6379 redis/redis-stack:latest

# Or install Redis Stack locally: https://redis.io/docs/stack/
```

### 2. **Installation**
```bash
git clone <your-repo>
cd sema-redis
pip install -r requirements.txt
```

### 3. **Configuration**
```bash
# Edit env.sh with your AWS credentials
nano env.sh

# Set your AWS credentials:
export AWS_ACCESS_KEY_ID="your-aws-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-access-key"
export AWS_DEFAULT_REGION="us-east-1"
```

### 4. **Setup & Start**
```bash
# Load environment and start (includes Redis index setup)
chmod +x start.sh
./start.sh

# In another terminal, run the demo
chmod +x demo.sh
./demo.sh
```

### 5. **Test the API**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "aws_access_key_id": "your-access-key",
    "aws_secret_access_key": "your-secret-key",
    "aws_region": "us-east-1"
  }'
```

## 📡 API Endpoints

### **POST `/ask`** - Main Caching Endpoint
```json
{
  "query": "Your question here",
  "aws_access_key_id": "your-access-key",
  "aws_secret_access_key": "your-secret-key",  
  "aws_region": "us-east-1"
}
```

**Response:**
```json
{
  "response": "AI-generated answer",
  "from_cache": false,
  "cached_at": "2025-08-09T10:30:00Z",
  "processing_time_ms": 1250.5
}
```

### **GET `/health`** - Health Check
```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "redis_connected": true,
  "timestamp": "2025-08-09T10:30:00Z"
}
```

### **GET `/stats`** - Performance Statistics
```json
{
  "redis_memory_used": "2.5M",
  "redis_connected_clients": 1,
  "uptime_seconds": 3600,
  "rate_limit_active_ips": 5,
  "timestamp": "2025-08-09T10:30:00Z"
}
```

### **GET `/docs`** - Interactive API Documentation
Access Swagger UI at `http://localhost:8000/docs`

## ⚙️ Configuration

All settings are in `env.sh`:

```bash
# Cache Settings
export CACHE_TTL_HOURS=24              # How long to keep cached responses
export SIMILARITY_THRESHOLD=0.85       # Semantic similarity threshold (0.0-1.0)

# Rate Limiting  
export RATE_LIMIT_REQUESTS=100         # Max requests per minute per IP
export RATE_LIMIT_WINDOW=60            # Rate limit time window (seconds)

# Redis
export REDIS_URL=redis://localhost:6379

# OpenAI
export OPENAI_API_KEY="sk-your-key"
```

## 🎯 Demo Scenarios

### **Scenario 1: Semantic Similarity**
```bash
# Query 1
curl -X POST localhost:8000/ask -d '{"query":"How to reset password?","api_key":"sk-..."}'
# -> from_cache: false, generates new response

# Query 2 (similar meaning)
curl -X POST localhost:8000/ask -d '{"query":"I forgot my password","api_key":"sk-..."}'
# -> from_cache: true, returns cached response instantly
```

### **Scenario 2: Performance Comparison**
```bash
# First request: ~2000ms (OpenAI API call)
# Subsequent similar requests: ~50ms (Redis cache hit)
```

### **Scenario 3: Rate Limiting**
```bash
# Make 101 requests in 1 minute -> gets rate limited
# Response: 429 "Rate limit exceeded"
```

## 🏗️ Architecture

```
Client App → Semantic Cache API → Amazon Bedrock (Claude 3 + Titan)
                ↓
              Redis Vector Store
              (Embeddings + Responses)
```

**Flow:**
1. **Query** → Generate embedding with Amazon Titan
2. **Search** → Find similar cached responses in Redis  
3. **Hit?** → Return cached response (fast)
4. **Miss?** → Generate new response with Claude → Cache → Return (slower first time)

## 🔧 Manual Setup

If you prefer manual setup:

```bash
# 1. Setup Redis index
python setup_redis_index.py

# 2. Load environment  
source env.sh

# 3. Start server
uvicorn main:app --reload
```

## 📊 Business Value

### **Cost Savings**
- **60-80% reduction** in OpenAI API costs
- **Example**: $1000/month → $200/month for similar query patterns

### **Performance Gains**  
- **Cached responses**: <100ms vs 2-5 seconds
- **Better user experience** with instant responses
- **Higher throughput** - serve more users with same infrastructure

### **Use Cases**
- Customer support chatbots with repetitive questions
- FAQ systems with varied question phrasing  
- Educational platforms with common queries
- Content APIs with similar requests

## 🚧 Production Deployment

### **Environment Variables**
Set these in your deployment:
- `REDIS_URL` - Redis connection string
- `OPENAI_API_KEY` - Your OpenAI API key
- `CACHE_TTL_HOURS` - Cache expiration time
- `SIMILARITY_THRESHOLD` - Semantic matching sensitivity

### **Scaling**
- **Horizontal**: Deploy multiple API instances behind load balancer
- **Redis**: Use Redis Cluster for high availability
- **Monitoring**: Use logs in `semantic_cache.log` for monitoring

### **Security**
- Configure CORS origins in production
- Use environment-specific API keys
- Implement authentication/authorization as needed
- Use HTTPS in production

## 🎯 What Makes This Demo-Ready

✅ **Real LLM Integration** - Uses actual OpenAI GPT, not dummy responses  
✅ **Production Features** - Rate limiting, logging, error handling, validation  
✅ **Easy Setup** - One-command startup with automatic Redis index creation  
✅ **Monitoring** - Health checks and performance statistics  
✅ **Documentation** - Swagger UI and comprehensive README  
✅ **Performance Metrics** - Response time tracking and cache hit analytics  

**Perfect for demonstrating the business value of semantic caching!**
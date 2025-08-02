#!/bin/bash

# Redis Configuration
export REDIS_URL=redis://localhost:6379

# AWS Bedrock Configuration
export AWS_ACCESS_KEY_ID="your-aws-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-access-key"
export AWS_DEFAULT_REGION="us-east-1"

# Cache Configuration
export CACHE_TTL_HOURS=24                    # Cache expiry time in hours
export SIMILARITY_THRESHOLD=0.85             # Semantic similarity threshold (0.0-1.0)

# Rate Limiting
export RATE_LIMIT_REQUESTS=100               # Requests per minute per IP
export RATE_LIMIT_WINDOW=60                  # Time window in seconds

# Logging
export LOG_LEVEL=INFO                        # DEBUG, INFO, WARNING, ERROR

# Application Settings
export API_TITLE="Semantic Cache API"
export API_VERSION="1.0.0"

echo "✅ Environment variables loaded for Semantic Cache API"
echo "🔧 Redis URL: $REDIS_URL"
echo "⏱️  Cache TTL: $CACHE_TTL_HOURS hours"
echo "🎯 Similarity threshold: $SIMILARITY_THRESHOLD"
echo "🛡️  Rate limit: $RATE_LIMIT_REQUESTS requests per ${RATE_LIMIT_WINDOW}s"
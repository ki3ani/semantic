#!/bin/bash

echo "🚀 Starting Semantic Cache API..."
echo

# Load environment variables
if [ -f "env.sh" ]; then
    source env.sh
    echo
else
    echo "❌ env.sh not found! Please create it first."
    exit 1
fi

# Check if OpenAI API key is set
if [ "$OPENAI_API_KEY" = "sk-your-openai-api-key-here" ]; then
    echo "⚠️  WARNING: Please set your actual OpenAI API key in env.sh"
    echo "   Current value: $OPENAI_API_KEY"
    echo
fi

# Check if Redis is running
echo "🔍 Checking Redis connection..."
if python -c "import redis; redis.from_url('$REDIS_URL').ping()" 2>/dev/null; then
    echo "✅ Redis is running"
else
    echo "❌ Redis is not running or not accessible at $REDIS_URL"
    echo "   Please start Redis Stack or update REDIS_URL in env.sh"
    exit 1
fi

# Check if Redis index exists
echo "🔍 Checking Redis Search index..."
if python -c "import redis; r=redis.from_url('$REDIS_URL'); r.execute_command('FT.INFO', 'llm_cache_idx')" 2>/dev/null; then
    echo "✅ Redis index exists"
else
    echo "⚠️  Redis index not found - setting it up..."
    python setup_redis_simple.py
    if [ $? -ne 0 ]; then
        echo "❌ Failed to setup Redis index"
        exit 1
    fi
fi

echo
echo "🌟 Starting FastAPI server..."
echo "📡 API will be available at: http://localhost:8000"
echo "📚 API docs will be at: http://localhost:8000/docs"
echo "❤️  Health check: http://localhost:8000/health"
echo "📊 Statistics: http://localhost:8000/stats"
echo
echo "Press Ctrl+C to stop the server"
echo

# Start the FastAPI application
uvicorn main:app --reload --host 0.0.0.0 --port 8000
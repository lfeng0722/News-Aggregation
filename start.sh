#!/bin/bash

# FOBOH News Aggregation System - Simple Startup Script
# Version 3.0.0

echo "🚀 Starting FOBOH News Aggregation & RAG Chatbot System..."

# Check if conda is available and activate environment
if command -v conda &> /dev/null; then
    echo "📦 Activating conda environment..."
    source /home/linfeng/anaconda3/etc/profile.d/conda.sh
    conda activate foboh-news
else
    echo "⚠️  Conda not found, using system Python"
fi

# Check if we're in the correct directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run from the correct directory."
    exit 1
fi

# Create necessary directories
mkdir -p data/chroma_db

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Load .env file if it exists
if [ -f ".env" ]; then
    echo "📄 Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check API key status
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "⚠️  OpenAI API key not configured. System will work with fallback responses."
    echo "💡 To enable full AI features, set OPENAI_API_KEY in .env file"
else
    echo "✅ OpenAI API key configured - Full AI + RAG features enabled"
fi

echo ""
echo "🔧 Environment setup complete!"
echo "📊 Starting FOBOH News System..."
echo "🌐 Access the dashboard at: http://localhost:8000"
echo "📚 API documentation at: http://localhost:8000/docs"
echo ""
echo "🤖 Features:"
echo "   • Real news scraping from Australian sources"
echo "   • Vector database with semantic search"
echo "   • RAG-powered intelligent chatbot"
echo "   • Automatic duplicate detection"
echo "   • Breaking news identification"
echo ""
echo "⏳ Initial startup may take a few minutes to:"
echo "   • Initialize vector database"
echo "   • Download embedding models"
echo "   • Scrape initial news articles"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the application
python main.py 
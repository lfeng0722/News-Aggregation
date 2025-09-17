#!/usr/bin/env python3
"""
FOBOH News Aggregation & LangChain RAG Chatbot System
Enhanced AI-powered news aggregation system with LangChain, GPT-4o, and FAISS

Features:
- Real-time news scraping from Australian sources
- LangChain-based RAG with GPT-4o
- FAISS vector database for fast similarity search
- OpenAI embeddings for superior semantic understanding
- Advanced text chunking and retrieval
- Web dashboard interface
- Automatic duplicate detection
- Breaking news identification

Author: FOBOH Team
Version: 4.0.0 - LangChain + GPT-4o + FAISS
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Dict, Optional

# Web framework
from fastapi import FastAPI, HTTPException, BackgroundTasks, Form, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn

# Environment
from dotenv import load_dotenv

# Import our services and models
from app.services import RealNewsScraperService, get_langchain_rag_service
from app.core.database import init_db

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
langchain_rag_service = None

async def initialize_services():
    """Initialize all services"""
    global langchain_rag_service

    try:
        # Initialize database
        init_db()
        logger.info("Database initialized")

        # Initialize LangChain RAG service
        langchain_rag_service = get_langchain_rag_service()
        logger.info("LangChain RAG service initialized")

        # Check if we need to populate the vector store
        stats = langchain_rag_service.get_collection_stats()
        if stats['total_articles'] == 0:
            logger.info("Vector store is empty, scraping initial news...")
            await scrape_and_populate_news()
        else:
            logger.info(f"Vector store contains {stats['total_articles']} articles")

    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise

async def scrape_and_populate_news():
    """Scrape news and populate RAG database"""
    try:
        logger.info("Starting news scraping...")

        # Scrape news from all sources
        async with RealNewsScraperService() as scraper:
            articles = await scraper.scrape_all_sources(max_articles_per_source=10)

        if articles:
            # Add to FAISS vector store
            added_count = await langchain_rag_service.add_articles(articles)
            logger.info(f"Added {added_count} document chunks to FAISS vector store")
        else:
            logger.warning("No articles were scraped")

    except Exception as e:
        logger.error(f"Error in news scraping: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    logger.info("🚀 Starting FOBOH News Aggregation System (LangChain + GPT-4o + FAISS)...")

    try:
        await initialize_services()
        logger.info("✅ All services initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")

    yield

    logger.info("🔄 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="FOBOH News Aggregation & LangChain RAG Chatbot",
    description="AI-powered news aggregation system with LangChain, GPT-4o, and FAISS",
    version="4.0.0",
    lifespan=lifespan
)

# Mount static files and templates
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# API Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard showing daily highlights"""
    try:
        return templates.TemplateResponse("dashboard.html", {"request": request})
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        raise HTTPException(status_code=500, detail="Error loading dashboard")

@app.post("/api/extract-news")
async def trigger_news_extraction(background_tasks: BackgroundTasks):
    """Manually trigger news extraction pipeline"""
    try:
        # Add background task for news scraping
        background_tasks.add_task(scrape_and_populate_news)

        return {
            "message": "News extraction started! This will run in the background and update the FAISS vector store.",
            "status": "started",
            "rag_enabled": True,
            "vector_store": "FAISS",
            "llm_model": "GPT-4o"
        }
    except Exception as e:
        logger.error(f"Extraction trigger error: {e}")
        raise HTTPException(status_code=500, detail="Error starting extraction")

@app.get("/api/articles")
async def get_articles(
    category: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get articles with optional filtering"""
    try:
        if category:
            all_articles = await langchain_rag_service.search_by_category(category, limit=limit)
        else:
            # Get recent articles from all categories
            all_articles = []
            for cat in ["sports", "lifestyle", "music", "finance"]:
                cat_articles = await langchain_rag_service.search_by_category(cat, limit=limit//4)
                all_articles.extend(cat_articles)

        # Calculate frequency and format for API response
        title_frequency = {}
        for article in all_articles:
            # Simple frequency calculation based on similar titles
            title_words = set(article['title'].lower().split())
            for existing_title, count in title_frequency.items():
                existing_words = set(existing_title.lower().split())
                # If titles share 60% of words, consider them similar
                if len(title_words.intersection(existing_words)) / len(title_words.union(existing_words)) > 0.6:
                    title_frequency[existing_title] += 1
                    break
            else:
                title_frequency[article['title']] = 1

        formatted_articles = []
        for i, article in enumerate(all_articles[offset:offset+limit]):
            # Find frequency for this article
            article_frequency = 1
            title_words = set(article['title'].lower().split())
            for title, freq in title_frequency.items():
                title_check_words = set(title.lower().split())
                if len(title_words.intersection(title_check_words)) / len(title_words.union(title_check_words)) > 0.6:
                    article_frequency = freq
                    break

            # Ensure all required fields are present with defaults
            formatted_articles.append({
                "id": i + offset + 1,
                "title": article.get('title', 'No Title'),
                "content": article.get('content', ''),
                "summary": article.get('content', '')[:200] + "..." if len(article.get('content', '')) > 200 else article.get('content', ''),
                "author": article.get('author', 'Unknown'),
                "source_name": article.get('source', 'Unknown Source'),
                "article_url": article.get('url', '#'),
                "category": article.get('category', 'general'),
                "published_date": article.get('published_date', datetime.utcnow().isoformat()),
                "extracted_date": article.get('extracted_date', datetime.utcnow().isoformat()),
                "sentiment_score": 0.1,
                "priority_score": article.get('relevance_score', 3.5),
                "is_breaking_news": article.get('is_breaking_news', False),
                "frequency": article_frequency
            })

        return formatted_articles
    except Exception as e:
        logger.error(f"Get articles error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching articles")

@app.get("/api/categories/{category}/highlights")
async def get_category_highlights(category: str):
    """Get highlights for a specific category"""
    try:
        articles = await langchain_rag_service.search_by_category(category, limit=10)

        # Calculate frequency for category articles
        title_frequency = {}
        for article in articles:
            title_words = set(article['title'].lower().split())
            for existing_title, count in title_frequency.items():
                existing_words = set(existing_title.lower().split())
                if len(title_words.intersection(existing_words)) / len(title_words.union(existing_words)) > 0.6:
                    title_frequency[existing_title] += 1
                    break
            else:
                title_frequency[article['title']] = 1

        formatted_articles = []
        for i, article in enumerate(articles):
            # Find frequency for this article
            article_frequency = 1
            title_words = set(article['title'].lower().split())
            for title, freq in title_frequency.items():
                title_check_words = set(title.lower().split())
                if len(title_words.intersection(title_check_words)) / len(title_words.union(title_check_words)) > 0.6:
                    article_frequency = freq
                    break

            # Ensure all required fields are present with defaults
            formatted_articles.append({
                "id": i + 1,
                "title": article.get('title', 'No Title'),
                "content": article.get('content', ''),
                "summary": article.get('content', '')[:200] + "..." if len(article.get('content', '')) > 200 else article.get('content', ''),
                "author": article.get('author', 'Unknown'),
                "source_name": article.get('source', 'Unknown Source'),
                "article_url": article.get('url', '#'),
                "category": article.get('category', 'general'),
                "published_date": article.get('published_date', datetime.utcnow().isoformat()),
                "extracted_date": article.get('extracted_date', datetime.utcnow().isoformat()),
                "sentiment_score": 0.1,
                "priority_score": article.get('relevance_score', 3.5),
                "is_breaking_news": article.get('is_breaking_news', False),
                "frequency": article_frequency
            })

        return formatted_articles
    except Exception as e:
        logger.error(f"Category highlights error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching category highlights")

@app.post("/api/chat")
async def chat_with_langchain_rag_bot(message: str = Form()):
    """LangChain RAG-powered chatbot endpoint with GPT-4o"""
    try:
        # Use LangChain RAG service for intelligent responses
        rag_result = await langchain_rag_service.chat_with_rag(message)

        return {
            "message": message,
            "response": rag_result['response'],
            "sources_used": rag_result.get('sources_used', 0),
            "timestamp": datetime.utcnow().isoformat(),
            "ai_powered": True,
            "model_used": rag_result.get('model_used', 'gpt-4o'),
            "vector_store": rag_result.get('vector_store', 'faiss'),
            "sources": rag_result.get('sources', []),
            "error": rag_result.get('error')
        }

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {
            "message": message,
            "response": "I'm sorry, I encountered an error while processing your question. Please try again.",
            "sources_used": 0,
            "timestamp": datetime.utcnow().isoformat(),
            "ai_powered": False,
            "error": str(e)
        }

@app.get("/api/search")
async def search_news(q: str, category: Optional[str] = None, limit: int = 10):
    """Search news articles using LangChain + FAISS"""
    try:
        articles = await langchain_rag_service.search_articles(
            query=q,
            n_results=limit,
            category_filter=category
        )

        return {
            "query": q,
            "category": category,
            "results": articles,
            "total_found": len(articles),
            "vector_store": "FAISS",
            "search_method": "LangChain similarity search"
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Error searching articles")

@app.get("/api/breaking-news")
async def get_breaking_news():
    """Get breaking news"""
    try:
        articles = await langchain_rag_service.get_breaking_news(limit=5)
        return {
            "breaking_news": articles,
            "count": len(articles),
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Breaking news error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching breaking news")

@app.get("/api/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        rag_stats = langchain_rag_service.get_collection_stats()

        # Count breaking news
        breaking_articles = await langchain_rag_service.get_breaking_news(limit=100)
        breaking_count = len(breaking_articles)

        return {
            "total_articles": rag_stats['total_articles'],
            "total_chunks": rag_stats.get('total_chunks', 0),
            "articles_today": rag_stats['total_articles'],
            "breaking_news": breaking_count,
            "categories": rag_stats['categories'],
            "sources": rag_stats['sources'],
            "rag_status": {
                "enabled": True,
                "database_size": rag_stats['total_articles'],
                "vector_store_type": rag_stats.get('vector_store_type', 'FAISS'),
                "last_updated": rag_stats['last_updated']
            },
            "ai_status": {
                "openai_configured": bool(os.getenv('OPENAI_API_KEY')),
                "rag_enabled": True,
                "llm_model": rag_stats.get('llm_model', 'gpt-4o'),
                "embedding_model": rag_stats.get('embedding_model', 'text-embedding-ada-002'),
                "chatbot_intelligent": True
            }
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching stats")

@app.get("/api/category-summary/{category}")
async def get_category_summary_endpoint(category: str):
    """Get AI-powered summary of category news using GPT-4o"""
    try:
        # Use RAG to get category summary
        summary_query = f"Provide a summary of the latest {category} news"
        rag_result = await langchain_rag_service.chat_with_rag(summary_query, category_filter=category)
        
        return {
            "category": category,
            "summary": rag_result['response'],
            "sources_used": rag_result.get('sources_used', 0),
            "model_used": rag_result.get('model_used', 'gpt-4o'),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Category summary error: {e}")
        raise HTTPException(status_code=500, detail="Error generating category summary")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        rag_stats = langchain_rag_service.get_collection_stats() if langchain_rag_service else {"total_articles": 0}

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "langchain_rag": "operational" if langchain_rag_service else "not_initialized",
                "vector_store": "FAISS",
                "articles_count": rag_stats.get('total_articles', 0),
                "chunks_count": rag_stats.get('total_chunks', 0)
            },
            "ai_enabled": bool(os.getenv('OPENAI_API_KEY')),
            "rag_enabled": True,
            "version": "4.0.0",
            "tech_stack": {
                "framework": "LangChain",
                "llm": "GPT-4o",
                "vector_store": "FAISS",
                "embeddings": "OpenAI text-embedding-ada-002"
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
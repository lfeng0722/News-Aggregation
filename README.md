# FOBOH News Aggregation & LangChain RAG Chatbot System

**Version 4.0.0** - A comprehensive AI-powered news aggregation system with LangChain, GPT-4o, and FAISS for Australian news outlets.

## 🚀 Features

- **Real-time News Scraping**: Automatically extracts news from Australian sources (ABC, Nine News)
- **LangChain RAG-Powered Chatbot**: Intelligent chatbot using LangChain, GPT-4o, and FAISS vector search
- **FAISS Vector Database**: High-performance similarity search with OpenAI embeddings
- **GPT-4o Integration**: Advanced language model for superior response quality
- **Breaking News Detection**: Automatic identification of urgent news
- **Web Dashboard**: Modern, responsive web interface
- **Duplicate Detection**: Content hashing to avoid duplicate articles
- **Multi-Category Support**: Sports, Lifestyle, Music, Finance categories
- **Advanced Text Chunking**: Intelligent document splitting for better retrieval

## 📋 Quick Start

### Prerequisites

- **Python 3.10+**
- **Conda** (recommended) or pip
- **OpenAI API Key** (required for GPT-4o and embeddings)

### Installation

1. **Clone and navigate to the project:**
   ```bash
   git clone <your-repo-url>
   cd FOBOH
   ```

2. **Create and activate conda environment:**
   ```bash
   conda create -n foboh-news python=3.10 -y
   conda activate foboh-news
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables (required):**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

5. **Start the system:**
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

### Access the Application

- **Dashboard**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🏗️ System Architecture

```
FOBOH System v4.0.0
├── News Scraper Service
│   ├── RSS Feed Processing
│   ├── Content Extraction
│   └── Breaking News Detection
├── LangChain RAG Service
│   ├── FAISS Vector Database
│   ├── OpenAI Embeddings (text-embedding-ada-002)
│   ├── GPT-4o Language Model
│   ├── Recursive Text Splitting
│   └── Semantic Search & Retrieval
├── Legacy Services (ChromaDB backup)
│   ├── RAG Service (ChromaDB)
│   └── RAG Chatbot Service
└── Web Interface
    ├── News Dashboard
    ├── Interactive LangChain Chatbot
    └── System Statistics
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/` | GET | Main dashboard interface |
| `/api/articles` | GET | Get news articles (with filtering) |
| `/api/categories/{category}/highlights` | GET | Get category-specific highlights |
| `/api/chat` | POST | Chat with LangChain RAG-powered bot |
| `/api/extract-news` | POST | Trigger news extraction |
| `/api/search` | GET | Search articles using LangChain + FAISS |
| `/api/breaking-news` | GET | Get breaking news articles |
| `/api/stats` | GET | Get system statistics |
| `/api/category-summary/{category}` | GET | AI-powered category summaries |
| `/health` | GET | Health check endpoint |

## 🤖 LangChain Chatbot Capabilities

The advanced LangChain RAG chatbot powered by GPT-4o can answer questions about:

- **Latest News**: "What's the latest sports news?"
- **Specific Topics**: "Tell me about AFL finals"
- **Breaking News**: "Any breaking news today?"
- **Category Summaries**: "Summarize finance news"
- **Complex Queries**: "Compare recent sports and finance developments"

### Example Interactions

```
User: "What's happening in Australian sports?"
Bot: "Based on recent articles from ABC Sport and Nine Sport, here are the key developments: [provides contextual GPT-4o powered response with sources]"

User: "Any breaking news?"
Bot: "🚨 I found 3 breaking news stories: [lists urgent articles with details and AI analysis]"

User: "Summarize the latest finance news"
Bot: [Provides comprehensive AI-generated summary using GPT-4o with source citations]
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI API Configuration (required)
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
FAISS_INDEX_PATH=./data/faiss_index
DATABASE_PATH=./data/chroma_db

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
MAX_ARTICLES_PER_SOURCE=10
```

### News Sources

The system automatically scrapes from:

- **ABC Sport**: Australian Broadcasting Corporation Sports
- **Nine Sport**: Nine Network Sports
- **ABC Lifestyle**: ABC Lifestyle content
- **Nine Finance**: Nine Network Finance news

## 🔧 Technical Details

### Dependencies

**Core Framework:**
- `fastapi` - Web framework and API
- `uvicorn` - ASGI server

**LangChain & AI:**
- `langchain` - RAG framework and chains
- `langchain-openai` - OpenAI integrations
- `openai` - GPT-4o and embeddings
- `faiss-cpu` - High-performance vector search

**Legacy AI Support:**
- `sentence-transformers` - Text embeddings (backup)
- `chromadb` - Vector database (backup)

**Web Scraping:**
- `aiohttp` - Async HTTP client
- `feedparser` - RSS feed parsing
- `newspaper3k` - Article extraction
- `beautifulsoup4` - HTML parsing

**Data Processing:**
- `python-dateutil` - Date parsing
- `python-dotenv` - Environment variables
- `sqlalchemy` - Database ORM

### Performance Features

- **Async Processing**: All I/O operations are asynchronous
- **FAISS Optimization**: High-performance similarity search
- **OpenAI Embeddings**: Superior semantic understanding
- **GPT-4o Integration**: Advanced language model responses
- **Intelligent Chunking**: Recursive text splitting for optimal retrieval
- **Background Tasks**: News extraction runs in background
- **Caching**: Vector embeddings cached in FAISS index

## 📱 Web Interface Features

### Dashboard
- **Live News Feed**: Real-time article updates
- **Category Filtering**: Filter by sports, lifestyle, finance, music
- **Breaking News Badges**: Visual indicators for urgent news
- **Source Attribution**: Clear source and author information
- **AI-Powered Summaries**: Category summaries using GPT-4o

### LangChain Chatbot Interface
- **Interactive Chat**: Real-time conversation with GPT-4o powered bot
- **Source Citations**: Shows which articles were used for responses
- **Advanced AI Responses**: GPT-4o powered intelligent responses
- **Context Awareness**: Maintains conversation context
- **Vector Search Integration**: FAISS-powered semantic search

### System Monitoring
- **Live Statistics**: Total articles, chunks, breaking news count
- **AI Service Status**: LangChain, GPT-4o, and FAISS health monitoring
- **Performance Metrics**: Response times and accuracy indicators
- **Vector Store Analytics**: FAISS index statistics

## 🚨 Troubleshooting

### Common Issues

**1. "OpenAI API Key not found":**
```bash
# Ensure .env file exists with valid API key
echo "OPENAI_API_KEY=your_actual_key_here" > .env
```

**2. "FAISS index initialization failed":**
```bash
# Clear FAISS index and restart
rm -rf data/faiss_index
./start.sh
```

**3. "LangChain import errors":**
```bash
# Ensure conda environment is activated and dependencies installed
conda activate foboh-news
pip install -r requirements.txt
```

**4. "No articles found":**
```bash
# Manually trigger news extraction
curl -X POST http://localhost:8000/api/extract-news
```

**5. "403 Forbidden" during scraping:**
- This is normal - some news sites block automated access
- The system uses fallback content from RSS feeds
- Articles will still be collected from available sources

### Logs and Debugging

The system provides detailed logging:
```bash
# View logs in real-time
tail -f logs/app.log  # If log file exists

# Or check console output for debugging
python main.py
```

## 📈 System Monitoring

### Health Checks

Monitor system health:
```bash
# Check system status
curl http://localhost:8000/health

# Get detailed statistics
curl http://localhost:8000/api/stats
```

### Performance Metrics

The system tracks:
- **Article Count**: Total articles in FAISS database
- **Document Chunks**: Total text chunks for retrieval
- **Scraping Success Rate**: Percentage of successful extractions
- **AI Response Time**: Average GPT-4o response time
- **Vector Search Performance**: FAISS query performance
- **Breaking News Detection**: Accuracy of urgent news identification

## 🔄 Updates and Maintenance

### Regular Maintenance

1. **Database Cleanup**: Old articles are automatically managed
2. **Vector Index Updates**: FAISS index is incrementally updated
3. **Model Caching**: OpenAI embeddings are cached for efficiency
4. **Source Monitoring**: RSS feeds are checked for availability

### Manual Updates

```bash
# Update news manually
curl -X POST http://localhost:8000/api/extract-news

# Clear and rebuild FAISS index (if needed)
rm -rf data/faiss_index
./start.sh

# Test LangChain chatbot
curl -X POST http://localhost:8000/api/chat -d "message=What's the latest news?"
```

## 📝 Development

### Project Structure

```
FOBOH/
├── main.py                    # Main FastAPI application
├── start.sh                   # Startup script
├── README.md                  # This documentation
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
├── app/
│   ├── core/
│   │   └── database.py        # Database initialization
│   ├── models/                # Data models
│   │   ├── article.py
│   │   ├── highlight.py
│   │   └── source.py
│   └── services/              # Service layer
│       ├── langchain_rag_service.py  # Main LangChain service
│       ├── real_news_scraper.py      # News scraping
│       ├── rag_service.py            # Legacy ChromaDB service
│       └── rag_chatbot.py            # Legacy chatbot service
├── config/
│   └── news_sources.json      # News source configuration
├── data/
│   ├── faiss_index/           # FAISS vector store (auto-created)
│   └── chroma_db/             # ChromaDB backup (auto-created)
└── templates/
    └── dashboard.html         # Web dashboard
```

### Adding New Features

**LangChain RAG Service** (`app/services/langchain_rag_service.py`):
- Add new retrieval strategies
- Modify text chunking approaches
- Enhance GPT-4o prompts
- Implement new vector search methods

**News Scraper** (`app/services/real_news_scraper.py`):
- Add new Australian news sources
- Implement new extraction methods
- Enhance breaking news detection

**Main Application** (`main.py`):
- Add new API endpoints
- Modify dashboard functionality
- Implement new chatbot features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes in the appropriate service files
4. Test thoroughly with `./start.sh`
5. Ensure OpenAI API key is configured for testing
6. Submit a pull request

## 📄 License

This project is developed for the FOBOH organization. All rights reserved.

## 🆘 Support

For issues and questions:

1. Check the **Troubleshooting** section above
2. Verify OpenAI API key configuration
3. Review logs for error details
4. Test with `./start.sh` after any changes
5. Ensure all dependencies are installed correctly

---

**FOBOH News Aggregation System v4.0.0** - Powered by LangChain, GPT-4o, and FAISS! 🇦🇺🤖✨ 
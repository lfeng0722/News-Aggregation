import os
import logging
import asyncio
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)

class RAGService:
    """Retrieval-Augmented Generation service using ChromaDB for vector storage"""

    def __init__(self, db_path: str = "./data/chroma_db"):
        self.db_path = db_path
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.model_name = "all-MiniLM-L6-v2"  # Fast and good quality

        # Ensure data directory exists
        os.makedirs(db_path, exist_ok=True)

        self.initialize()

    def initialize(self):
        """Initialize ChromaDB client and embedding model"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="news_articles",
                metadata={"description": "Australian news articles for RAG"}
            )

            # Initialize embedding model
            logger.info(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)

            logger.info("RAG service initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing RAG service: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def add_articles(self, articles: List[Dict]) -> int:
        """Add articles to the vector database"""
        if not articles:
            return 0

        try:
            documents = []
            metadatas = []
            ids = []

            for article in articles:
                # Create searchable text combining title and content
                searchable_text = f"{article.get('title', '')} {article.get('content', '')}"

                if len(searchable_text.strip()) < 50:  # Skip very short articles
                    continue

                # Generate unique ID with timestamp to avoid duplicates
                content_hash = article.get('content_hash', str(uuid.uuid4()))
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                article_id = f"{content_hash}_{timestamp}_{uuid.uuid4().hex[:8]}"

                # Prepare metadata
                metadata = {
                    'title': article.get('title', '')[:500],  # ChromaDB has metadata limits
                    'source': article.get('source_name', ''),
                    'category': article.get('category', 'lifestyle'),
                    'url': article.get('url', ''),
                    'author': article.get('author', 'Unknown'),
                    'published_date': article.get('published_date', datetime.utcnow()).isoformat() if article.get('published_date') else datetime.utcnow().isoformat(),
                    'extracted_date': article.get('extracted_date', datetime.utcnow()).isoformat() if article.get('extracted_date') else datetime.utcnow().isoformat(),
                    'is_breaking_news': article.get('is_breaking_news', False),
                    'content_length': len(article.get('content', ''))
                }

                documents.append(searchable_text)
                metadatas.append(metadata)
                ids.append(article_id)

            if not documents:
                logger.warning("No valid articles to add to RAG database")
                return 0

            # Add to ChromaDB collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(documents)} articles to RAG database")
            return len(documents)

        except Exception as e:
            logger.error(f"Error adding articles to RAG database: {e}")
            return 0

    async def search_articles(self, query: str, n_results: int = 5, category_filter: Optional[str] = None) -> List[Dict]:
        """Search for relevant articles using vector similarity"""
        try:
            # Prepare where clause for filtering
            where_clause = {}
            if category_filter:
                where_clause["category"] = category_filter

            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )

            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i] if results.get('distances') else 0.0

                    formatted_results.append({
                        'content': doc,
                        'title': metadata.get('title', ''),
                        'source': metadata.get('source', ''),
                        'category': metadata.get('category', ''),
                        'url': metadata.get('url', ''),
                        'author': metadata.get('author', ''),
                        'published_date': metadata.get('published_date', ''),
                        'is_breaking_news': metadata.get('is_breaking_news', False),
                        'relevance_score': max(0.0, 1.0 - distance),  # Convert distance to similarity, ensure non-negative
                        'content_length': metadata.get('content_length', 0)
                    })

            logger.info(f"Found {len(formatted_results)} relevant articles for query: '{query}'")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching RAG database: {e}")
            return []

    async def search_by_category(self, category: str, limit: int = 10) -> List[Dict]:
        """Get articles by category"""
        try:
            results = self.collection.get(
                where={"category": category},
                limit=limit
            )

            formatted_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    metadata = results['metadatas'][i]

                    formatted_results.append({
                        'content': doc,
                        'title': metadata.get('title', ''),
                        'source': metadata.get('source', ''),
                        'category': metadata.get('category', ''),
                        'url': metadata.get('url', ''),
                        'author': metadata.get('author', ''),
                        'published_date': metadata.get('published_date', ''),
                        'is_breaking_news': metadata.get('is_breaking_news', False),
                        'content_length': metadata.get('content_length', 0)
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Error getting articles by category: {e}")
            return []

    async def get_breaking_news(self, limit: int = 5) -> List[Dict]:
        """Get breaking news articles"""
        try:
            results = self.collection.get(
                where={"is_breaking_news": True},
                limit=limit
            )

            formatted_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    metadata = results['metadatas'][i]

                    formatted_results.append({
                        'content': doc,
                        'title': metadata.get('title', ''),
                        'source': metadata.get('source', ''),
                        'category': metadata.get('category', ''),
                        'url': metadata.get('url', ''),
                        'author': metadata.get('author', ''),
                        'published_date': metadata.get('published_date', ''),
                        'is_breaking_news': True,
                        'content_length': metadata.get('content_length', 0)
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Error getting breaking news: {e}")
            return []

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()

            # Get sample of metadata to analyze categories
            sample = self.collection.get(limit=1000)
            categories = {}
            sources = {}

            if sample['metadatas']:
                for metadata in sample['metadatas']:
                    category = metadata.get('category', 'unknown')
                    source = metadata.get('source', 'unknown')

                    categories[category] = categories.get(category, 0) + 1
                    sources[source] = sources.get(source, 0) + 1

            return {
                'total_articles': count,
                'categories': categories,
                'sources': sources,
                'last_updated': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'total_articles': 0, 'categories': {}, 'sources': {}}

    async def clear_old_articles(self, days_old: int = 7):
        """Clear articles older than specified days"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)

            # Get all articles
            results = self.collection.get()

            ids_to_delete = []
            if results['metadatas']:
                for i, metadata in enumerate(results['metadatas']):
                    extracted_date_str = metadata.get('extracted_date')
                    if extracted_date_str:
                        try:
                            extracted_date = datetime.fromisoformat(extracted_date_str.replace('Z', '+00:00'))
                            if extracted_date < cutoff_date:
                                ids_to_delete.append(results['ids'][i])
                        except:
                            continue

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} old articles")

        except Exception as e:
            logger.error(f"Error clearing old articles: {e}")

    async def reset_database(self):
        """Reset the entire database (use with caution)"""
        try:
            self.client.reset()
            self.initialize()
            logger.info("RAG database reset successfully")
        except Exception as e:
            logger.error(f"Error resetting database: {e}")

# Global RAG service instance
rag_service = None

def get_rag_service() -> RAGService:
    """Get global RAG service instance"""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service 
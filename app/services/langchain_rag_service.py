import os
import logging
import asyncio
import pickle
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import uuid

# LangChain imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Environment
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class LangChainRAGService:
    """LangChain-based RAG service using GPT-4o and FAISS"""

    def __init__(self, vector_store_path: str = "./data/faiss_index"):
        self.vector_store_path = vector_store_path
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.text_splitter = None
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        
        self.initialize()

    def initialize(self):
        """Initialize LangChain components"""
        try:
            # Check if OpenAI API key is available
            if not os.getenv('OPENAI_API_KEY'):
                raise ValueError("OPENAI_API_KEY not found in environment variables")

            # Initialize embeddings
            logger.info("Initializing OpenAI embeddings...")
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",  # Stable OpenAI embedding model
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )

            # Initialize LLM (GPT-4o)
            logger.info("Initializing GPT-4o...")
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.7,
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                max_tokens=500
            )

            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )

            # Try to load existing vector store
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded existing FAISS vector store")
            except:
                logger.info("No existing vector store found, will create new one")
                self.vector_store = None

            # Initialize QA chain
            self._setup_qa_chain()

            logger.info("LangChain RAG service initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing LangChain RAG service: {e}")
            raise

    def _setup_qa_chain(self):
        """Setup the QA chain with custom prompt"""
        if not self.vector_store:
            return

        # Custom prompt template
        prompt_template = """You are a helpful Australian news assistant. Use the following news articles to answer the user's question about current events in Australia.

Context from recent Australian news articles:
{context}

Question: {question}

Instructions:
- Provide accurate, informative responses based on the news articles
- Focus on Australian news and current events
- Mention specific sources when relevant
- If the articles contain relevant information, summarize the key points
- Be conversational and helpful

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

    async def add_articles(self, articles: List[Dict]) -> int:
        """Add articles to the FAISS vector store"""
        if not articles:
            return 0

        try:
            documents = []
            
            for article in articles:
                # Create document content
                content = f"Title: {article.get('title', '')}\n\n{article.get('content', '')}"
                
                if len(content.strip()) < 100:  # Skip very short articles
                    continue

                # Create metadata
                metadata = {
                    'title': article.get('title', ''),
                    'source': article.get('source_name', ''),
                    'category': article.get('category', 'general'),
                    'url': article.get('url', ''),
                    'author': article.get('author', 'Unknown'),
                    'published_date': str(article.get('published_date', '')),
                    'extracted_date': str(article.get('extracted_date', datetime.utcnow())),
                    'is_breaking_news': article.get('is_breaking_news', False),
                    'content_hash': article.get('content_hash', str(uuid.uuid4()))
                }

                # Split text into chunks
                text_chunks = self.text_splitter.split_text(content)
                
                # Create documents for each chunk
                for i, chunk in enumerate(text_chunks):
                    doc_metadata = metadata.copy()
                    doc_metadata['chunk_id'] = i
                    doc_metadata['total_chunks'] = len(text_chunks)
                    
                    documents.append(Document(
                        page_content=chunk,
                        metadata=doc_metadata
                    ))

            if not documents:
                logger.warning("No valid documents to add to vector store")
                return 0

            # Create or update vector store
            if self.vector_store is None:
                logger.info("Creating new FAISS vector store...")
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                logger.info("Adding documents to existing FAISS vector store...")
                self.vector_store.add_documents(documents)

            # Save vector store
            self.vector_store.save_local(self.vector_store_path)
            
            # Setup QA chain if not already done
            if self.qa_chain is None:
                self._setup_qa_chain()

            logger.info(f"Added {len(documents)} document chunks to FAISS vector store")
            return len(documents)

        except Exception as e:
            logger.error(f"Error adding articles to FAISS vector store: {e}")
            return 0

    async def search_articles(self, query: str, n_results: int = 5, category_filter: Optional[str] = None) -> List[Dict]:
        """Search for relevant articles using FAISS similarity search"""
        try:
            if not self.vector_store:
                logger.warning("Vector store not initialized")
                return []

            # Perform similarity search
            if category_filter:
                # Filter by category
                docs = self.vector_store.similarity_search(
                    query, 
                    k=n_results * 2,  # Get more to filter
                    filter={"category": category_filter}
                )
                docs = docs[:n_results]  # Limit results
            else:
                docs = self.vector_store.similarity_search(query, k=n_results)

            # Format results
            formatted_results = []
            seen_articles = set()  # To avoid duplicate articles
            
            for doc in docs:
                metadata = doc.metadata
                content_hash = metadata.get('content_hash', '')
                
                # Skip if we've already seen this article
                if content_hash in seen_articles:
                    continue
                seen_articles.add(content_hash)

                formatted_results.append({
                    'content': doc.page_content,
                    'title': metadata.get('title', ''),
                    'source': metadata.get('source', ''),
                    'category': metadata.get('category', ''),
                    'url': metadata.get('url', ''),
                    'author': metadata.get('author', ''),
                    'published_date': metadata.get('published_date', ''),
                    'extracted_date': metadata.get('extracted_date', datetime.utcnow().isoformat()),
                    'is_breaking_news': metadata.get('is_breaking_news', False),
                    'relevance_score': 0.8,  # FAISS doesn't return scores by default
                    'content_length': len(doc.page_content)
                })

            logger.info(f"Found {len(formatted_results)} relevant articles for query: '{query}'")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching FAISS vector store: {e}")
            return []

    async def chat_with_rag(self, query: str, category_filter: Optional[str] = None) -> Dict:
        """Chat using LangChain QA chain with RAG"""
        try:
            if not self.qa_chain:
                return {
                    'response': "RAG system is not properly initialized. Please check the configuration.",
                    'sources': [],
                    'error': 'QA chain not initialized'
                }

            # Run the QA chain
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.qa_chain({"query": query})
            )

            # Extract source information
            sources = []
            if 'source_documents' in result:
                seen_sources = set()
                for doc in result['source_documents']:
                    metadata = doc.metadata
                    source_key = (metadata.get('title', ''), metadata.get('source', ''))
                    
                    if source_key not in seen_sources:
                        seen_sources.add(source_key)
                        sources.append({
                            'title': metadata.get('title', ''),
                            'source': metadata.get('source', ''),
                            'url': metadata.get('url', ''),
                            'category': metadata.get('category', ''),
                            'relevance': 0.8
                        })

            return {
                'response': result['result'],
                'sources': sources[:3],  # Limit to top 3 sources
                'sources_used': len(sources),
                'model_used': 'gpt-4o',
                'vector_store': 'faiss'
            }

        except Exception as e:
            logger.error(f"Error in RAG chat: {e}")
            return {
                'response': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'error': str(e)
            }

    async def search_by_category(self, category: str, limit: int = 10) -> List[Dict]:
        """Get articles by category"""
        try:
            if not self.vector_store:
                return []

            # Search with category filter
            docs = self.vector_store.similarity_search(
                f"{category} news",
                k=limit * 2,
                filter={"category": category}
            )

            # Format and deduplicate results
            formatted_results = []
            seen_articles = set()
            
            for doc in docs[:limit]:
                metadata = doc.metadata
                content_hash = metadata.get('content_hash', '')
                
                if content_hash in seen_articles:
                    continue
                seen_articles.add(content_hash)

                formatted_results.append({
                    'content': doc.page_content,
                    'title': metadata.get('title', ''),
                    'source': metadata.get('source', ''),
                    'category': metadata.get('category', ''),
                    'url': metadata.get('url', ''),
                    'author': metadata.get('author', ''),
                    'published_date': metadata.get('published_date', ''),
                    'extracted_date': metadata.get('extracted_date', datetime.utcnow().isoformat()),
                    'is_breaking_news': metadata.get('is_breaking_news', False),
                    'relevance_score': metadata.get('relevance_score', 3.5),
                    'content_length': len(doc.page_content)
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error getting articles by category: {e}")
            return []

    async def get_breaking_news(self, limit: int = 5) -> List[Dict]:
        """Get breaking news articles"""
        try:
            if not self.vector_store:
                return []

            # Search for breaking news
            docs = self.vector_store.similarity_search(
                "breaking news urgent alert",
                k=limit * 3,
                filter={"is_breaking_news": True}
            )

            # Format results
            formatted_results = []
            seen_articles = set()
            
            for doc in docs:
                if len(formatted_results) >= limit:
                    break
                    
                metadata = doc.metadata
                content_hash = metadata.get('content_hash', '')
                
                if content_hash in seen_articles:
                    continue
                seen_articles.add(content_hash)

                formatted_results.append({
                    'content': doc.page_content,
                    'title': metadata.get('title', ''),
                    'source': metadata.get('source', ''),
                    'category': metadata.get('category', ''),
                    'url': metadata.get('url', ''),
                    'author': metadata.get('author', ''),
                    'published_date': metadata.get('published_date', ''),
                    'extracted_date': metadata.get('extracted_date', datetime.utcnow().isoformat()),
                    'is_breaking_news': True,
                    'relevance_score': metadata.get('relevance_score', 4.0),
                    'content_length': len(doc.page_content)
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error getting breaking news: {e}")
            return []

    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector store"""
        try:
            if not self.vector_store:
                return {'total_articles': 0, 'categories': {}, 'sources': {}}

            # Get sample of documents to analyze
            sample_docs = self.vector_store.similarity_search("news", k=1000)
            
            categories = {}
            sources = {}
            total_chunks = len(sample_docs)
            unique_articles = set()

            for doc in sample_docs:
                metadata = doc.metadata
                category = metadata.get('category', 'unknown')
                source = metadata.get('source', 'unknown')
                content_hash = metadata.get('content_hash', '')
                
                categories[category] = categories.get(category, 0) + 1
                sources[source] = sources.get(source, 0) + 1
                unique_articles.add(content_hash)

            return {
                'total_articles': len(unique_articles),
                'total_chunks': total_chunks,
                'categories': categories,
                'sources': sources,
                'last_updated': datetime.utcnow().isoformat(),
                'vector_store_type': 'FAISS',
                'embedding_model': 'text-embedding-ada-002',
                'llm_model': 'gpt-4o'
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'total_articles': 0, 'categories': {}, 'sources': {}}

    async def reset_vector_store(self):
        """Reset the vector store (use with caution)"""
        try:
            import shutil
            if os.path.exists(self.vector_store_path):
                shutil.rmtree(self.vector_store_path)
            self.vector_store = None
            self.qa_chain = None
            logger.info("Vector store reset successfully")
        except Exception as e:
            logger.error(f"Error resetting vector store: {e}")

# Global service instance
langchain_rag_service = None

def get_langchain_rag_service() -> LangChainRAGService:
    """Get global LangChain RAG service instance"""
    global langchain_rag_service
    if langchain_rag_service is None:
        langchain_rag_service = LangChainRAGService()
    return langchain_rag_service 
import os
import logging
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from .rag_service import get_rag_service

# AI imports with fallback
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class RAGChatbotService:
    """RAG-powered chatbot for news queries"""

    def __init__(self):
        self.rag_service = get_rag_service()
        self.openai_client = None

        # Initialize OpenAI if available and configured
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            try:
                self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                logger.info("RAG Chatbot initialized with OpenAI")
            except Exception as e:
                logger.error(f"Error initializing OpenAI: {e}")
        else:
            logger.info("RAG Chatbot initialized without OpenAI (using fallback responses)")

    async def chat(self, message: str, category_filter: Optional[str] = None) -> Dict:
        """Main chat interface"""
        try:
            # Analyze the query
            query_analysis = await self.analyze_query(message)

            # Search for relevant articles
            relevant_articles = await self.search_relevant_articles(
                message,
                category_filter=category_filter or query_analysis.get('category'),
                n_results=5
            )

            # Generate response
            if self.openai_client and relevant_articles:
                response = await self.generate_ai_response(message, relevant_articles, query_analysis)
            else:
                response = await self.generate_fallback_response(message, relevant_articles, query_analysis)

            return {
                'message': message,
                'response': response,
                'sources_used': len(relevant_articles),
                'category': query_analysis.get('category'),
                'query_type': query_analysis.get('type'),
                'timestamp': datetime.utcnow().isoformat(),
                'ai_powered': bool(self.openai_client),
                'sources': [
                    {
                        'title': article['title'],
                        'source': article['source'],
                        'url': article['url'],
                        'relevance': article['relevance_score']
                    }
                    for article in relevant_articles[:3]  # Top 3 sources
                ]
            }

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {
                'message': message,
                'response': "I'm sorry, I encountered an error while processing your question. Please try again.",
                'sources_used': 0,
                'timestamp': datetime.utcnow().isoformat(),
                'ai_powered': False,
                'error': str(e)
            }

    async def analyze_query(self, query: str) -> Dict:
        """Analyze the user query to understand intent and category"""
        query_lower = query.lower()

        # Determine query type
        query_type = 'general'
        if any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where']):
            query_type = 'question'
        elif any(word in query_lower for word in ['latest', 'recent', 'today', 'news']):
            query_type = 'latest_news'
        elif any(word in query_lower for word in ['breaking', 'urgent', 'alert']):
            query_type = 'breaking_news'
        elif any(word in query_lower for word in ['summary', 'summarize', 'overview']):
            query_type = 'summary'

        # Determine category
        category = None
        if any(word in query_lower for word in ['sport', 'afl', 'nrl', 'cricket', 'tennis', 'football', 'rugby']):
            category = 'sports'
        elif any(word in query_lower for word in ['music', 'album', 'song', 'artist', 'concert', 'festival']):
            category = 'music'
        elif any(word in query_lower for word in ['finance', 'market', 'asx', 'stock', 'economy', 'investment']):
            category = 'finance'
        elif any(word in query_lower for word in ['lifestyle', 'health', 'travel', 'food', 'culture']):
            category = 'lifestyle'

        return {
            'type': query_type,
            'category': category,
            'keywords': self.extract_keywords(query)
        }

    def extract_keywords(self, query: str) -> List[str]:
        """Extract key terms from the query"""
        # Simple keyword extraction - could be enhanced with NLP
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'about', 'what', 'how', 'when', 'where', 'why', 'who'}

        words = query.lower().replace('?', '').replace('.', '').replace(',', '').split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        return keywords[:10]  # Limit to top 10 keywords

    async def search_relevant_articles(self, query: str, category_filter: Optional[str] = None, n_results: int = 5) -> List[Dict]:
        """Search for articles relevant to the query"""
        try:
            # Use RAG service to search
            articles = await self.rag_service.search_articles(
                query=query,
                n_results=n_results,
                category_filter=category_filter
            )

            # Filter out very low relevance articles (ChromaDB distances can be negative)
            filtered_articles = [
                article for article in articles
                if article.get('relevance_score', -1) > -0.8  # More permissive threshold
            ]

            return filtered_articles

        except Exception as e:
            logger.error(f"Error searching articles: {e}")
            return []

    async def generate_ai_response(self, query: str, articles: List[Dict], query_analysis: Dict) -> str:
        """Generate AI-powered response using OpenAI"""
        try:
            # Prepare context from articles
            context = self.prepare_context(articles)

            # Create system prompt based on query type
            system_prompt = self.create_system_prompt(query_analysis['type'])

            # Create user prompt
            user_prompt = f"""
            User Question: {query}

            Relevant News Articles:
            {context}

            Please provide a helpful, informative response based on the news articles above.
            Summarize the key information from the articles that relates to the user's question.
            Always try to extract relevant information from the articles provided.
            """

            # Call OpenAI
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return await self.generate_fallback_response(query, articles, query_analysis)

    def create_system_prompt(self, query_type: str) -> str:
        """Create system prompt based on query type"""
        base_prompt = "You are a helpful news assistant for FOBOH News Aggregation system. You provide accurate, informative responses about Australian news based on the provided articles."

        if query_type == 'breaking_news':
            return f"{base_prompt} Focus on urgent, time-sensitive information and breaking developments."
        elif query_type == 'summary':
            return f"{base_prompt} Provide clear, concise summaries of the main points from multiple articles."
        elif query_type == 'latest_news':
            return f"{base_prompt} Focus on the most recent developments and current events."
        elif query_type == 'question':
            return f"{base_prompt} Answer the specific question asked, providing relevant details from the news articles."
        else:
            return f"{base_prompt} Provide helpful information based on the available news articles."

    def prepare_context(self, articles: List[Dict]) -> str:
        """Prepare context from articles for AI prompt"""
        context_parts = []

        for i, article in enumerate(articles[:5], 1):  # Limit to top 5
            context_parts.append(f"""
            Article {i}:
            Title: {article['title']}
            Source: {article['source']}
            Category: {article['category']}
            Content: {article['content'][:500]}...
            Relevance: {article['relevance_score']:.2f}
            """)

        return "\n".join(context_parts)

    async def generate_fallback_response(self, query: str, articles: List[Dict], query_analysis: Dict) -> str:
        """Generate fallback response without AI"""
        if not articles:
            return f"I couldn't find any recent news articles related to '{query}'. This might be because the news database is still being populated with the latest articles. Please try asking about sports, lifestyle, music, or finance topics."

        # Generate response based on query type
        query_type = query_analysis['type']
        category = query_analysis.get('category', 'general')

        if query_type == 'breaking_news':
            breaking_articles = [a for a in articles if a.get('is_breaking_news')]
            if breaking_articles:
                article = breaking_articles[0]
                return f"🚨 Breaking News: {article['title']} - {article['content'][:200]}... (Source: {article['source']})"

        elif query_type == 'latest_news':
            if articles:
                response_parts = [f"Here are the latest {category or 'news'} updates I found:"]
                for i, article in enumerate(articles[:3], 1):
                    response_parts.append(f"{i}. {article['title']} - {article['source']}")
                return "\n".join(response_parts)

        elif query_type == 'summary':
            if articles:
                return f"Based on {len(articles)} recent articles, here's what I found: {articles[0]['title']} from {articles[0]['source']}. {articles[0]['content'][:150]}..."

        # Default response - always use articles if available
        if articles:
            best_article = articles[0]
            return f"I found relevant information about your query. Here's what {best_article['source']} reported: {best_article['title']} - {best_article['content'][:200]}... You can read more at: {best_article['url']}"

        return "I couldn't find specific information about your query in our current news database. Please try asking about recent Australian news in sports, lifestyle, music, or finance."

    async def get_breaking_news_summary(self) -> str:
        """Get a summary of breaking news"""
        try:
            breaking_articles = await self.rag_service.get_breaking_news(limit=3)

            if not breaking_articles:
                return "No breaking news at the moment. All systems are monitoring for urgent updates."

            if self.openai_client:
                context = self.prepare_context(breaking_articles)
                prompt = f"Summarize these breaking news stories in 2-3 sentences:\n{context}"

                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150
                    )
                )
                return response.choices[0].message.content.strip()
            else:
                summaries = []
                for article in breaking_articles:
                    summaries.append(f"🚨 {article['title']} ({article['source']})")
                return "Breaking News:\n" + "\n".join(summaries)

        except Exception as e:
            logger.error(f"Error getting breaking news summary: {e}")
            return "Unable to retrieve breaking news at this time."

    async def get_category_summary(self, category: str) -> str:
        """Get a summary of news in a specific category"""
        try:
            articles = await self.rag_service.search_by_category(category, limit=5)

            if not articles:
                return f"No recent {category} news found in our database."

            if self.openai_client:
                context = self.prepare_context(articles)
                prompt = f"Provide a brief summary of the latest {category} news:\n{context}"

                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200
                    )
                )
                return response.choices[0].message.content.strip()
            else:
                summaries = []
                for article in articles[:3]:
                    summaries.append(f"• {article['title']} - {article['source']}")
                return f"Latest {category} news:\n" + "\n".join(summaries)

        except Exception as e:
            logger.error(f"Error getting category summary: {e}")
            return f"Unable to retrieve {category} news at this time."

# Global chatbot instance
rag_chatbot = None

def get_rag_chatbot() -> RAGChatbotService:
    """Get global RAG chatbot instance"""
    global rag_chatbot
    if rag_chatbot is None:
        rag_chatbot = RAGChatbotService()
    return rag_chatbot 
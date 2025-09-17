from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Article(Base):
    __tablename__ = "articles"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text)
    author = Column(String(200))
    source_name = Column(String(100), nullable=False)
    article_url = Column(String(500), unique=True, nullable=False)
    category = Column(String(50), nullable=False)
    published_date = Column(DateTime)
    extracted_date = Column(DateTime, default=datetime.utcnow)
    content_hash = Column(String(32), unique=True, nullable=False)
    sentiment_score = Column(Float, default=0.0)
    priority_score = Column(Float, default=0.0)
    is_breaking_news = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<Article(title='{self.title[:50]}...', source='{self.source_name}')>" 
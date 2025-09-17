from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class NewsSource(Base):
    __tablename__ = "news_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    url = Column(String(500), nullable=False)
    rss_feed = Column(String(500))
    category = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True)
    success_rate = Column(Float, default=0.0)
    last_scraped = Column(DateTime)
    articles_extracted = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<NewsSource(name='{self.name}', category='{self.category}')>" 
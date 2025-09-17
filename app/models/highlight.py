from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Highlight(Base):
    __tablename__ = "highlights"
    
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(50), nullable=False)
    title = Column(String(500), nullable=False)
    summary = Column(Text, nullable=False)
    article_count = Column(Integer, default=1)
    priority_score = Column(Float, default=0.0)
    generated_date = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Highlight(category='{self.category}', title='{self.title[:30]}...')>" 
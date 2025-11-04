"""
ResumeGuard Database Models
SQLAlchemy ORM for storing resume analyses
"""

from sqlalchemy import create_engine, Column, String, Float, Integer, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import uuid

Base = declarative_base()


class Analysis(Base):
    """
    Stores complete resume analysis results
    """
    __tablename__ = 'analyses'
    
    # Primary identifiers
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)  # For multi-user support
    
    # Resume info
    candidate_name = Column(String(255), nullable=False)
    filename = Column(String(255))
    resume_text = Column(Text)  # Store full text for re-analysis
    
    # Scores
    trust_score = Column(Float, nullable=False, index=True)
    star_rating = Column(Float, nullable=False)
    tier = Column(String(50), nullable=False)
    verdict = Column(Text, nullable=False)
    recommended_action = Column(Text)
    
    # Analysis details
    industry = Column(String(100))
    industry_confidence = Column(Float)
    ai_probability = Column(Float)
    ai_probability_display = Column(String(50))  # "Low (5-15%)"
    
    # Metrics
    problems_count = Column(Integer, default=0)
    authenticity_count = Column(Integer, default=0)
    career_growth = Column(Integer, default=0)
    has_long_tenure = Column(Integer, default=0)  # Boolean as int
    word_count = Column(Integer)
    
    # Arrays stored as JSON
    red_flags = Column(JSON)
    authenticity_markers = Column(JSON)
    universal_questions = Column(JSON)
    
    # Metadata
    analyzed_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<Analysis(id={self.id}, candidate={self.candidate_name}, score={self.trust_score})>"
    
    def to_dict(self):
        """Convert to dictionary for JSON response"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "candidate_name": self.candidate_name,
            "filename": self.filename,
            "trust_score": self.trust_score,
            "star_rating": self.star_rating,
            "star_display": "★" * int(self.star_rating) + "☆" * (5 - int(self.star_rating)),
            "tier": self.tier,
            "verdict": self.verdict,
            "recommended_action": self.recommended_action,
            "industry": self.industry,
            "industry_confidence": self.industry_confidence,
            "ai_probability": self.ai_probability,
            "ai_probability_display": self.ai_probability_display,
            "problems_count": self.problems_count,
            "authenticity_count": self.authenticity_count,
            "career_growth": self.career_growth,
            "has_long_tenure": bool(self.has_long_tenure),
            "word_count": self.word_count,
            "red_flags": self.red_flags or [],
            "authenticity_markers": self.authenticity_markers or [],
            "universal_questions": self.universal_questions or [],
            "analyzed_at": self.analyzed_at.isoformat() if self.analyzed_at else None
        }


# Database connection setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./resumeguard.db")

# Handle Railway PostgreSQL URL format (starts with postgres:// instead of postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database - create all tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized!")


def get_db():
    """Dependency for getting DB session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Run on import (for development)
if __name__ == "__main__":
    init_db()

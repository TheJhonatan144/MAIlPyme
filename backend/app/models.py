from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from app.database import Base


class Email(Base):
    __tablename__ = "emails"

    id = Column(Integer, primary_key=True, index=True)

    gmail_id = Column(
        String,
        unique=True,
        nullable=True
    )

    sender = Column(String)
    subject = Column(String)
    body = Column(Text)
    predicted_category = Column(String)
    confidence = Column(Float)
    processing_time_ms = Column(Float)
    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )
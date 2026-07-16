from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.database import Base


class Email(Base):
    __tablename__ = "emails"

    id = Column(Integer, primary_key=True, index=True)
    sender = Column(String(255), nullable=False)
    subject = Column(String(255), nullable=False)
    body = Column(Text, nullable=False)
    predicted_category = Column(String(50), nullable=False)
    confidence = Column(String(20), nullable=False, default="temporal")
    created_at = Column(DateTime, default=datetime.utcnow)
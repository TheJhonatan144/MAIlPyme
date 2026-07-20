from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


Category = Literal[
    "Contratos",
    "Facturas",
    "Colaboraciones",
    "Clientes",
    "Publicidad",
    "Varios",
]


class EmailCreate(BaseModel):
    gmail_id: str | None = None
    sender: str = Field(..., min_length=3, max_length=255)
    subject: str = Field(..., min_length=1, max_length=255)
    body: str = Field(..., min_length=10)


class EmailResponse(BaseModel):
    id: int
    sender: str
    subject: str
    body: str
    predicted_category: Category
    confidence: float
    processing_time_ms: float
    created_at: datetime | None = None

    model_config = {
        "from_attributes": True
    }
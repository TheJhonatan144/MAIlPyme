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
    sender: str = Field(..., min_length=3, max_length=255)
    subject: str = Field(..., min_length=1, max_length=255)
    body: str = Field(..., min_length=10)


class EmailResponse(BaseModel):
    id: int
    sender: str
    subject: str
    body: str
    predicted_category: Category
    confidence: str
    created_at: datetime

    model_config = {
        "from_attributes": True
    }
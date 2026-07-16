from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app import models
from app.database import get_db

router = APIRouter(
    prefix="/metrics",
    tags=["Metrics"],
)


@router.get("/summary")
def get_metrics_summary(db: Session = Depends(get_db)):
    total_emails = db.query(models.Email).count()

    category_rows = (
        db.query(
            models.Email.predicted_category,
            func.count(models.Email.id),
        )
        .group_by(models.Email.predicted_category)
        .all()
    )

    by_category = {
        category: count
        for category, count in category_rows
    }

    return {
        "total_emails": total_emails,
        "by_category": by_category,
    }
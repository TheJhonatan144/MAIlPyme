from fastapi import APIRouter, Depends
from sqlalchemy import func, desc
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


    average_processing_time = db.query(
        func.avg(models.Email.processing_time_ms)
    ).scalar()


    min_processing_time = db.query(
        func.min(models.Email.processing_time_ms)
    ).scalar()


    max_processing_time = db.query(
        func.max(models.Email.processing_time_ms)
    ).scalar()


    average_confidence = db.query(
        func.avg(models.Email.confidence)
    ).scalar()


    last_email = (
        db.query(models.Email)
        .order_by(desc(models.Email.created_at))
        .first()
    )


    return {

        "total_emails": total_emails,

        "average_processing_time_ms": round(
            average_processing_time or 0,
            2
        ),

        "min_processing_time_ms": round(
            min_processing_time or 0,
            2
        ),

        "max_processing_time_ms": round(
            max_processing_time or 0,
            2
        ),

        "average_confidence": round(
            average_confidence or 0,
            4
        ),

        "by_category": by_category,


        "last_processed_email": (
            {
                "id": last_email.id,
                "category": last_email.predicted_category,
                "confidence": last_email.confidence,
                "processing_time_ms": last_email.processing_time_ms,
                "created_at": last_email.created_at,
            }
            if last_email
            else None
        )
    }
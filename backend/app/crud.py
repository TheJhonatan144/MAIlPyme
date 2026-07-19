from time import perf_counter
from sqlalchemy.orm import Session

from app import models, schemas
from app.classifier import classify_email



def create_classified_email(db: Session, email: schemas.EmailCreate) -> models.Email:
    start_time = perf_counter()

    predicted_category, confidence = classify_email(
        sender=email.sender,
        subject=email.subject,
        body=email.body,
    )

    end_time = perf_counter()
    processing_time_ms = round((end_time - start_time) * 1000, 2)

    db_email = models.Email(
        sender=email.sender,
        subject=email.subject,
        body=email.body,
        predicted_category=predicted_category,
        confidence=confidence,
        processing_time_ms=processing_time_ms,
    )

    db.add(db_email)
    db.commit()
    db.refresh(db_email)

    return db_email


def get_emails(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Email).offset(skip).limit(limit).all()


def get_email_by_id(db: Session, email_id: int):
    return db.query(models.Email).filter(models.Email.id == email_id).first()

def delete_email(db: Session, email_id: int) -> bool:
    db_email = get_email_by_id(db=db, email_id=email_id)

    if db_email is None:
        return False

    db.delete(db_email)
    db.commit()

    return True
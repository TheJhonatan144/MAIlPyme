from sqlalchemy.orm import Session

from app import models, schemas
from app.classifier import classify_email


def create_classified_email(db: Session, email: schemas.EmailCreate) -> models.Email:
    predicted_category, confidence = classify_email(
        subject=email.subject,
        body=email.body,
    )

    db_email = models.Email(
        sender=email.sender,
        subject=email.subject,
        body=email.body,
        predicted_category=predicted_category,
        confidence=confidence,
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
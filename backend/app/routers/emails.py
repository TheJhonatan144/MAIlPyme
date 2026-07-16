from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import get_db

router = APIRouter(
    prefix="/emails",
    tags=["Emails"],
)


@router.post("/classify", response_model=schemas.EmailResponse)
def classify_and_save_email(
    email: schemas.EmailCreate,
    db: Session = Depends(get_db),
):
    return crud.create_classified_email(db=db, email=email)


@router.get("/", response_model=List[schemas.EmailResponse])
def list_emails(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    return crud.get_emails(db=db, skip=skip, limit=limit)


@router.get("/{email_id}", response_model=schemas.EmailResponse)
def get_email(
    email_id: int,
    db: Session = Depends(get_db),
):
    db_email = crud.get_email_by_id(db=db, email_id=email_id)

    if db_email is None:
        raise HTTPException(status_code=404, detail="Correo no encontrado")

    return db_email


@router.delete("/{email_id}")
def delete_email(
    email_id: int,
    db: Session = Depends(get_db),
):
    was_deleted = crud.delete_email(db=db, email_id=email_id)

    if not was_deleted:
        raise HTTPException(status_code=404, detail="Correo no encontrado")

    return {
        "message": "Correo eliminado correctamente",
        "email_id": email_id,
    }
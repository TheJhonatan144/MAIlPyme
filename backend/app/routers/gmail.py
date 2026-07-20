from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.database import get_db
from app import crud, schemas
from app.integrations.gmail_client import get_gmail_service

from googleapiclient.errors import HttpError


router = APIRouter(
    prefix="/gmail",
    tags=["Gmail"],
)


@router.post("/classify")
def classify_gmail_emails(
    db: Session = Depends(get_db),
):

    try:
        service = get_gmail_service()

    except Exception as e:
        raise HTTPException(
        status_code=500,
        detail=f"No se pudo conectar con Gmail: {str(e)}"
    )

    try:
        results = service.users().messages().list(
            userId="me",
            maxResults=10
        ).execute()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error consultando Gmail: {str(e)}"
        )

    messages = results.get(
        "messages",
        []
    )

    classified = []


    for message in messages:

        try:

            email = service.users().messages().get(
                userId="me",
                id=message["id"],
                format="full"
            ).execute()

        except Exception as e:

            print(
                f"Error leyendo correo {message['id']}: {e}"
            )

            continue


        headers = email["payload"]["headers"]

        sender = ""
        subject = ""

        for header in headers:

            if header["name"] == "From":
                sender = header["value"]

            if header["name"] == "Subject":
                subject = header["value"]


        body = ""

        payload = email["payload"]

        if "parts" in payload:

            part = payload["parts"][0]

            if part["body"].get("data"):
                import base64

                body = base64.urlsafe_b64decode(
                    part["body"]["data"]
                ).decode(
                    "utf-8",
                    errors="ignore"
                )


        email_create = schemas.EmailCreate(
            sender=sender,
            subject=subject,
            body=body,
            gmail_id=message["id"]
        )


        try:
            existing_email = crud.get_email_by_gmail_id(
                db=db,
                gmail_id=message["id"]
            )

            if existing_email:
                saved_email = existing_email

            else:
                existing_email = crud.get_email_by_gmail_id(
                    db=db,
                    gmail_id=message["id"]
                )

                if existing_email:
                    saved_email = existing_email

                else:
                    saved_email = crud.create_classified_email(
                        db=db,
                        email=email_create,
                    )

        except Exception as e:

            print(
                f"Error clasificando correo: {e}"
            )

            continue


        classified.append(
            {
                "id": saved_email.id,
                "subject": saved_email.subject,
                "category": saved_email.predicted_category,
                "confidence": saved_email.confidence,
            }
        )


    return {
        "processed": len(classified),
        "emails": classified,
    }
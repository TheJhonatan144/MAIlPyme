from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

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

    service = get_gmail_service()

    results = service.users().messages().list(
        userId="me",
        maxResults=10
    ).execute()

    messages = results.get(
        "messages",
        []
    )

    classified = []


    for message in messages:

        email = service.users().messages().get(
            userId="me",
            id=message["id"],
            format="full"
        ).execute()


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
        )


        saved_email = crud.create_classified_email(
            db=db,
            email=email_create,
        )


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
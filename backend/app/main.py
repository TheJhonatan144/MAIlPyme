from fastapi import FastAPI

from app import models
from app.database import engine
from app.routers import emails

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="MailPyme API",
    description="API backend para clasificación de correos empresariales de MiPYMEs ecuatorianas.",
    version="0.2.0",
)

app.include_router(emails.router)


@app.get("/")
def root():
    return {
        "message": "Bienvenido a MailPyme API",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "MailPyme API",
        "database": "connected",
    }
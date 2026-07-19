from fastapi import FastAPI


from app import models
from app.database import engine
from app.routers import categories, emails, metrics, model
from app.model_loader import load_model
from app.routers import gmail

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="MailPyme API",
    description="API backend para clasificación de correos empresariales de MiPYMEs ecuatorianas.",
    version="0.2.0",
)

@app.on_event("startup")
def startup_event():
    load_model()

app.include_router(model.router)
app.include_router(emails.router)
app.include_router(metrics.router)
app.include_router(categories.router)
app.include_router(gmail.router)

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
from fastapi import FastAPI

app = FastAPI(
    title="MailPyme API",
    description="API backend para clasificación de correos empresariales de MiPYMEs ecuatorianas.",
    version="0.1.0",
)


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
    }
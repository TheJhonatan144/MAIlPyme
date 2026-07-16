# Backend - MailPyme API

Backend del proyecto **MailPyme AI**, desarrollado con FastAPI para clasificar correos electrónicos empresariales de MiPYMEs ecuatorianas.

Este backend permite recibir correos, clasificarlos en una categoría empresarial, guardarlos en una base de datos SQLite y exponer métricas básicas para el dashboard.

## Tecnologías usadas

- Python
- FastAPI
- Uvicorn
- SQLAlchemy
- SQLite
- Pydantic

## Estructura del backend

```text
backend/
├── app/
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   ├── crud.py
│   ├── classifier.py
│   └── routers/
│       ├── emails.py
│       └── metrics.py
├── tests/
├── requirements.txt
├── .env.example
└── README.md

Cómo levantar el backend:
python -m uvicorn app.main:app --reload

Dónde ver Swagger:
http://127.0.0.1:8000/docs

Endpoint para clasificar:
POST /emails/classify

Endpoint para métricas:
GET /metrics/summary

Categorías válidas:
Contratos, Facturas, Colaboraciones, Clientes, Publicidad, Varios


El clasificador actual es temporal.
Luego será reemplazado por BETO.
La base mailpyme.db no se sube a GitHub.
El entorno .venv no se sube a GitHub.

GET /health

POST /emails/classify

body

{
  "sender": "compras@empresa.com",
  "subject": "Consulta sobre factura pendiente",
  "body": "Buenas tardes, necesitamos confirmar si la factura del último pedido ya fue emitida."
}

respuesta
{
  "id": 1,
  "sender": "compras@empresa.com",
  "subject": "Consulta sobre factura pendiente",
  "body": "Buenas tardes, necesitamos confirmar si la factura del último pedido ya fue emitida.",
  "predicted_category": "Facturas",
  "confidence": "temporal",
  "created_at": "2026-07-15T20:00:00"
}
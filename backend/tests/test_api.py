from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_get_categories():
    response = client.get("/categories/")

    assert response.status_code == 200
    data = response.json()

    assert "categories" in data
    assert "Contratos" in data["categories"]
    assert "Facturas" in data["categories"]
    assert "Varios" in data["categories"]


def test_classify_email():
    payload = {
        "sender": "compras@empresa.com",
        "subject": "Factura pendiente",
        "body": "Buenas tardes, necesitamos confirmar el pago de la factura emitida.",
    }

    response = client.post("/emails/classify", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["sender"] == payload["sender"]
    assert data["predicted_category"] in [
        "Contratos",
        "Facturas",
        "Colaboraciones",
        "Clientes",
        "Publicidad",
        "Varios",
    ]
    assert "processing_time_ms" in data


def test_metrics_summary():
    response = client.get("/metrics/summary")

    assert response.status_code == 200
    data = response.json()

    assert "total_emails" in data
    assert "by_category" in data
    assert "average_processing_time_ms" in data
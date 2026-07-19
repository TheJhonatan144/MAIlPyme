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


def test_classify_factura_email():
    payload = {
        "sender": "compras@empresa.com",
        "subject": "Factura pendiente de pago",
        "body": "Buenas tardes, adjuntamos la factura emitida correspondiente al servicio contratado.",
    }

    response = client.post("/emails/classify", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["sender"] == payload["sender"]
    assert data["predicted_category"] == "Facturas"
    assert float(data["confidence"]) > 0.5
    assert data["processing_time_ms"] > 0
    
    
def test_classify_contrato_email():
    payload = {
        "sender": "legal@empresa.com",
        "subject": "Contrato comercial actualizado",
        "body": "Adjuntamos el contrato actualizado para revisión y firma.",
    }

    response = client.post("/emails/classify", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_category"] == "Contratos"
    assert float(data["confidence"]) > 0.5


def test_metrics_summary():
    response = client.get("/metrics/summary")

    assert response.status_code == 200
    data = response.json()

    assert "total_emails" in data
    assert "by_category" in data
    assert "average_processing_time_ms" in data
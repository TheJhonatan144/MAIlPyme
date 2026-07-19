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
    
def test_model_info():

    response = client.get("/model/info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_name"] == "MailPyme BETO v2"
    assert data["base_model"] == "dccuchile/bert-base-spanish-wwm-cased"
    assert data["architecture"] == "BERT Sequence Classification"
    assert data["categories"] == 6
    assert data["max_length"] == 128

    assert "Contratos" in data["labels"]
    assert "Facturas" in data["labels"]
    assert "Varios" in data["labels"]


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

    assert "min_processing_time_ms" in data
    assert "max_processing_time_ms" in data
    assert "average_confidence" in data
    assert "last_processed_email" in data

    assert data["total_emails"] >= 0
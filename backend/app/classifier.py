from app.schemas import Category


def classify_email(subject: str, body: str) -> tuple[Category, str]:
    """
    Clasificador temporal para probar el flujo backend + base de datos.

    Luego será reemplazado por el modelo BETO entrenado.
    """
    text = f"{subject} {body}".lower()

    if any(word in text for word in ["contrato", "convenio", "acuerdo", "cláusula", "firma"]):
        return "Contratos", "temporal"

    if any(word in text for word in ["factura", "pago", "comprobante", "ruc", "valor pendiente"]):
        return "Facturas", "temporal"

    if any(word in text for word in ["alianza", "colaboración", "propuesta conjunta", "auspicio"]):
        return "Colaboraciones", "temporal"

    if any(word in text for word in ["cliente", "pedido", "reclamo", "soporte", "consulta"]):
        return "Clientes", "temporal"

    if any(word in text for word in ["promoción", "descuento", "oferta", "campaña", "publicidad"]):
        return "Publicidad", "temporal"

    return "Varios", "temporal"
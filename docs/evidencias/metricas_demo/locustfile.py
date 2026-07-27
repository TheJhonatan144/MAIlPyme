from locust import HttpUser, task, between
import random

# Correos de prueba variados, uno por categoría
CORREOS = [
    ("cliente@empresa.com", "Solicitud de cotizacion",
     "Buenas tardes, quisiera conocer el precio del servicio mensual para mi oficina en Quito."),
    ("facturacion@proveedor.com", "Factura pendiente de pago",
     "Adjuntamos la factura correspondiente al mes de junio con el valor pendiente de cancelacion."),
    ("legal@socio.com", "Contrato para revision",
     "Enviamos el contrato de prestacion de servicios para su revision y posterior firma."),
    ("marketing@ofertas.com", "Aprovecha 50% de descuento",
     "Promocion exclusiva en software empresarial valida solo por esta semana, no la dejes pasar."),
    ("alianza@marca.com", "Propuesta de colaboracion comercial",
     "Nos gustaria proponer una alianza comercial conjunta entre nuestras dos empresas."),
    ("info@camara.com", "Invitacion a charla informativa",
     "Se convoca a la reunion de la camara de comercio el proximo jueves en la tarde."),
]


class UsuarioMailPyme(HttpUser):
    # Cada usuario espera entre 1 y 3 segundos entre peticiones (simula uso real)
    wait_time = between(1, 3)

    @task
    def clasificar_correo(self):
        sender, subject, body = random.choice(CORREOS)
        self.client.post(
            "/emails/classify",
            json={"sender": sender, "subject": subject, "body": body},
        )

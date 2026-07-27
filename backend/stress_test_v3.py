"""
Stress test del modelo BETO v3 - MailPyme AI
Envia 18 correos dificiles al backend, recoge las predicciones y guarda un CSV.

COMO USAR:
1. Asegurate de que el BACKEND este corriendo (uvicorn en http://127.0.0.1:8000).
2. Coloca este archivo en:  ENSAYO_FINAL\backend\
3. Con el venv del backend activo, ejecuta:
       python stress_test_v3.py
4. Se genera el archivo:  stress_test_v3_resultados.csv
   y ademas se imprime un resumen en pantalla.
"""

import csv
import requests

# URL del backend (cambia el puerto si usas otro)
API_URL = "http://127.0.0.1:8000/emails/classify"

# Los 18 correos: (grupo, esperado, sender, subject, body)
CORREOS = [
    ("A-Frontera", "Contratos/Facturas", "admin@constructora.com",
     "Anticipo segun contrato",
     "Adjunto la factura del anticipo correspondiente a la clausula cuarta del contrato firmado el mes pasado."),

    ("A-Frontera", "Colab/Clientes", "gerencia@imprentaquito.com",
     "Trabajar juntos y cotizar",
     "Somos una imprenta en Ambato, nos gustaria explorar si podemos trabajar juntos y de paso cotizar sus servicios de diseno."),

    ("A-Frontera", "Publicidad", "no-reply@promos.com",
     "Su cuenta requiere atencion",
     "Estimado cliente, su cuenta necesita atencion inmediata. Renueve hoy su plan con 40 por ciento de descuento exclusivo."),

    ("A-Frontera", "Colab/Publicidad", "alianzas@marketing.com",
     "Campana conjunta con descuentos",
     "Proponemos una campana de marketing conjunta donde ambas empresas ofrezcan promociones y ofertas a sus clientes."),

    ("A-Frontera", "Contratos/Colab", "legal@empresa.com",
     "Convenio de alianza para firma",
     "Enviamos el convenio de colaboracion estrategica para su revision y firma por ambas partes interesadas."),

    ("A-Frontera", "Facturas/Clientes", "cliente@negocio.com",
     "Consulta sobre mi factura",
     "Buenas tardes, tengo una consulta sobre el valor de la factura que me enviaron, creo que hay un error en el monto."),

    ("B-SinSenales", "Varios/Clientes", "juan@correo.com",
     "Sobre lo de ayer",
     "Buenos dias, respecto a lo que conversamos ayer en la reunion, quisiera saber si seguimos adelante con lo acordado."),

    ("B-SinSenales", "Varios/Clientes", "info@empresa.com",
     "Seguimiento",
     "Hola, escribo para dar seguimiento al tema pendiente que quedamos de revisar la semana pasada. Quedo atento a su respuesta."),

    ("B-SinSenales", "Varios/Clientes", "contacto@web.com",
     "Informacion",
     "Quisiera recibir mas informacion por favor, gracias de antemano por su tiempo y atencion."),

    ("C-Ruido", "Baja confianza", "test@test.com",
     "asdkjh qwerty",
     "asdkjhaskjdh qwerty jajaja xd no se que escribir aqui lorem ipsum banana teclado random."),

    ("C-Ruido", "Baja confianza", "random@mail.com",
     "xxxxxx yyyyyy",
     "zxcvbnm asdfghjkl qwertyuiop mnbvcxz poiuytrewq lkjhgfdsa random texto sin proposito alguno."),

    ("C-Ruido", "Baja confianza", "user@chat.com",
     "hola q tal",
     "holaaa q mas como va todo por ahi jajaja bueno nada solo saludar un abrazo nos vemos luego chao."),

    ("D-Varios", "Varios", "eventos@camara.com",
     "Invitacion a charla",
     "Se convoca a la charla informativa sobre emprendimiento el proximo jueves a las tres de la tarde en el auditorio."),

    ("D-Varios", "Varios", "rrhh@interno.com",
     "Recordatorio reunion",
     "Recordamos a todo el personal que la reunion mensual sera el viernes. Por favor confirmar asistencia con anticipacion."),

    ("E-Control", "Facturas", "cobranzas@proveedor.com",
     "Factura 001-234 vencida",
     "Le recordamos que la factura numero 001-234 se encuentra vencida. Favor realizar el pago del valor pendiente."),

    ("E-Control", "Contratos", "juridico@bufete.com",
     "Contrato de arrendamiento",
     "Adjuntamos el contrato de arrendamiento con las clausulas y terminos para su revision, firma y posterior devolucion."),

    ("E-Control", "Publicidad", "ofertas@tienda.com",
     "Mega descuentos esta semana",
     "Aprovecha nuestras ofertas exclusivas con hasta 70 por ciento de descuento en toda la tienda solo por esta semana."),

    ("E-Control", "Clientes", "comprador@empresa.com",
     "Cotizacion de servicio",
     "Buenas tardes, necesito una cotizacion para el servicio de soporte tecnico mensual para una oficina de diez personas."),
]

UMBRAL = 0.75  # umbral oficial del v3


def main():
    print("=" * 70)
    print("STRESS TEST BETO v3 - 18 correos dificiles")
    print("=" * 70)
    print(f"Backend: {API_URL}")
    print(f"Umbral de revision: {UMBRAL}\n")

    # Verificar que el backend responde antes de empezar
    try:
        requests.get("http://127.0.0.1:8000/health", timeout=5)
    except requests.exceptions.RequestException:
        print("ERROR: El backend no responde en http://127.0.0.1:8000")
        print("Asegurate de que uvicorn este corriendo antes de ejecutar esto.")
        return

    resultados = []
    n_baja_confianza = 0

    for i, (grupo, esperado, sender, subject, body) in enumerate(CORREOS, start=1):
        payload = {"sender": sender, "subject": subject, "body": body}
        try:
            r = requests.post(API_URL, json=payload, timeout=15)
            r.raise_for_status()
            data = r.json()
            categoria = data.get("predicted_category", "ERROR")
            confianza = float(data.get("confidence", 0))
            latencia = data.get("processing_time_ms", 0)
        except requests.exceptions.RequestException as e:
            categoria = "ERROR_CONEXION"
            confianza = 0.0
            latencia = 0

        # Estado segun umbral
        if confianza < UMBRAL:
            estado = "REVISION MANUAL"
            n_baja_confianza += 1
        else:
            estado = "Clasificado"

        resultados.append({
            "n": i,
            "grupo": grupo,
            "esperado": esperado,
            "predicho": categoria,
            "confianza": round(confianza, 4),
            "estado": estado,
            "latencia_ms": latencia,
            "asunto": subject,
        })

        # Imprimir en vivo
        marca = "  <-- revision" if estado == "REVISION MANUAL" else ""
        print(f"{i:2d}. [{grupo:12s}] {categoria:14s} conf={confianza:.4f}{marca}")

    # Guardar CSV
    salida = "stress_test_v3_resultados.csv"
    with open(salida, "w", newline="", encoding="utf-8-sig") as f:
        campos = ["n", "grupo", "esperado", "predicho", "confianza",
                  "estado", "latencia_ms", "asunto"]
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Total de correos probados: {len(resultados)}")
    print(f"Marcados para revision manual (conf < {UMBRAL}): {n_baja_confianza}")
    print(f"Resultados guardados en: {salida}")
    print("\nAbre ese CSV en Excel para ver la tabla completa.")


if __name__ == "__main__":
    main()

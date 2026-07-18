# MAIlPyme AI

Clasificador inteligente de correos empresariales para MiPYMEs ecuatorianas.

## Objetivo

Desarrollar un MVP de aplicación web que lea, clasifique y organice correos electrónicos entrantes de una PYME ecuatoriana mediante una arquitectura basada en BERT/BETO.

## Categorías oficiales

1. Contratos
2. Facturas
3. Colaboraciones
4. Clientes
5. Publicidad
6. Varios / Extras

## Alcance del MVP

El sistema permite:

- Leer correos desde CSV, formulario o cuenta de prueba.
- Extraer asunto, remitente, fecha y cuerpo.
- Clasificar correos usando BERT/BETO.
- Mostrar categoría predicha y confianza del modelo.
- Guardar resultados en base de datos.
- Visualizar resultados en dashboard web.
- Presentar métricas técnicas y logs.

El sistema no permite:

- Responder correos automáticamente.
- Eliminar correos.
- Validar legalmente contratos.
- Validar facturas con el SRI.
- Procesar pagos.
- Modificar de forma irreversible una bandeja real de Gmail u Outlook.

## Estructura del proyecto

```text
MAIlPyme/
├── backend/
│   ├── app/
│   ├── models/
│   ├── database/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   └── README.md
├── notebooks/
│   └── beto_email_classifier.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── docs/
│   ├── arquitectura.md
│   ├── modelo_negocio.md
│   ├── metricas.md
│   └── riesgos.md
├── presentation/
│   └── ppt_final/
└── README.md
```

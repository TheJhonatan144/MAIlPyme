# Frontend - MailPyme AI

Dashboard web para visualizar la clasificación automática de correos electrónicos empresariales de MiPYMEs ecuatorianas.

## Tecnología utilizada

- Python
- Streamlit
- Pandas
- Plotly

## Funcionalidades principales

- Visualización de correos clasificados.
- Vista simplificada para usuario final.
- Vista técnica protegida mediante contraseña.
- Filtro por categoría.
- Búsqueda por asunto, remitente o contenido.
- Métricas generales del sistema para el equipo técnico.
- Gráficos de distribución por categoría.
- Gráfico de confianza por correo.
- Vista detallada de cada correo.
- Preparado para conectarse posteriormente con la API del backend.

## Categorías oficiales

- Contratos
- Facturas
- Colaboraciones
- Clientes
- Publicidad
- Varios

## Integración con backend

El dashboard está preparado para consumir datos desde la API del backend.

Por defecto intenta conectarse a:

```text
http://127.0.0.1:8000

## Estructura del frontend

```text
frontend/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── correos_demo.csv
│
├── assets/
│   └── .gitkeep
│
└── .streamlit/
    ├── secrets.example.toml
    └── secrets.toml
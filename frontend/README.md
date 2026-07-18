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
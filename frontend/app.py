import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path


# =========================
# Configuración de la página
# =========================
st.set_page_config(
    page_title="MailPyme AI",
    page_icon="📬",
    layout="wide"
)


# =========================
# Carga de datos
# =========================
@st.cache_data
def cargar_datos():
    ruta = Path(__file__).parent / "data" / "correos_demo.csv"
    return pd.read_csv(ruta)


df = cargar_datos()


# =========================
# Encabezado
# =========================
st.title("📬 MailPyme AI")
st.caption("Dashboard para clasificación automática de correos empresariales usando BETO/BERT en español.")

st.markdown(
    """
    Este panel permite visualizar correos electrónicos clasificados automáticamente en categorías empresariales:
    **Contratos, Facturas, Colaboraciones, Clientes, Publicidad y Varios**.
    """
)


# =========================
# Barra lateral de filtros
# =========================
st.sidebar.header("Filtros")

categorias = ["Todas"] + sorted(df["categoria"].unique().tolist())
categoria_seleccionada = st.sidebar.selectbox("Categoría", categorias)

confianza_minima = st.sidebar.slider(
    "Confianza mínima",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05
)

busqueda = st.sidebar.text_input("Buscar por asunto, remitente o contenido")


# =========================
# Aplicación de filtros
# =========================
df_filtrado = df.copy()

if categoria_seleccionada != "Todas":
    df_filtrado = df_filtrado[df_filtrado["categoria"] == categoria_seleccionada]

df_filtrado = df_filtrado[df_filtrado["confianza"] >= confianza_minima]

if busqueda:
    busqueda_lower = busqueda.lower()
    df_filtrado = df_filtrado[
        df_filtrado["asunto"].str.lower().str.contains(busqueda_lower)
        | df_filtrado["remitente"].str.lower().str.contains(busqueda_lower)
        | df_filtrado["cuerpo"].str.lower().str.contains(busqueda_lower)
    ]


# =========================
# Métricas principales
# =========================
total_correos = len(df_filtrado)
confianza_promedio = df_filtrado["confianza"].mean() if total_correos > 0 else 0
categoria_mas_frecuente = (
    df_filtrado["categoria"].mode()[0] if total_correos > 0 else "Sin datos"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Correos clasificados", total_correos)

with col2:
    st.metric("Confianza promedio", f"{confianza_promedio:.2%}")

with col3:
    st.metric("Categoría más frecuente", categoria_mas_frecuente)


st.divider()


# =========================
# Gráficos
# =========================
col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    st.subheader("Distribución por categoría")

    if total_correos > 0:
        conteo_categorias = df_filtrado["categoria"].value_counts().reset_index()
        conteo_categorias.columns = ["categoria", "cantidad"]

        fig_categorias = px.bar(
            conteo_categorias,
            x="categoria",
            y="cantidad",
            text="cantidad",
            title="Cantidad de correos por categoría"
        )
        st.plotly_chart(fig_categorias, use_container_width=True)
    else:
        st.info("No hay datos para mostrar con los filtros actuales.")

with col_graf2:
    st.subheader("Confianza por correo")

    if total_correos > 0:
        fig_confianza = px.scatter(
            df_filtrado,
            x="fecha",
            y="confianza",
            color="categoria",
            hover_data=["asunto", "remitente"],
            title="Nivel de confianza de clasificación"
        )
        st.plotly_chart(fig_confianza, use_container_width=True)
    else:
        st.info("No hay datos para mostrar con los filtros actuales.")


st.divider()


# =========================
# Tabla de correos
# =========================
st.subheader("Correos clasificados")

columnas_tabla = [
    "fecha",
    "remitente",
    "asunto",
    "categoria",
    "confianza",
    "estado"
]

st.dataframe(
    df_filtrado[columnas_tabla],
    use_container_width=True,
    hide_index=True
)


# =========================
# Vista de detalle
# =========================
st.subheader("Detalle del correo")

if total_correos > 0:
    opciones = df_filtrado["id"].astype(str) + " - " + df_filtrado["asunto"]

    correo_seleccionado = st.selectbox(
        "Selecciona un correo para ver el detalle",
        opciones
    )

    id_correo = int(correo_seleccionado.split(" - ")[0])
    correo = df_filtrado[df_filtrado["id"] == id_correo].iloc[0]

    col_det1, col_det2 = st.columns([2, 1])

    with col_det1:
        st.markdown(f"**Asunto:** {correo['asunto']}")
        st.markdown(f"**Remitente:** {correo['remitente']}")
        st.markdown(f"**Fecha:** {correo['fecha']}")
        st.markdown("**Contenido:**")
        st.info(correo["cuerpo"])

    with col_det2:
        st.markdown("### Resultado del modelo")
        st.success(f"Categoría: {correo['categoria']}")
        st.metric("Confianza", f"{correo['confianza']:.2%}")
        st.markdown(f"**Estado:** {correo['estado']}")
else:
    st.warning("No hay correos disponibles para mostrar detalle.")


# =========================
# Nota técnica
# =========================
st.divider()

st.caption(
    "Versión demo: actualmente los datos se cargan desde un archivo CSV. "
    "En la siguiente fase, el dashboard consumirá los resultados desde la API del backend."
)
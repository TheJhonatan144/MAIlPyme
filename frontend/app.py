from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================
# Configuración de página
# =========================
st.set_page_config(
    page_title="MailPyme AI",
    page_icon="",
    layout="wide"
)


# =========================
# Carga de datos
# =========================
@st.cache_data
def cargar_datos():
    ruta = Path(__file__).parent / "data" / "correos_demo.csv"
    return pd.read_csv(ruta)


# =========================
# Acceso técnico
# =========================
def verificar_acceso_tecnico():
    if "mostrar_login_tecnico" not in st.session_state:
        st.session_state.mostrar_login_tecnico = False

    if "modo_tecnico" not in st.session_state:
        st.session_state.modo_tecnico = False

    st.sidebar.markdown("---")

    if st.session_state.modo_tecnico:
        st.sidebar.success("Vista técnica habilitada")

        if st.sidebar.button("Volver a vista usuario final"):
            st.session_state.modo_tecnico = False
            st.session_state.mostrar_login_tecnico = False
            st.rerun()

        return True

    st.sidebar.info("Vista actual: usuario final")

    if not st.session_state.mostrar_login_tecnico:
        if st.sidebar.button("Acceso técnico"):
            st.session_state.mostrar_login_tecnico = True
            st.rerun()

        return False

    st.sidebar.subheader("Acceso técnico")

    clave_ingresada = st.sidebar.text_input(
        "Contraseña técnica",
        type="password",
        placeholder="Ingrese la clave técnica"
    )

    col_login, col_cancelar = st.sidebar.columns(2)

    with col_login:
        ingresar = st.button("Ingresar")

    with col_cancelar:
        cancelar = st.button("Cancelar")

    if cancelar:
        st.session_state.mostrar_login_tecnico = False
        st.rerun()

    if ingresar:
        clave_tecnica = st.secrets.get("TECH_PASSWORD", "admin123")

        if clave_ingresada == clave_tecnica:
            st.session_state.modo_tecnico = True
            st.session_state.mostrar_login_tecnico = False
            st.rerun()
        else:
            st.sidebar.error("Contraseña incorrecta")

    return False


# =========================
# Filtros
# =========================
def aplicar_filtros(df, modo_tecnico=False):
    st.sidebar.header("Filtros")

    categorias = ["Todas"] + sorted(df["categoria"].unique().tolist())
    categoria_seleccionada = st.sidebar.selectbox("Categoría", categorias)

    busqueda = st.sidebar.text_input(
        "Buscar por asunto, remitente o contenido"
    )

    df_filtrado = df.copy()

    if categoria_seleccionada != "Todas":
        df_filtrado = df_filtrado[
            df_filtrado["categoria"] == categoria_seleccionada
        ]

    if busqueda:
        busqueda_lower = busqueda.lower()
        df_filtrado = df_filtrado[
            df_filtrado["asunto"].str.lower().str.contains(busqueda_lower)
            | df_filtrado["remitente"].str.lower().str.contains(busqueda_lower)
            | df_filtrado["cuerpo"].str.lower().str.contains(busqueda_lower)
        ]

    if modo_tecnico:
        confianza_minima = st.sidebar.slider(
            "Confianza mínima",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )

        df_filtrado = df_filtrado[
            df_filtrado["confianza"] >= confianza_minima
        ]

    return df_filtrado


# =========================
# Vista usuario final
# =========================
def mostrar_vista_usuario(df_filtrado):
    st.title("MailPyme AI")
    st.caption("Bandeja de correos clasificados automáticamente.")

    st.markdown(
        """
        En esta vista puedes revisar tus correos organizados por categoría.
        El objetivo es ayudarte a identificar rápidamente si un mensaje corresponde a
        **Contratos, Facturas, Colaboraciones, Clientes, Publicidad o Varios**.
        """
    )

    st.divider()

    total_correos = len(df_filtrado)

    st.metric("Correos encontrados", total_correos)

    st.subheader("Mis correos")

    columnas_usuario = [
        "fecha",
        "remitente",
        "asunto",
        "categoria"
    ]

    if total_correos > 0:
        st.dataframe(
            df_filtrado[columnas_usuario],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No hay correos disponibles con los filtros actuales.")
        return

    st.divider()

    st.subheader("Detalle del correo")

    opciones = df_filtrado["id"].astype(str) + " - " + df_filtrado["asunto"]

    correo_seleccionado = st.selectbox(
        "Selecciona un correo para ver el contenido",
        opciones
    )

    id_correo = int(correo_seleccionado.split(" - ")[0])
    correo = df_filtrado[df_filtrado["id"] == id_correo].iloc[0]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"**Asunto:** {correo['asunto']}")
        st.markdown(f"**Remitente:** {correo['remitente']}")
        st.markdown(f"**Fecha:** {correo['fecha']}")
        st.markdown("**Contenido:**")
        st.info(correo["cuerpo"])

    with col2:
        st.markdown("### Clasificación")
        st.success(correo["categoria"])

    st.caption(
        "Vista de usuario final: se ocultan métricas técnicas, confianza del modelo y gráficos internos."
    )


# =========================
# Vista desarrollador/modelo
# =========================
def mostrar_vista_tecnica(df_filtrado):
    st.title("📊 Panel técnico del modelo")
    st.caption(
        "Vista para revisar métricas, confianza y comportamiento de la clasificación."
    )

    total_correos = len(df_filtrado)
    confianza_promedio = (
        df_filtrado["confianza"].mean() if total_correos > 0 else 0
    )
    categoria_mas_frecuente = (
        df_filtrado["categoria"].mode()[0]
        if total_correos > 0
        else "Sin datos"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Correos clasificados", total_correos)

    with col2:
        st.metric("Confianza promedio", f"{confianza_promedio:.2%}")

    with col3:
        st.metric("Categoría más frecuente", categoria_mas_frecuente)

    st.divider()

    col_graf1, col_graf2 = st.columns(2)

    with col_graf1:
        st.subheader("Distribución por categoría")

        if total_correos > 0:
            conteo_categorias = (
                df_filtrado["categoria"]
                .value_counts()
                .reset_index()
            )
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

    st.subheader("Correos clasificados - vista técnica")

    columnas_tecnicas = [
        "fecha",
        "remitente",
        "asunto",
        "categoria",
        "confianza",
        "estado"
    ]

    if total_correos > 0:
        st.dataframe(
            df_filtrado[columnas_tecnicas],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No hay correos disponibles con los filtros actuales.")
        return

    st.divider()

    st.subheader("Detalle técnico del correo")

    opciones = df_filtrado["id"].astype(str) + " - " + df_filtrado["asunto"]

    correo_seleccionado = st.selectbox(
        "Selecciona un correo para revisar el resultado del modelo",
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

    st.caption(
        "Vista técnica: acceso restringido para revisión del modelo, métricas y confianza."
    )

# =========================
# Flujo principal
# =========================
df = cargar_datos()

modo_tecnico = verificar_acceso_tecnico()
df_filtrado = aplicar_filtros(df, modo_tecnico)

if modo_tecnico:
    mostrar_vista_tecnica(df_filtrado)
else:
    mostrar_vista_usuario(df_filtrado)
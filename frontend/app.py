from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
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
@st.cache_data(ttl=30)
def cargar_datos_desde_api(api_base_url):
    response = requests.get(f"{api_base_url}/emails/", timeout=5)
    response.raise_for_status()

    correos = response.json()

    if not correos:
        return pd.DataFrame()

    df_api = pd.DataFrame(correos)

    df_api = df_api.rename(
        columns={
            "sender": "remitente",
            "subject": "asunto",
            "body": "cuerpo",
            "predicted_category": "categoria",
            "confidence": "confianza",
            "created_at": "fecha",
        }
    )

    if "processing_time_ms" not in df_api.columns:
        df_api["processing_time_ms"] = 0.0

    df_api["estado"] = "Clasificado"

    columnas_necesarias = [
        "id",
        "fecha",
        "remitente",
        "asunto",
        "cuerpo",
        "categoria",
        "confianza",
        "estado",
        "processing_time_ms",
    ]

    for columna in columnas_necesarias:
        if columna not in df_api.columns:
            df_api[columna] = ""

    return df_api[columnas_necesarias]


@st.cache_data
def cargar_datos_demo():
    ruta = Path(__file__).parent / "data" / "correos_demo.csv"
    df_demo = pd.read_csv(ruta)

    if "processing_time_ms" not in df_demo.columns:
        df_demo["processing_time_ms"] = 0.0

    return df_demo


def cargar_datos():
    api_base_url = st.secrets.get("API_BASE_URL", "http://127.0.0.1:8000")

    try:
        df_api = cargar_datos_desde_api(api_base_url)

        if not df_api.empty:
            st.sidebar.success("Datos cargados desde backend")
            return df_api

        st.sidebar.warning("Backend sin correos. Usando datos demo.")
        return cargar_datos_demo()

    except requests.exceptions.RequestException:
        st.sidebar.warning("Backend no disponible. Usando datos demo.")
        return cargar_datos_demo()

def sincronizar_gmail():
    api_base_url = st.secrets.get(
        "API_BASE_URL",
        "http://127.0.0.1:8000"
    )

    response = requests.post(
        f"{api_base_url}/gmail/classify",
        timeout=60
    )

    response.raise_for_status()

    return response.json()

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
        confianza_numerica = pd.to_numeric(
            df_filtrado["confianza"],
            errors="coerce"
        )

        if confianza_numerica.notna().any():
            confianza_minima = st.sidebar.slider(
                "Confianza mínima",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05
            )

            df_filtrado = df_filtrado[
                confianza_numerica >= confianza_minima
            ]
        else:
            st.sidebar.info(
                "La confianza actual del backend es temporal y no numérica."
            )

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

    mostrar_formulario_clasificacion()

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
    st.title("Panel técnico del modelo")
    st.caption(
        "Vista para revisar métricas, confianza y comportamiento de la clasificación."
    )

    total_correos = len(df_filtrado)
    confianza_numerica = pd.to_numeric(
        df_filtrado["confianza"],
        errors="coerce"
    )

    confianza_promedio = (
        confianza_numerica.mean()
        if total_correos > 0 and confianza_numerica.notna().any()
        else None
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
        if confianza_promedio is not None:
            st.metric("Confianza promedio", f"{confianza_promedio:.2%}")
        else:
            st.metric("Confianza promedio", "Temporal")

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

        confianza_numerica = pd.to_numeric(
            df_filtrado["confianza"],
            errors="coerce"
        )

        if total_correos > 0 and confianza_numerica.notna().any():
            df_grafico = df_filtrado.copy()
            df_grafico["confianza_numerica"] = confianza_numerica

            fig_confianza = px.scatter(
                df_grafico,
                x="fecha",
                y="confianza_numerica",
                color="categoria",
                hover_data=["asunto", "remitente"],
                title="Nivel de confianza de clasificación"
            )

            st.plotly_chart(fig_confianza, use_container_width=True)
        elif total_correos > 0:
            st.info(
                "El backend actual devuelve una confianza temporal. "
                "El gráfico estará disponible cuando la confianza sea numérica."
            )
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
        confianza_correo = pd.to_numeric(
            pd.Series([correo["confianza"]]),
            errors="coerce"
        ).iloc[0]

        if pd.notna(confianza_correo):
            st.metric("Confianza", f"{confianza_correo:.2%}")
        else:
            st.metric("Confianza", str(correo["confianza"]))
        st.markdown(f"**Estado:** {correo['estado']}")

    st.caption(
        "Vista técnica: acceso restringido para revisión del modelo, métricas y confianza."
    )

# ==============================
# Ingreso de correo a clasificar
# ==============================
def mostrar_formulario_clasificacion():
    api_base_url = st.secrets.get("API_BASE_URL", "http://127.0.0.1:8000")

    with st.expander("Clasificar nuevo correo"):
        with st.form("form_clasificar_correo"):
            remitente = st.text_input("Remitente")
            asunto = st.text_input("Asunto")
            cuerpo = st.text_area("Contenido del correo")

            enviar = st.form_submit_button("Clasificar correo")

            if enviar:
                if not remitente or not asunto or not cuerpo:
                    st.warning("Completa remitente, asunto y contenido.")
                    return

                payload = {
                    "sender": remitente,
                    "subject": asunto,
                    "body": cuerpo,
                }

                try:
                    response = requests.post(
                        f"{api_base_url}/emails/classify",
                        json=payload,
                        timeout=10
                    )
                    response.raise_for_status()

                    st.success("Correo clasificado correctamente.")
                    st.cache_data.clear()
                    st.rerun()

                except requests.exceptions.RequestException:
                    st.error(
                        "No se pudo conectar con el backend para clasificar el correo."
                    )

# =========================
# Flujo principal
# =========================
# =========================
# Flujo principal
# =========================

if st.sidebar.button("Sincronizar Gmail"):

    try:
        resultado = sincronizar_gmail()

        st.sidebar.success(
            f"Se procesaron {resultado['processed']} correos"
        )

        st.cache_data.clear()
        st.rerun()

    except requests.exceptions.RequestException:
        st.sidebar.error(
            "No se pudo conectar con Gmail"
        )


df = cargar_datos()


if st.sidebar.button("Actualizar datos"):
    st.cache_data.clear()
    st.rerun()


modo_tecnico = verificar_acceso_tecnico()

df_filtrado = aplicar_filtros(
    df,
    modo_tecnico
)

if modo_tecnico:
    mostrar_vista_tecnica(df_filtrado)
else:
    mostrar_vista_usuario(df_filtrado)
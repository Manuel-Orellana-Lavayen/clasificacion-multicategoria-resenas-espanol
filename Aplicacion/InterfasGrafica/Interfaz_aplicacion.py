import pandas as pd
import streamlit as st
import requests
import plotly.io as pio
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title= "Clasificador de Reseñas- Manuel Orellana",layout= "wide")
st.title("Clasificador de textos - Regresión Logística",text_alignment= "center" )

diccionario_urls = {
    "TF-IDF": ["http://127.0.0.1:8000/PredecirTfidf"],
    "Embeddings": ["http://127.0.0.1:8000/PredecirEmbeddings"],
    "Ensamble" : ["http://127.0.0.1:8000/PredecirEnsamble"]
}

#===============================================================================================#
# Texto individual (Para pruebas con texto individual del modelo)                               #
#===============================================================================================#
st.subheader("Texto Individual",text_alignment= "center" )
# Cargando texto
with st.form("Formulario_texto_individual"):
    texto= st.text_input("Ingresa tu comentario")
    modelo = st.radio(label = "Modelo a usar", options=["TF-IDF", "Embeddings", "Ensamble"])
    enviar = st.form_submit_button(label= "Clasificar Comentario")

# Analizando texto
if enviar:
    if len(texto.strip()) < 1:
        st.error("El texto esta vacio")
    else:
        texto_serie = pd.Series(texto)
        respuesta = requests.post(
                                        url= diccionario_urls[modelo][0], #La url de la función
                                        json={"texto": texto_serie.tolist()} #Lo que se le envia a la función
                                        )

        if respuesta.status_code == 200:
            datos = respuesta.json()

            predicciones = datos["Predicciones"]
            reporte_baja_confianza = datos["Reporte Baja Confianza"]
            reporte_no_procesados = datos["Reporte No Procesados"]

            predicciones = pd.DataFrame(predicciones)
            reporte_baja_confianza = pd.DataFrame(reporte_baja_confianza)
            reporte_no_procesados= pd.DataFrame(reporte_no_procesados)

            if not predicciones.empty:
                st.dataframe(predicciones)
            if not reporte_baja_confianza.empty:
                st.dataframe(reporte_baja_confianza)
            if not reporte_no_procesados.empty:
                st.dataframe(reporte_no_procesados)

        else:
            st.error("Error al comunicarse con la API")
            st.error(f"Error HTTP: {respuesta.status_code}")
            st.write(respuesta.text)

#===============================================================================================#
# Archivo CSV (Para analizar y obtener predicciones de varios textos en simultaneo)             #
#===============================================================================================#
st.subheader("Archivo CSV",text_alignment= "center" )
df_archivo_ejemplo = pd.read_csv(filepath_or_buffer = BASE_DIR/"ArchivoEjemplo.csv",usecols=["text","label"])
csv_archivo_ejemplo = df_archivo_ejemplo.to_csv(index=False)
# Borton de descarga Para Archivo de ejemplo
st.download_button(
    label="Pulsa este boton para descargar archivo de Ejemplo",
    data=csv_archivo_ejemplo,
    file_name="ejempo_csv_reseñas.csv",
    mime="text/plain"
)
#Cargando Archivo CSV
with st.form("formulario_archivo"):
    archivo_csv = st.file_uploader(label = "Sube archivo CSV", type=["csv"])
    cargar_csv = st.form_submit_button("Cargar archivo")

# Una vez que el CSV se cargue, lo prepara
if cargar_csv:
    if archivo_csv is None:
        st.error("Debes subir un archivo CSV.")
    else:
        df_textos_csv = pd.read_csv(archivo_csv)
        columnas_csv = df_textos_csv.select_dtypes(include=["object", "string"]).columns.tolist()

        if not columnas_csv:
            st.error("El archivo no contiene columnas de texto.")
        else:
            st.session_state["df_textos_csv"] = df_textos_csv #Se usa sesion estate para evitar cargar el archivo a cada rato y lo guarda enla sesion
            st.session_state["columnas_csv"] = columnas_csv
            st.session_state["archivo_cargado_csv"] = True #Valor Booleano para ingresar a el menu de opciones

#Solicita informacion de confighuracion de analisis y hace llamada a la función
if (st.session_state.get("archivo_cargado_csv", False)):
    # Cargando Elementos der sesion (Evita que en cada llamada haya que llamarlos nuevamente)
    df_textos = st.session_state["df_textos_csv"]
    columnas = st.session_state["columnas_csv"]

    #Formulario de configuracion de analisis de CSV
    with st.form("formulario_configuracion_csv"):
        columna = st.radio(label = "Columna a analizar", options=columnas)
        modelo = st.radio(label = "Modelo a usar", options=["TF-IDF", "Embeddings", "Ensamble"])
        enviar_configuracion = st.form_submit_button("Ejecutar clasificación")

    # Haciendo llamada con configuración
    if enviar_configuracion:
        textos = df_textos[columna]
        respuesta = requests.post(
                                    url= diccionario_urls[modelo][0],
                                    json ={"texto": textos.to_list()}
                                 )

        if respuesta.status_code == 200:
            datos = respuesta.json()

            # Obtener elementos de la llamada
            predicciones_csv = datos["Predicciones"]
            reporte_baja_confianza_csv = datos["Reporte Baja Confianza"]
            reporte_nulos_csv = datos["Reporte No Procesados"]
            dict_graficos_csv = datos["Graficos"]

            # Convertir elementos de la llamada
            predicciones_csv = pd.DataFrame(predicciones_csv)
            reporte_baja_confianza_csv = pd.DataFrame(reporte_baja_confianza_csv)
            reporte_nulos_csv = pd.DataFrame(reporte_nulos_csv)

            # Guardar en la sesion los elementos de la llamada
            st.session_state["predicciones_csv"] = predicciones_csv
            st.session_state["reporte_nulos_csv"] = reporte_nulos_csv
            st.session_state["reporte_baja_confianza_csv"] = reporte_baja_confianza_csv
            st.session_state["dict_graficos_csv"] = dict_graficos_csv

        else:
            st.error("Error al comunicarse con la API")
            st.error(f"Error HTTP: {respuesta.status_code}")
            st.write(respuesta.text)

#Realizando Predicciones
if ("predicciones_csv" in st.session_state):
    predicciones_csv = st.session_state["predicciones_csv"]
    reporte_nulos_csv = st.session_state["reporte_nulos_csv"]
    reporte_baja_confianza_csv = st.session_state["reporte_baja_confianza_csv"]
    dict_graficos_csv = st.session_state["dict_graficos_csv"]

    grafico_frecuencia_clases_1 =  dict_graficos_csv["Frecuencias de clases"]
    grafico_frecuencia_clases_2 = dict_graficos_csv["Frecuencias de clases No Clasificados"]
    grafico_frecuencia_clasificaciones = dict_graficos_csv["Frecuencia de clasificaciones"]

    if grafico_frecuencia_clases_1 != "No_grafico":
        grafico_frecuencia_clases_1 = pio.from_json(grafico_frecuencia_clases_1)
    if grafico_frecuencia_clases_2 != "No_grafico":
        grafico_frecuencia_clases_2 = pio.from_json(grafico_frecuencia_clases_2)
    grafico_frecuencia_clasificaciones = pio.from_json(grafico_frecuencia_clasificaciones)

    columna_1_csv, columna_2_csv, columna_3_csv, columna_4_csv = st.columns(4)

    with columna_1_csv:
        ver_predicciones_csv = st.button("Ver predicciones")

    with columna_2_csv:
        ver_baja_confianza_csv = st.button("Ver reporte de baja confianza")

    with columna_3_csv:
        ver_nulos_csv = st.button("Ver tabla textos no procesados")

    with columna_4_csv:
        ver_grafico_csv = st.button( "Ver gráficos")

    if ver_predicciones_csv:
        st.dataframe(predicciones_csv)

    if ver_baja_confianza_csv:
        st.dataframe(reporte_baja_confianza_csv)

    if ver_nulos_csv:
        st.dataframe(reporte_nulos_csv)

    if ver_grafico_csv:
        if not isinstance(grafico_frecuencia_clases_1, str):
            st.plotly_chart(grafico_frecuencia_clases_1)
        if not isinstance(grafico_frecuencia_clases_2, str):
            st.plotly_chart(grafico_frecuencia_clases_2)
        if not isinstance(grafico_frecuencia_clasificaciones, str):
            st.plotly_chart(grafico_frecuencia_clasificaciones)

# Autor al pie de página
st.caption("2026 - Desarrollado por: Manuel Elias Orellana Lavayen")
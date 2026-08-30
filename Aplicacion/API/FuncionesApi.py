# Creado Por Manuel Elias Orellana Lavayen - 2026
"""
    Funciónes que Predicen

    Este modulo contiene 2 funciones que tienen como onjetivo usar los predictores previamente creados en "usar_pipelines".
    La tabla de informacion entrega las prediccione sen bruto, mientras que 'predecir' aplica reglas logicas sobre las
    probabilidades de las predicciones para mejorar su resultado final, descartando predicciones de baja confianza.
"""


from API.CreacionPipelines.Pipelines.usar_pipelines import predictor_tfidf, predictor_embeddings, predictor_ensamble
import pandas as pd
import plotly.express as px
from API.CreacionPipelines.Recursos.EstilosGraficosPlotly.EstilosPlotly import estilos_plotly

def tabla_informacion_predicciones (texto, predictor):
    """Realiza las predicciones con el predictor, tanto las predicciones normales como las de sus probabilidades,
    luego elimina columas repetidas y concatena las predicciones normales con la de las probabilidades.

        Args:
                textos (pd.Series): Serie de textos que se desea procesar.
                predictor: Predictor creado en 'usar_pipelines'.

            Returns:
                tuple:
                    - tabla_predicciones
                    - Reporte de no procesados
    """
    tabla_predicciones, reporte = predictor.predict(texto)
    tabla_probabilidades, _= predictor.predict_proba(texto)
    tabla_probabilidades = tabla_probabilidades.drop(columns=["texto"])
    tabla_resultados = pd.concat(objs= [tabla_predicciones, tabla_probabilidades], axis=1)

    return tabla_resultados, reporte

def predecir(texto, tfidf = False, embeddings= False, ensamble = False):
    """Clasifica textos aplicando reglas de confianza sobre las predicciones.

        Selecciona uno de los predictores disponibles y obtiene las predicciones junto con las probabilidades de cada clase.

        Una predicción se considera válida cuando la probabilidad de su clase supera el umbral establecido para dicha clase.
        Las predicciones que no superan este umbral pasan a una etapa de revisión.

        Las predicciones de baja confianza pueden ser reclasificadas como neutras cuando la diferencia entre la probabilidad
        de la clase neutra y las otras clases se encuentra dentro del margen de duda establecido. Esta regla busca mejorar
        el tratamiento de la clase neutra, debido a las dificultades del modelo para identificarla.

            Args:
                    texto: Textos que se desean clasificar.
                    tfidf (bool): Utiliza el predictor basado en TF-IDF.
                    embeddings (bool): Utiliza el predictor basado en embeddings.
                    ensamble (bool): Utiliza el predictor basado en el ensamble.

                Returns:
                    tuple:
                        - DataFrame con las predicciones aceptadas y sus probabilidades.
                        - DataFrame con las predicciones clasificadas como de baja confianza.
                        - DataFrame con los textos que no pudieron ser procesados.
        """
    if tfidf:
        tabla_predicciones, reporte_no_procesados = tabla_informacion_predicciones(texto, predictor_tfidf)

        umbral_clase_0 = 0.60
        umbral_clase_2 = 0.60
        umbral_clase_1 = 0.50

    if embeddings:
        tabla_predicciones, reporte_no_procesados = tabla_informacion_predicciones(texto, predictor_embeddings)

        umbral_clase_0 = 0.60
        umbral_clase_2 = 0.60
        umbral_clase_1 = 0.50

    if ensamble:
        tabla_predicciones, reporte_no_procesados = tabla_informacion_predicciones(texto, predictor_ensamble)

        umbral_clase_0 = 0.60
        umbral_clase_2 = 0.60
        umbral_clase_1 = 0.50


    margen_duda = 0.15

    probabilidades = tabla_predicciones[[0, 1, 2]]
    confianza = probabilidades.max(axis=1)
    prediccion = tabla_predicciones["Predicción"]

    #Mascara de validos, donde solo pasan quienes tengan un umbral reqeurido en las predicciones positivas y negativas
    mascara_validos = \
        (((prediccion == 0) & (confianza >= umbral_clase_0)) |
         ((prediccion == 2) & (confianza >= umbral_clase_2))|
         ((prediccion == 1) & (confianza >= umbral_clase_1)))

    # Los validos por el momento
    tabla_predicciones_validos = tabla_predicciones.loc[mascara_validos].copy()


    # Los que se van a revisar
    tabla_predicciones_revision= tabla_predicciones.loc[~mascara_validos].copy()

    probabilidades_revision_0= tabla_predicciones_revision[0]
    probabilidades_revision_1= tabla_predicciones_revision[1]
    probabilidades_revision_2= tabla_predicciones_revision[2]

    #mascaras booleanas de desicion por duda
    duda_negativo_neutro = (
            (tabla_predicciones_revision["Predicción"] == 0) &
            (abs(probabilidades_revision_0 - probabilidades_revision_1) <= margen_duda)
    )

    duda_positivo_neutro = (
            (tabla_predicciones_revision["Predicción"] == 2) &
            (abs(probabilidades_revision_2 - probabilidades_revision_1) <= margen_duda)
    )

    duda_neutro_negativo = (
            (tabla_predicciones_revision["Predicción"] == 1) &
            (abs(probabilidades_revision_0 - probabilidades_revision_1) <= margen_duda)
    )

    duda_neutro_positivo = (
            (tabla_predicciones_revision["Predicción"] == 1) &
            (abs(probabilidades_revision_2 - probabilidades_revision_1) <= margen_duda)
    )

    mascara_neutro = duda_negativo_neutro | duda_positivo_neutro | duda_neutro_positivo | duda_neutro_negativo


    tabla_predicciones_neutras_aprobadas = tabla_predicciones_revision.loc[mascara_neutro].copy()
    tabla_predicciones_neutras_aprobadas["Predicción"] = 1

    tabla_predicciones_validos = pd.concat(objs=[tabla_predicciones_validos, tabla_predicciones_neutras_aprobadas],axis=0)

    reporte_baja_confianza = tabla_predicciones_revision[~mascara_neutro]

    valores = {
        0: "Negativo",
        1: "Neutro",
        2: "Positivo"
    }
    nombres= {
        0: "Probabilidad Negativo",
        1: "Probabilidad Neutro",
        2: "Probabilidad Positivo"
    }
    tabla_predicciones_validos["Predicción"] = tabla_predicciones_validos["Predicción"].map(valores)
    tabla_predicciones_validos = tabla_predicciones_validos.rename(columns= nombres)
    reporte_baja_confianza["Predicción"] = reporte_baja_confianza["Predicción"].map(valores)
    reporte_baja_confianza = reporte_baja_confianza.rename(columns=nombres)

    return tabla_predicciones_validos, reporte_baja_confianza, reporte_no_procesados


def graficos(tabla_predicciones : pd.DataFrame, tabla_reporte_baja_confianza:pd.DataFrame, tabla_reportes_no_procesados:pd.DataFrame):
    """Crea 3 graficos en base a los resultados de la tupla resultante de la función 'predecir',
       Además se agregan estilos a el grafico usando la funcion 'estilos_plotly' del modulo 'EstilosPlotly.py', la cual
       tambien permite convertir el grafico a Json, para que pase por la Api
       En caso de que un grafico este vacio por la falta de datos, se eguarda un texto de 'No_grafico'

        Args:
            tabla_predicciones : pd.DataFrame
            tabla_reporte_baja_confianza:pd.DataFrame
            tabla_reportes_no_procesados:pd.DataFrame

        Returns:
            tuple:
                - Json de Grafico de pastel de frecuencias de clases de las predicciones,
                - Json de Grafico de pastel de frecuencias de clases de baja confianza de las predicciones ,
                - Json de Grafico de barras de cantidad de clasificados, nulos y de baja confianza
    """

    num_clasificados = tabla_predicciones.shape[0]
    num_reportes_baja_confianza = tabla_reporte_baja_confianza.shape[0]
    num_reportes_no_procesados = tabla_reportes_no_procesados.shape[0]
    colores_sentimiento = {
        "Negativo": "#EF553B",  # Rojo
        "Neutro": "#FECB52",  # Amarillo
        "Positivo": "#00CC96"  # Verde
    }
    grafico_pastel_frecuencias_clases = px.pie(
                                                data_frame=tabla_predicciones,
                                                names="Predicción",
                                                color="Predicción",
                                                color_discrete_map=colores_sentimiento,
                                               )

    grafico_pastel_frecuencias_clases_baja_confianza = px.pie(
                                                                data_frame=tabla_reporte_baja_confianza,
                                                                names="Predicción",
                                                                color="Predicción",
                                                                color_discrete_map=colores_sentimiento,
                                                             )

    grafico_barras = px.bar(

                                x = ["Clasificados", "Baja Confianza", "No Clasificados"],
                                y= [num_clasificados, num_reportes_baja_confianza, num_reportes_no_procesados],
                                color=["Clasificados", "Baja Confianza",  "No Clasificados"],
                                color_discrete_map={
                                    "Clasificados": "#EF553B",
                                    "Baja Confianza": "#FECB52",
                                    "No Clasificados": "#00CC96"
                                }
                           )
    grafico_pastel_frecuencias_clases = estilos_plotly(figura= grafico_pastel_frecuencias_clases, titulo= "Frecuencia de Clases", color_titulo="#FFFFFF", titulo_leyenda="Clases", color_texto_leyenda="#FFFFFF", fondo_transparente= True, leyenda= True, json= True)
    grafico_pastel_frecuencias_clases_baja_confianza = estilos_plotly(figura=grafico_pastel_frecuencias_clases_baja_confianza,titulo="Frecuencia de Clases En Baja Confianza", color_titulo="#FFFFFF",titulo_leyenda="Clases", color_texto_leyenda="#FFFFFF",fondo_transparente=True, leyenda=True, json=True)
    grafico_barras = estilos_plotly(figura= grafico_barras, titulo= "Clasificados VS Baja Confianza VS No Clasificados", color_titulo="#FFFFFF", titulo_leyenda="Reportes", color_texto_leyenda="#FFFFFF", fondo_transparente= True, leyenda= True, json= True)

    if num_clasificados == 0:
        grafico_pastel_frecuencias_clases = "No_grafico"
    if num_reportes_baja_confianza ==0:
        grafico_pastel_frecuencias_clases_baja_confianza = "No_grafico"
    return grafico_pastel_frecuencias_clases, grafico_pastel_frecuencias_clases_baja_confianza, grafico_barras

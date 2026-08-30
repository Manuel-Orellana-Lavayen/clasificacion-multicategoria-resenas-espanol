# Creado Por Manuel Elias Orellana Lavayen - 2026
"""
    Clases predictoras para los modelos de clasificación.

    Este módulo contiene clases que reciben un pipeline previamente construido
    y proporcionan métodos para realizar predicciones y obtener probabilidades.

    Las clases predictoras se encargan de evitar que textos eliminados durante
    la limpieza lleguen a los vectorizadores o modelos. Esto es necesario porque
    un texto puede quedar vacío después de eliminar elementos como stopwords,
    lo que puede provocar errores durante la vectorización o clasificación.

    Los textos que no pueden ser procesados se incluyen en un reporte separado.
"""

import pandas as pd
from sklearn.pipeline import Pipeline

# Clases Predictores
class Predictor:
    """Ejecuta predicciones sobre un pipeline de clasificación.

        Esta clase permite realizar predicciones sobre textos evitando que los
        textos eliminados durante la etapa de limpieza lleguen a las etapas de
        vectorización y clasificación.

        Los textos válidos se procesan mediante el pipeline, mientras que los
        textos eliminados se devuelven en un reporte separado.
        """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def _preparar_textos(self, textos):
        """Prepara los textos antes de ejecutar la vectorización y predicción.

            Ejecuta únicamente la etapa de limpieza del pipeline, identifica los textos que fueron eliminados y crea un nuevo pipeline
            que comienza desde la etapa de vectorización.

            Returns:
                tuple:
                    - Serie con los textos originales.
                    - Serie con los textos válidos.
                    - DataFrame con los textos eliminados.
                    - Pipeline restante desde la vectorización.
        """

        textos = pd.Series(textos)

        # Ejecutar únicamente la limpieza
        textos_limpios = self.pipeline.named_steps["Limpieza"].transform(textos)

        # Detectar válidos e inválidos
        textos_validos = textos_limpios.dropna()
        nulos = textos_limpios.isna() #Son nulos

        # Reporte
        reporte_nulos = pd.DataFrame({"indice": textos.index[nulos],"texto": textos.loc[nulos]})

        # Pipeline desde Vectorización
        pipeline_restante = Pipeline(self.pipeline.steps[1:])

        return textos, textos_validos, reporte_nulos, pipeline_restante

    def predict(self, textos):
        """Realiza la predicción de clases para los textos proporcionados.

            Los textos que resulten inválidos durante la limpieza no se envían al modelo y se incluyen en el reporte de textos eliminados.

            Returns:
                tuple:
                    - DataFrame con los textos válidos y sus predicciones.
                    - DataFrame con los textos eliminados.
        """

        (textos, textos_validos, reporte_nulos, pipeline_restante) = self._preparar_textos(textos)

        #Si no hay textos validos, enviar predicciones vacias
        if textos_validos.empty:
            predicciones_df = pd.DataFrame(columns=["texto", "Predicción"])

            return predicciones_df, reporte_nulos

        # Texto original correspondiente a los válidos
        textos_originales= textos.loc[textos_validos.index]

        # Vectorización + modelo
        predicciones= pipeline_restante.predict(textos_validos)

        predicciones_df = pd.DataFrame({"texto": textos_originales,"Predicción": predicciones})

        return predicciones_df, reporte_nulos

    def predict_proba(self, textos):
        """Obtiene las probabilidades de pertenencia a cada clase.

            Los textos que resulten inválidos durante la limpieza no se envían al modelo y se incluyen en el reporte de textos eliminados.

            Returns:
                tuple:
                    - DataFrame con los textos válidos y sus probabilidades.
                    - DataFrame con los textos eliminados.
        """

        (textos, textos_validos, reporte_nulos, pipeline_restante) = self._preparar_textos(textos)

        clases = pipeline_restante.named_steps["modelo"].classes_

        if textos_validos.empty:
            probabilidades_df = pd.DataFrame(columns=clases)
            probabilidades_df.insert(0, "texto", [])

            return probabilidades_df, reporte_nulos

        # Texto original correspondiente a los válidos
        textos_originales = textos.loc[textos_validos.index]

        # Vectorización + modelo
        probabilidades = pipeline_restante.predict_proba(textos_validos)

        probabilidades_df = pd.DataFrame(probabilidades, index=textos_validos.index, columns=clases)

        probabilidades_df.insert(0,"texto",textos_originales)

        return probabilidades_df, reporte_nulos

# Clases Predictores
class PredictorEnsamble:
    """Ejecuta predicciones sobre un pipeline de clasificación por ensamble.

        A diferencia de `Predictor`, esta clase trabaja con dos representaciones del texto
        Un texto se considera válido únicamente cuando ambas representaciones pueden ser generadas correctamente.
        """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def _preparar_textos(self, textos):
        """Prepara las dos representaciones de los textos para el ensamble.

            Ejecuta la etapa de limpieza, identifica los textos que fueroneliminados y conserva únicamente aquellos que
            poseen tanto una representación TF-IDF como una representación mediante embeddings.

            Returns:
                tuple:
                    - Serie con los textos originales.
                    - Tupla con los textos TF-IDF y embeddings válidos.
                    - DataFrame con los textos eliminados.
                    - Máscara booleana de textos válidos.
                    - Pipeline restante desde la vectorización.
        """

        textos = pd.Series(textos)

        # Ejecutar únicamente la limpieza
        tupla_textos_limpios = self.pipeline.named_steps["Limpieza"].transform(textos)

        # Detectar válidos e inválidos
        textos_limpios_tfidf = tupla_textos_limpios[0]
        textos_limpios_embeddings = tupla_textos_limpios[1]

        mascara_booleana_textos_validos = (textos_limpios_tfidf.notna() & textos_limpios_embeddings.notna())
        mascara_booleana_textos_eliminados = ~mascara_booleana_textos_validos

        # Reporte
        reporte_nulos = pd.DataFrame({"Indice": textos.index[mascara_booleana_textos_eliminados], "Textos": textos.loc[mascara_booleana_textos_eliminados]})
        # Tupla de textos Validos
        textos_validos_tfidf = textos_limpios_tfidf.loc[mascara_booleana_textos_validos]
        textos_validos_embeddings = textos_limpios_embeddings.loc[mascara_booleana_textos_validos]

        tupla_textos_validos = (textos_validos_tfidf, textos_validos_embeddings)
        # Pipeline desde Vectorización
        pipeline_restante = Pipeline(self.pipeline.steps[1:])

        return textos, tupla_textos_validos, reporte_nulos, mascara_booleana_textos_validos, pipeline_restante

    def predict(self, textos):
        """Realiza predicciones utilizando el modelo de ensamble.

            Returns:
                tuple:
                    - DataFrame con los textos válidos y sus predicciones.
                    - DataFrame con los textos eliminados.
        """
        (textos, tupla_textos_validos, reporte_nulos, mascara_booleana_textos_validos, pipeline_restante) = self._preparar_textos(textos)

        #Si no hay textos validos, enviar predicciones vacias
        if not mascara_booleana_textos_validos.any():
            predicciones_df = pd.DataFrame(columns=["texto", "Predicción"])

            return predicciones_df, reporte_nulos

        # Texto original correspondiente a los válidos
        textos_originales= textos.loc[mascara_booleana_textos_validos]

        # Vectorización + modelo
        # Eliminar
        modelo = pipeline_restante.named_steps["modelo"]

        print("\n========== DEBUG ENSAMBLE ==========")
        print("Tipo modelo:", type(modelo))
        print("Tiene le_:", hasattr(modelo, "le_"))

        if hasattr(modelo, "le_"):
            print("LabelEncoder:", modelo.le_)
            print("Tiene classes_:", hasattr(modelo.le_, "classes_"))

            if hasattr(modelo.le_, "classes_"):
                print("Clases:", modelo.le_.classes_)

        print("====================================\n")
        ### Hasta aqui eliminar
        predicciones= pipeline_restante.predict(tupla_textos_validos)

        predicciones_df = pd.DataFrame({"texto": textos_originales,"Predicción": predicciones}, index=textos_originales.index)

        return predicciones_df, reporte_nulos

    def predict_proba(self, textos):
        """Obtiene las probabilidades generadas por el modelo de ensamble.

            Returns:
                tuple:
                    - DataFrame con los textos válidos y sus probabilidades.
                    - DataFrame con los textos eliminados.
                """
        (textos, tupla_textos_validos, reporte_nulos, mascara_booleana_textos_validos, pipeline_restante) = self._preparar_textos(textos)

        clases = pipeline_restante.named_steps["modelo"].classes_

        if not mascara_booleana_textos_validos.any():
            probabilidades_df = pd.DataFrame(columns=clases)
            probabilidades_df.insert(0, "texto", [])

            return probabilidades_df, reporte_nulos

        # Texto original correspondiente a los válidos
        textos_originales = textos.loc[mascara_booleana_textos_validos]

        # Vectorización + modelo
        probabilidades = pipeline_restante.predict_proba(tupla_textos_validos)

        probabilidades_df = pd.DataFrame(probabilidades,index=textos_originales.index,columns=clases)

        probabilidades_df.insert(0,"texto",textos_originales)

        return probabilidades_df, reporte_nulos
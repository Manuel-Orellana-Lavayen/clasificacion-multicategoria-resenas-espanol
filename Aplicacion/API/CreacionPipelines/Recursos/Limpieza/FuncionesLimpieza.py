# Creado Por Manuel Elias Orellana Lavayen - 2026
"""
    Funciones de limpieza y preprocesamiento de texto.

    Este módulo contiene las funciones utilizadas para preparar los textos antes de ser enviados a los modelos de clasificación.

    Se proporcionan tres niveles de procesamiento:

    - Limpieza básica
    - Limpieza para TF-IDF
    - Limpieza para ensamble

    Los textos que quedan completamente vacíos después de la eliminación de stopwords se representan como valores NaN
    para evitar que sean procesados por las etapas posteriores del pipeline.
"""

import re
import spacy
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ruta_modelo_spacy = BASE_DIR/"es_core_news_md"

# Configuraciones Necesarias
nlp = spacy.load(
    ruta_modelo_spacy,
    disable=["parser", "ner"] # Descartar Análisis sintáctico de dependencias Y Reconocimiento de entidades
)

stop_words = nlp.Defaults.stop_words.copy()
remover = {
    # negaciones
    "no", "nunca", "jamás", "sin", "ni", "nadie",

    # palabras de sentimiento
    "bueno", "buena", "buenos",
    "malo", "mal",
    "mejor", "peor"
}
stop_words.difference_update(remover)

#Limpieza Básica
def limpiar_texto(texto: str):
    """Realiza la limpieza básica de un texto.

        Convierte el texto a minúsculas, elimina etiquetas HTML, elimina URLs y normaliza los espacios en blanco.

        Args:
            texto (str): Texto que se desea limpiar.

        Returns:
            str: Texto limpio.
    """
    # Minúsculas
    texto = texto.lower()

    # Eliminar HTML
    texto = re.sub(r"<.*?>", " ", texto)

    # Eliminar URL
    texto = re.sub(r"http\S+|www\S+", "", texto)

    # Eliminar espacios extra
    texto = " ".join(texto.split())

    return texto


#Limpieza Básica Serie Pandas
def limpiar_texto_serie(textos: pd.Series):
    """Aplica la limpieza básica a una serie de textos.

        Args:
            textos (pd.Series): Serie de textos que se desea limpiar.

        Returns:
            pd.Series: Serie con los textos limpiados.
        """
    textos_limpios = textos.apply(limpiar_texto)

    return textos_limpios

# Eliminación de Stop words y lematización Serie Pandas
def limpiar_texto_tfidf_serie(textos: pd.Series):
    """Preprocesa una serie de textos para su uso con TF-IDF.

        Primero realiza la limpieza básica del texto y posteriormente aplica lematización y eliminación de stopwords

        Si un texto queda completamente vacío después del procesamiento, se representa mediante NaN

        Args:
            textos (pd.Series): Serie de textos que se desea procesar.

        Returns:
            list: Lista de textos procesados
        """
    textos_limpios_basico = limpiar_texto_serie(textos)

    docs = nlp.pipe(textos_limpios_basico, batch_size=500)
    textos_limpios = []

    for doc in docs:
        tokens = []
        for token in doc:
            if token.lemma_ not in stop_words:
                tokens.append(token.lemma_)

        texto_resultado = " ".join(tokens)

        if not texto_resultado:
            texto_resultado = np.nan

        textos_limpios.append(texto_resultado)

    return textos_limpios

def limpiar_texto_ensamble_serie(textos: pd.Series):
    """Genera las representaciones de texto necesarias para el ensamble.

        Genera dos versiones de cada texto:

        - Una versión procesada para TF-IDF.
        - Una versión con limpieza básica para generar embeddings.

        Args:
            textos (pd.Series): Serie de textos que se desea procesar.

        Returns:
            tuple:
                Una tupla con la forma
                ``(textos_limpios_tfidf, textos_limpios_embeddings)``.
        """
    textos_limpios_embeddings = limpiar_texto_serie(textos)

    textos_limpios_tfidf = limpiar_texto_tfidf_serie(textos)

    return textos_limpios_tfidf, textos_limpios_embeddings
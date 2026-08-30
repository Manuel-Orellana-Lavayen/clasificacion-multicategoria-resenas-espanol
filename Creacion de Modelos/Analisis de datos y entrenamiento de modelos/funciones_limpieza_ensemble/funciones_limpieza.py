import re
import spacy
import pandas as pd
from pathlib import Path

"""
__file__ -> es la ruta del archivo actual (FuncionesLimpieza.py).
.resolve() -> obtiene la ruta absoluta.
.parent → obtiene la carpeta que contiene ese archivo
"""
BASE_DIR = Path(__file__).resolve().parent

ruta_modelo_spacy = BASE_DIR.parent / "Recursos de procesamiento de texto/es_core_news_md"

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

    # Minúsculas
    texto = texto.lower()

    # Eliminar HTML
    texto = re.sub(r"<.*?>", " ", texto)

    # Eliminar URL
    texto = re.sub(r"http\S+|www\S+", "", texto)

    # Eliminar espacios extra
    texto = " ".join(texto.split())

    texto = " ".join(texto.split())

    return texto


#Limpieza Básica Serie Pandas
def limpiar_texto_serie(textos: pd.Series):

    textos_limpios = textos.apply(limpiar_texto)

    return textos_limpios

# Eliminación de Stop words y lematización Serie Pandas
def limpiar_texto_tfidf_serie(textos: pd.Series):

    textos_limpios_basico = limpiar_texto_serie(textos)

    docs = nlp.pipe(textos_limpios_basico, batch_size=500)
    textos_limpios = []

    for doc in docs:
        tokens = []
        for token in doc:
            if token.lemma_ not in stop_words:
                tokens.append(token.lemma_)
        textos_limpios.append(" ".join(tokens))

    return textos_limpios


def limpiar_texto(texto):
    """
    Limpiar un texot de etiquetas HTML, URL, espacios extra y lo convierte a minusculas

    Args:
        texto: texto

    Returns:
        texto limpio
    """

    # Minúsculas
    texto = texto.lower()

    # Eliminar HTML
    texto = re.sub(r"<.*?>", " ", texto)

    # Eliminar URL
    texto = re.sub(r"http\S+|www\S+", "", texto)

    # Eliminar espacios extra
    texto = " ".join(texto.split())

    texto = " ".join(texto.split())

    if texto.strip() == "":
        return "[vacío]"
    return texto

# Eliminación de Stop words y lematización
def limpiar_texto_tfidf (dataframe: pd.DataFrame, ortografia = False):
    """
    Limpia el texto, además aplica lematizacion y eliminacion de stop words

    Args:
        dataframe: Dataframe que tenga los textos
        ortografia: Booleano que indique si se va limpiar texto con correccion ortografica

    Returns:
        lista con texto limpio
    """

    if ortografia:
        textos_limpios = dataframe["text_ortografia"].apply(limpiar_texto)
    else:
        textos_limpios = dataframe["text"].apply(limpiar_texto)

    docs = nlp.pipe(textos_limpios, batch_size=4000)

    resultado = []

    for doc in docs:

        tokens = []

        for token in doc:

            if token.lemma_ not in stop_words:
                tokens.append(token.lemma_)

        texto_final = " ".join(tokens)
        if texto_final.strip() == "":
            resultado.append("[vacío]")
        else:
            resultado.append(texto_final)

    return resultado
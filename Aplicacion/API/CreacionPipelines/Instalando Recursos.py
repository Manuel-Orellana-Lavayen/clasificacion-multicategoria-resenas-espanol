from pathlib import Path
import urllib.request
import os
import spacy
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
REPO_HF = "https://huggingface.co/Manuel-Orellana-Lavayen/modelos-clasificacion-multicategoria-resenas-espanol/resolve/main"

# RUTAS
MODELOS_CLASIFICACION = {
    BASE_DIR / "Recursos" / "ModelosClasificacion" / "modelo_embeddings.skops": f"{REPO_HF}/Modelos_clasificadores/modelo_embeddings.skops",
    BASE_DIR / "Recursos" / "ModelosClasificacion" / "modelo_ensamble.skops": f"{REPO_HF}/Modelos_clasificadores/modelo_ensamble.skops",
    BASE_DIR / "Recursos" / "ModelosClasificacion" / "modelo_tfidf.skops": f"{REPO_HF}/Modelos_clasificadores/modelo_tfidf.skops",
}

VECTORIZADORES = {
    BASE_DIR / "Recursos" / "Vectorizacion" / "vectorizador_tfidf.skops": f"{REPO_HF}/Vectorizador_tfidf/vectorizador_tfidf.skops"
}

RUTA_SPACY = (BASE_DIR / "Recursos" / "Limpieza" / "es_core_news_md")
RUTA_E5 = (BASE_DIR/ "Recursos"/ "Vectorizacion" / "multilingual-e5-base")

# DESCARGA DE ARCHIVOS SKOPS
def descargar_grupo_archivos(grupo_dict: dict):
    for ruta_local, url_descarga in grupo_dict.items():
        # Crea todas las carpetas necesarias para poder guardar el archivo en ruta_local y si ya existen, no las hace
        os.makedirs(os.path.dirname(ruta_local), exist_ok=True)
        if os.path.exists(ruta_local):
            continue

        try:
            urllib.request.urlretrieve(url_descarga,ruta_local)

        except Exception as e:
            raise RuntimeError(
                f"No se pudo descargar el recurso:\n"
                f"Archivo: {ruta_local}\n"
                f"URL: {url_descarga}\n"
                f"Error: {e}"
            ) from e

    return True

# MODELO SPACY
def preparar_spacy():
    if RUTA_SPACY.exists():
        return True

    try:
        nlp = spacy.load("es_core_news_md")
        RUTA_SPACY.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        nlp.to_disk(RUTA_SPACY)

        return True

    except Exception as e:
        raise RuntimeError(
            f"No se pudo preparar el modelo spaCy.\n"
            f"Ruta: {RUTA_SPACY}\n"
            f"Error: {e}"
        ) from e

# MODELO E5
def preparar_e5():

    if RUTA_E5.exists():
        return True

    try:
        modelo = SentenceTransformer("intfloat/multilingual-e5-base")
        RUTA_E5.parent.mkdir(parents=True,exist_ok=True)
        modelo.save(str(RUTA_E5))

        return True

    except Exception as e:
        raise RuntimeError(
            f"No se pudo preparar multilingual-e5-base.\n"
            f"Ruta: {RUTA_E5}\n"
            f"Error: {e}"
        ) from e


# PREPARAR TODO
def preparar_recursos():

    descargar_grupo_archivos(MODELOS_CLASIFICACION)
    descargar_grupo_archivos(VECTORIZADORES)
    preparar_spacy()
    preparar_e5()

    return True

try:
    if not preparar_recursos():
        raise RuntimeError("No se pudieron preparar los recursos.")

except Exception as e:
    print(f"ERROR AL PREPARAR RECURSOS: {e}")
    raise
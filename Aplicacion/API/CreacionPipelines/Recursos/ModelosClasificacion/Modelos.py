# Creado Por Manuel Elias Orellana Lavayen - 2026
"""
    Modelos de clasificación entrenados.

    Este módulo carga los modelos previamente entrenados y almacenados en archivos .pkl
"""

import skops.io as sio
from API.CreacionPipelines.Recursos.Clases.ClasesAdaptadorasEnsamble import TfidfAdaptador, EmbeddingAdaptador
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from pathlib import Path

# Ruta general
BASE_DIR = Path(__file__).resolve().parent


_tipos_regresion_logistica_tfidf = sio.get_untrusted_types(file= BASE_DIR / "modelo_tfidf.skops")
_tipos_regresion_logistica_embeddings = sio.get_untrusted_types(file= BASE_DIR / "modelo_embeddings.skops")
_tipos_ensamble = sio.get_untrusted_types(file=BASE_DIR / "modelo_ensamble.skops")

ENSAMBLE: VotingClassifier = sio.load(BASE_DIR/ "modelo_ensamble.skops",trusted=_tipos_ensamble)
REGRESION_LOGISTICA_TFIDF : LogisticRegression = sio.load(BASE_DIR / "modelo_tfidf.skops", trusted= _tipos_regresion_logistica_tfidf)
REGRESION_LOGISTICA_EMBEDDINGS : LogisticRegression = sio.load(BASE_DIR / "modelo_embeddings.skops", trusted= _tipos_regresion_logistica_embeddings)

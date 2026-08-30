# Creado Por Manuel Elias Orellana Lavayen - 2026
"""
    Vectorizadores

    Este módulo carga los vectorizadores tanto para TFIDF como para Embeddings
"""

from sentence_transformers import SentenceTransformer
import skops.io as sio
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
_tipos_vectorizador_tfidf = sio.get_untrusted_types(file= BASE_DIR / "vectorizador_tfidf.skops")
VECTORIZADOR_TFIDF = sio.load(BASE_DIR / "vectorizador_tfidf.skops", trusted=_tipos_vectorizador_tfidf)
VECTORIZADOR_EMBEDDINGS= SentenceTransformer(str(BASE_DIR /"multilingual-e5-base"), device= "cpu")
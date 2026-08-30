# Creado Por Manuel Elias Orellana Lavayen - 2026
"""
    Clases Envoltorio

    Este modulo contiene las clases Envoltorio. Las cuales son clases que permiten manipular como se comportan ciertos
    elementos dentro de un pipeline y a que metodos reaccionan.

    Se incluyen clases envoltorio para:
    - Limpieza básica de texto.
    - Limpieza específica para TF-IDF.
    - Limpieza para el modelo de ensamble.
    - Vectorización de Ensamble.
    - Vectorización de embeddings.
"""

from API.CreacionPipelines.Recursos.Limpieza.FuncionesLimpieza import limpiar_texto_serie, limpiar_texto_tfidf_serie, limpiar_texto_ensamble_serie
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class LimpiezaBasica(BaseEstimator, TransformerMixin):
    """Clase envoltorio para aplicar la limpieza básica de texto en un pipeline."""
    def __init__(self):
        pass

    def fit(self, textos, y=None):
        return self

    def transform(self, textos:pd.Series):
        textos = pd.Series(textos)
        return limpiar_texto_serie(textos)

class LimpiezaTFIDF(BaseEstimator, TransformerMixin):
    """Clase envoltorio para aplicar la limpieza de texto utilizada por TF-IDF."""
    def __init__(self):
        pass

    def fit(self, textos, y=None):
        return self

    def transform(self, textos:pd.Series):
        textos = pd.Series(textos)

        textos_limpios = limpiar_texto_tfidf_serie(textos) #Lista

        textos_limpios = pd.Series(textos_limpios,index=textos.index) #Serie

        return textos_limpios

class LimpiezaEnsamble(BaseEstimator, TransformerMixin):
    """Clase envoltorio para aplicar la limpieza utilizada por el modelo de ensamble.

        La transformación genera dos representaciones del texto: una destinada
        al vectorizador TF-IDF y otra destinada al modelo de embeddings.
        """
    def __init__(self):
        pass

    def fit(self, textos, y=None):
        return self

    def transform(self, textos:pd.Series):
        textos = pd.Series(textos)

        textos_limpios_tfidf, textos_limpios_embeddings = limpiar_texto_ensamble_serie(textos) #tupla

        textos_limpios_tfidf = pd.Series(textos_limpios_tfidf,index=textos.index) #Serie
        textos_limpios_embeddings= pd.Series(textos_limpios_embeddings, index=textos.index) #Serie

        return textos_limpios_tfidf, textos_limpios_embeddings

class vectorizador_ensamble_sklearn(BaseEstimator, TransformerMixin):
    """Clase envoltorio para vectorizar simultáneamente TF-IDF y embeddings.

       Recibe una tupla con los textos destinados a cada representación y
       devuelve una tupla con los vectores TF-IDF y los embeddings generados.
       """
    def __init__(self, v_tfidf, v_embeddings):
        self.v_tfidf = v_tfidf
        self.v_embeddings= v_embeddings

    def fit(self, textos, y=None):
        return self

    def transform(self, tupla_textos):
        textos_tfidf = list(tupla_textos[0])
        textos_embeddinggs= list(tupla_textos[1])

        # vectorizar tfidf
        vectores_tfidf = self.v_tfidf.transform(textos_tfidf)

        # vectorizar embeddings
        vectores_embeddings = self.v_embeddings.encode(textos_embeddinggs,show_progress_bar=False)

        return vectores_tfidf, vectores_embeddings

class vectorizador_embeddings_sklearn(BaseEstimator, TransformerMixin):
    """Clase envoltorio para generar embeddings dentro de un pipeline."""
    def __init__(self, model):
        self.model = model

    def fit(self, textos, y=None):
        return self

    def transform(self, textos):
        textos = list(textos)
        return self.model.encode(textos,show_progress_bar=False)
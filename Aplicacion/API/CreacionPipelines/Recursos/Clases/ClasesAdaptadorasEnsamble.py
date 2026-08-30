# Creado Por Manuel Elias Orellana Lavayen - 2026
"""
    Clases Adaptadoras para modelo Ensamble

    Este modulo contiene las clases adaptadoras para el modelo ensamble.
    Las clases adaptadoras son las que permiten a el modelo ensamble poder usar los respectivos modelos,
    ya que como el modelo Ensamble recibe una tupla con ambos tipos de vectorizacion,
    cada modelo debe de 'agarrar' su respectiva vectorización
"""

from sklearn.base import BaseEstimator, ClassifierMixin

class TfidfAdaptador(BaseEstimator, ClassifierMixin):
    """Adaptador para modelo que utiliza vectorización TF-IDF."""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        X_tfidf = X[0]
        return self.model.predict(X_tfidf)

    def predict_proba(self, X):
        X_tfidf = X[0]
        return self.model.predict_proba(X_tfidf)


class EmbeddingAdaptador(BaseEstimator, ClassifierMixin):
    """Adaptador para modelos que utilizan vectorización com Embeddings."""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        X_embeddings = X[1]
        return self.model.predict(X_embeddings)

    def predict_proba(self, X):
        X_embeddings = X[1]
        return self.model.predict_proba(X_embeddings)

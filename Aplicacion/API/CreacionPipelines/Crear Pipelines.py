# Creado Por Manuel Elias Orellana Lavayen - 2026
"""
    Creación de pipelines

    Este módulo contiene el codigo necesario para crear los Pipelines que proximanente va a usar los predictores.
    Este módulo puede ejecutarse si existen las clases envoltorio, vectorizadores y modelos
"""

from API.CreacionPipelines.Recursos.Clases.ClasesAdaptadorasEnsamble import EmbeddingAdaptador, TfidfAdaptador
from API.CreacionPipelines.Recursos.Clases.ClasesEnvoltorio import LimpiezaBasica, LimpiezaTFIDF, LimpiezaEnsamble, vectorizador_embeddings_sklearn, vectorizador_ensamble_sklearn
from API.CreacionPipelines.Recursos.Vectorizacion.Vectorizadores import  VECTORIZADOR_TFIDF, VECTORIZADOR_EMBEDDINGS
from API.CreacionPipelines.Recursos.ModelosClasificacion.Modelos import REGRESION_LOGISTICA_TFIDF, REGRESION_LOGISTICA_EMBEDDINGS, ENSAMBLE

from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

# Ruta general
BASE_DIR = Path(__file__).resolve().parent

# Modelos
regresion_logistica_tfidf = REGRESION_LOGISTICA_TFIDF
regresion_logistica_embeddings = REGRESION_LOGISTICA_EMBEDDINGS
ensamble =  ENSAMBLE

# Clases Envoltorio
limpieza_basica = LimpiezaBasica
limpieza_tfidf = LimpiezaTFIDF
limpieza_ensamble = LimpiezaEnsamble

vectorizador_embeddings= vectorizador_embeddings_sklearn(VECTORIZADOR_EMBEDDINGS)
vectorizador_tfidf= VECTORIZADOR_TFIDF
vectorizador_ensamble = vectorizador_ensamble_sklearn(VECTORIZADOR_TFIDF, VECTORIZADOR_EMBEDDINGS)

#Pipelines
pipeline_embeddings = Pipeline([
    ("Limpieza", LimpiezaBasica()),
    ("Vectorizacion", vectorizador_embeddings),
    ("modelo", regresion_logistica_embeddings)
])

pipeline_tf_idf= Pipeline([
    ("Limpieza", LimpiezaTFIDF()),
    ("Vectorizacion", vectorizador_tfidf),
    ("modelo", regresion_logistica_tfidf)
])

pipeline_ensamble= Pipeline([
    ("Limpieza", LimpiezaEnsamble()),
    ("Vectorizacion", vectorizador_ensamble),
    ("modelo", ensamble)
])

#Cargando Pipelines
joblib.dump(pipeline_embeddings, BASE_DIR/'Pipelines/pipeline_embeddings.joblib')
joblib.dump(pipeline_tf_idf, BASE_DIR/'Pipelines/pipeline_tfidf.joblib')
joblib.dump(pipeline_ensamble, BASE_DIR/'Pipelines/pipeline_ensamble.joblib')


# Creado Por Manuel Elias Orellana Lavayen - 2026
"""
    Predictores para diferentes modelos

    Este modulo contiene los predictores para cada uno de los modelos. Estos predictores fueron creados usando las respectivas
    clases envoltorio, adapatadores, vectorizadores y pipelines
"""

from API.CreacionPipelines.Recursos.Clases.ClasesAdaptadorasEnsamble import EmbeddingAdaptador, TfidfAdaptador
from API.CreacionPipelines.Recursos.Clases.ClasesEnvoltorio import LimpiezaBasica, LimpiezaTFIDF, LimpiezaEnsamble, vectorizador_embeddings_sklearn, vectorizador_ensamble_sklearn
from API.CreacionPipelines.Recursos.Clases.ClasesPredictoras import Predictor, PredictorEnsamble
from API.CreacionPipelines.Recursos.Vectorizacion.Vectorizadores import  VECTORIZADOR_TFIDF, VECTORIZADOR_EMBEDDINGS


from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

# Ruta general
BASE_DIR = Path(__file__).resolve().parent

from API.CreacionPipelines.Recursos.ModelosClasificacion.Modelos import REGRESION_LOGISTICA_TFIDF, REGRESION_LOGISTICA_EMBEDDINGS, ENSAMBLE
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
pipeline_tfidf: Pipeline = joblib.load(BASE_DIR/ 'pipeline_tfidf.joblib')
pipeline_embeddings: Pipeline = joblib.load(BASE_DIR/ 'pipeline_embeddings.joblib')
pipeline_ensamble: Pipeline = joblib.load(BASE_DIR/ 'pipeline_ensamble.joblib')

# Clases Predictoras
predictor_tfidf = Predictor(pipeline_tfidf)
predictor_embeddings = Predictor(pipeline_embeddings)
predictor_ensamble = PredictorEnsamble(pipeline_ensamble)
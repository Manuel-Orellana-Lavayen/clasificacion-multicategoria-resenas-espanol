# Creado Por Manuel Elias Orellana Lavayen - 2026
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from API.CreacionPipelines.Recursos.Clases.ClasesAdaptadorasEnsamble import (TfidfAdaptador,EmbeddingAdaptador)
from API.FuncionesApi import predecir, graficos
# Instancia de FastApi
app = FastAPI()

#Modelo Pydantic
class TextoEntrada(BaseModel):
    texto: list[str]

@app.post("/PredecirTfidf")
async def predecir_tfidf(datos: TextoEntrada):
    predicciones_tfidf, reporte_baja_confianza, reporte_nulos= predecir(datos.texto, tfidf= True)
    grafico_clases_1, grafico_clases_2, grafico_barras = graficos(predicciones_tfidf, reporte_baja_confianza, reporte_nulos)

    return {
            "Predicciones": predicciones_tfidf.to_dict(orient="records"),
            "Reporte Baja Confianza": reporte_baja_confianza.to_dict(orient="records"),
            "Reporte No Procesados": reporte_nulos.to_dict(orient="records"),
            "Graficos": {
                        "Frecuencias de clases": grafico_clases_1,
                        "Frecuencias de clases No Clasificados": grafico_clases_2,
                        "Frecuencia de clasificaciones": grafico_barras
                        }
            }

@app.post("/PredecirEmbeddings")
async def predecir_embeddings(datos: TextoEntrada):
    predicciones_embeddings, reporte_baja_confianza, reporte_nulos= predecir(datos.texto, embeddings= True)
    grafico_clases_1, grafico_clases_2, grafico_barras = graficos(predicciones_embeddings, reporte_baja_confianza, reporte_nulos)

    return {
            "Predicciones": predicciones_embeddings.to_dict(orient="records"),
            "Reporte Baja Confianza": reporte_baja_confianza.to_dict(orient="records"),
            "Reporte No Procesados": reporte_nulos.to_dict(orient="records"),
            "Graficos": {
                        "Frecuencias de clases": grafico_clases_1,
                        "Frecuencias de clases No Clasificados": grafico_clases_2,
                        "Frecuencia de clasificaciones": grafico_barras
                        }
            }

@app.post("/PredecirEnsamble")
async def predecir_ensamble(datos: TextoEntrada):
    predicciones_ensamble, reporte_baja_confianza, reporte_nulos= predecir(datos.texto, ensamble= True)
    grafico_clases_1, grafico_clases_2, grafico_barras = graficos(predicciones_ensamble, reporte_baja_confianza, reporte_nulos)

    return {
            "Predicciones": predicciones_ensamble.to_dict(orient="records"),
            "Reporte Baja Confianza": reporte_baja_confianza.to_dict(orient="records"),
            "Reporte No Procesados": reporte_nulos.to_dict(orient="records"),
            "Graficos": {
                        "Frecuencias de clases": grafico_clases_1,
                        "Frecuencias de clases No Clasificados": grafico_clases_2,
                        "Frecuencia de clasificaciones": grafico_barras
                        }
            }

if __name__ == "__main__":
    uvicorn.run(
        app= "api_aplicacion:app",
        host= "127.0.0.1",
        port= 8000,
        reload= False
    )

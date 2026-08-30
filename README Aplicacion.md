# clasificacion-multicategoria-resenas-espanol

## Aplicación

### librerías
# De preferencia instalar en Python 3.12.10

scikit-learn
- pandas
- numpy
- spacy
- sentence-transformers
- fastapi
- uvicorn
- pydantic
- cloudpathlib
- joblib
- plotly
- requests
- streamlit

```
├── Aplicación/
│   │
│   ├── API/
│   │   │
│   │   ├── CreacionPipelines/
│   │   │   │
│   │   │   ├── Pipelines/
│   │   │   │   └── usar_pipelines.py
│   │   │   │       └─ Funciones para utilizar los pipelines de predicción.
│   │   │   │
│   │   │   ├── Recursos/
│   │   │   │   │
│   │   │   │   ├── Clases/
│   │   │   │   │   ├── ClasesAdaptadorasEnsamble.py
│   │   │   │   │   ├── ClasesEnvoltorio.py
│   │   │   │   │   └── ClasesPredictoras.py
│   │   │   │   │       └─ Clases auxiliares utilizadas por los modelos.
│   │   │   │   │
│   │   │   │   ├── EstilosGraficosPlotly/
│   │   │   │   │   └── EstilosPlotly.py
│   │   │   │   │       └─ Define estilos reutilizables para gráficos Plotly.
│   │   │   │   │
│   │   │   │   ├── Limpieza/
│   │   │   │   │   └── FuncionesLimpieza.py
│   │   │   │   │       └─ Funciones utilizadas para el preprocesamiento del texto.
│   │   │   │   │
│   │   │   │   ├── ModelosClasificacion/
│   │   │   │   │   ├── modelo_embeddings.pkl
│   │   │   │   │   ├── modelo_ensamble.pkl
│   │   │   │   │   ├── modelo_tfidf.pkl
│   │   │   │   │   └── Modelos.py
│   │   │   │   │       └─ Carga los modelos de clasificación entrenados.
│   │   │   │   │
│   │   │   │   └── Vectorizacion/
│   │   │   │       ├── multilingual-e5-base/
│   │   │   │       ├── vectorizador_tfidf/
│   │   │   │       └── Vectorizadores.py
│   │   │   │           └─ Carga los recursos necesarios para vectorizar textos.
│   │   │   │
│   │   │   └── Crear Pipelines.py
│   │   │       └─ Construye los pipelines utilizando modelos,
│   │   │          vectorizadores y funciones de procesamiento.
│   │   │
│   │   ├── api_aplicacion.py
│   │   │   └─ Define la API y sus endpoints mediante FastAPI.
│   │   │
│   │   └── FuncionesApi.py
│   │       └─ Contiene las funciones finales utilizadas por la API.
│   │
│   └── Interfaz Gráfica/
│       │
│       ├── .streamlit/
│       │   └─ Configuración de la interfaz de Streamlit.
│       │
│       ├── static/
│       │   └─ Fuentes utilizadas por la interfaz.
│       │
│       ├── Interfaz_aplicacion.py
│       │   └─ Código principal de la interfaz gráfica.
│       │
│       └── ArchivoEjemplo.csv
│           └─ Archivo de ejemplo para probar la aplicación.
│
└── requirements_aplicación.txt
    └─ Dependencias necesarias para ejecutar la aplicación.
```

## Instalación y ejecución

### Creacion de Modelos
1. Crear una imagen del contenedor usando Dockerfile
2. Abrir la carpeta donde esten todos los archivos de Creacion de Modelos
3. En la terminal escribir el siguiente comando para crear un contenedor en base a la imagen ya creada:
docker run -it --gpus all --cpus="10" --memory="14.5g" --name proyectosaplicaciones-clasificaciondetexto-creaciondemodelos -v "$(pwd):/espacio_trabajo" entorno-ubuntu
4. Dentro del contenedor copiar los archivos pixi.toml y pixi.lock a la carpeta llamada "entorno_virtual"
5. Abrir una terminal para esa carpeta y ejecutar el siguiente comando para instalar en entorno virtual .pixi  -> "pixi install"
6. Configurar el interprete dentro de nuestro contenedor

### Aplicacion
1. Crear una imagen del contenedor usando Dockerfile
2. Abrir la carpeta donde esten todos los archivos de Creacion de Modelos
3. En la terminal escribir el siguiente comando para crear un contenedor en base a la imagen ya creada:
docker run -it --gpus all --cpus="10" --memory="14.5g" --name proyectosaplicaciones-clasificaciondetexto-aplicacion -v "$(pwd):/espacio_trabajo" entorno-python
4. Dentro del contenedor instalar directamente las librerias con pip install requirements_aplicación.txt
5. Configurar el interprete

## Autor
Manuel Elias Orellana Lavayen 

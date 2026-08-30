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

### API
#### CreacionPipelines
##### Pipelines
-  usar_pipelines.py -> Modulo con funciones para uso de los Pipelines
##### Recursos
###### Clases
- ClasesAdaptadorasEnsamble.py
- ClasesEnvoltorio.py
- ClasesPredictoras.py
###### EstilosGraficosPlotly
- EstilosPlotly.py -> Modulo con función de estilos para gráficos creados con Plotly
###### Limpieza
- FuncionesLimpieza.py 
###### ModelosClasificacion
- modelo_embeddings.pkl
- modelo_ensamble.pkl
- modelo_tfidf.pkl
- Modelos.py -> Modulo con carga de modelos
###### Vectorizacion
- multilingual-e5-base
- vectorizador_tfidf
- Vectorizadores.py -> Modulo con carga de vectorizadores
##### Crear Pipelines.py
Archivo de python que utilizando todas las herramientas anteriores, crea los pipelines para cada uno de los modelos.
#### api_aplicacion.py
Api de la aplicación que usando las funciones del modulo FuncionesApi crea los respectivos endpoints. 
#### FuncionesApi.py
Modulo que usando las funciones del modulo "usar_pipelines.py", crea las funciones finales que va utilizar la Api de nuestra aplicación.
### Interfaz Gráfica
- .streamlit -> Configuración de interfaz de la pagina web
- static -> Distintas fuentes de letras
- Interfaz_aplicacion.py -> Código de streamlit para interfaz gráfica
- ArchivoEjemplo.csv -> Archivo de ejemplo para probar aplicación

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

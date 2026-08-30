# clasificacion-multicategoria-resenas-espanol

## Desarrollo de modelos

### librerías
- python
- cuml
- cupy
- torch
- sentence-transformers
- symspellpy
- scikit-learn
- pandas
- spacy
- datasets
- scipy
- matplotlib
- es-core-news-md
- joblib 
- watermark 

### Descripción
Modelos de Machine Learning para clasificar reseñas en español en tres categorías de sentimiento: positiva, neutral y negativa.

### Objetivo
Crear una aplicación que clasifique las reseñas de un documento CSV en 3 clases y muestre las distribución de cada una de sus clases en un gráfico de pastel

### Dataset
amazon_reviews_multi_es -> Dataset de 200.000 reseñas en español de productos, las reseñas están clasificadas en 5 clases. Además de 5.000 Reseñas de testeo y 5.000 Reseñas de validación

### Limpieza
#### Corrección Ortográfica y Sin corrección Ortográfica
##### Embeddings
Limpieza básica que implica convertir todo el texto a minúsculas, eliminación de etiquetas HTML, eliminación de URL y eliminación de espacios extra.
##### TF-IDF
Se aplica la limpieza para embeddings, además de eliminación de palabras de parada y lematización utilizando el modelo de Spacy "es_core_news_md"

### Vectorización
#### Embeddings
Se aplica vectorización para los textos con y sin corrección ortográfica que recibieron la limpieza para Embeddings utilizando el modelo "multilingual-e5-base".
#### TF-IDF
Se aplica vectorización para los textos con y sin corrección ortográfica que recibieron la limpieza para TF-IDF utilizando el vectorizador TF-IDF de sklearn. La vectorización se aplica con diferentes configuraciones en sus parámetros en "sublinear_tf", "ngram_range" y "max_features" para conocer la mejor configuración. Obteniendo de esa manera 240 vectorizadores TF-IDF.

### Modelos Experimentales 
#### Embeddings
Se entrenan tres tipos de modelos (Regresión logística, Clasificador Lineal de Vectores de Soporte y Naive Bayes Gausiano), con los vectores de  Embeddings, tanto los de corrección ortográfica como los que no tienen corrección ortográfica, y se muestran en una tabla ordenados en base a su mejor Exactitud. Obteniendo 6 modelos en total.

#### TF-IDF
Se entrenan cuatro tipos de modelos (Regresión logística, Clasificador Lineal de Vectores de Soporte, Naive Bayes Multinomial y Naive Bayes Complement) con cada uno de las vectorizaciones realizadas en "Vectorización". Los resultados se organizan en una tabla en base a su exactitud, lo que nos permite observar la mejor configuración para TF-IDF. Obteniendo 240 modelos en total.

### Modelos Finales 
#### Entrenamiento con Ajuste de Hiperparametros
Se escoge la mejor configuración para los modelos basados en Embeddings y TF-IDF. Luego usando "GridSearchCV" de Sklearn ajustamos los hiperparámetros de los modelos para mejorar su rendimiento. 

#### Métricas
Se hace una comparación de ambos modelos en sus métricas utilizando el conjunto de datos de testeo, validación y entrenamiento (Para verificar sobreajuste)

#### Guardado de modelos
Se guardan los modelos obtenidos, tanto con TF-IDF como con Embeddings

### Modelo Ensamble

#### Creación de modelo Ensamble
Una vez guardados ambos modelos basados en TF-IDF y en Embeddings, se fusionan para crear el modelo Ensamble. utilizando diferentes pesos (Importancia) para cada modelo.

#### Métricas
Usando el conjunto de validación se testea el modelo ensamble y se compara con los resultados de los modelos independientes.

Una vez que ya obtenemos los 3 modelos y los vectorizadores, podemos trasladar estos a el apartado de aplicación para su respectivo uso.

## Autor
Manuel Elias Orellana Lavayen 

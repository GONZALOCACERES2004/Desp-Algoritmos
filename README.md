# Análisis de Sentimientos y Despliegue de API

## Descripción del Proyecto

Este proyecto tiene como objetivo realizar un análisis de sentimientos sobre reseñas de productos utilizando técnicas de procesamiento de lenguaje natural (NLP) y modelos de aprendizaje automático. Se implementa un modelo de regresión logística para clasificar las reseñas en positivas o negativas. Además, se utiliza MLflow y se despliega una API utilizando FastAPI.

## Estructura del Proyecto

La estructura del proyecto incluye los siguientes archivos y carpetas:

/proyecto
│
├── masterclass_despliegue_practica.py # Archivo principal que contiene el flujo del análisis y entrenamiento del modelo.
├── app_fastapi.py # Archivo que define la API utilizando FastAPI.
├── datos_extraidos.csv # Conjunto de datos original con las reseñas.
├── datos_procesados.csv # Conjunto de datos procesados listos para el modelado.
└── util # Carpeta que contiene funciones auxiliares para el preprocesamiento y análisis.
text

## Archivos Principales

### `masterclass_despliegue_practica.py`

Este archivo es el núcleo del proyecto. Aquí se realiza:

- **Carga y preprocesamiento de datos**: Se cargan las reseñas desde un archivo CSV, se limpian y se preparan para el análisis.
- **Entrenamiento del modelo**: Se entrena un modelo de regresión logística utilizando TF-IDF para vectorizar el texto.
- **Evaluación del modelo**: Se evalúa el rendimiento del modelo y se registran las métricas utilizando MLflow.

### `app_fastapi.py`

Este archivo define la API RESTful utilizando FastAPI. Las características clave incluyen:

- **Clasificación de texto**: Permite a los usuarios enviar reseñas y recibir la clasificación (positiva o negativa).
- **Generación de texto**: Ofrece la capacidad de generar texto basado en un prompt dado.
- **Funciones adicionales**: Incluye endpoints para generar números aleatorios, multiplicar dos números y generar una secuencia de Fibonacci.

## Importancia de MLflow

MLflow es una herramienta esencial en este proyecto, ya que permite:

- **Seguimiento de Experimentos**: Registra automáticamente los parámetros, métricas y artefactos generados durante el entrenamiento del modelo, lo que facilita la comparación entre diferentes ejecuciones.
- **Gestión de Modelos**: Permite guardar y versionar modelos entrenados, facilitando su implementación en producción.
- **Visualización**: Proporciona una interfaz web para visualizar resultados y métricas, ayudando en la toma de decisiones sobre qué modelo utilizar.

## Importancia de FastAPI

FastAPI es un marco moderno y rápido para construir APIs con Python. Sus ventajas incluyen:

- **Rendimiento**: FastAPI es altamente eficiente gracias a su diseño asíncrono, lo que permite manejar múltiples solicitudes simultáneamente.
- **Facilidad de Uso**: Su sintaxis intuitiva permite desarrollar APIs rápidamente con validación automática de datos utilizando Pydantic.
- **Documentación Automática**: Genera automáticamente documentación interactiva (Swagger UI) para facilitar la prueba y exploración de los endpoints.

## Cómo Ejecutar el Proyecto

1. **Instalación de Dependencias**:
   Asegúrate de tener Python instalado y ejecuta:
pip install -r requirements.txt
text

2. **Ejecutar el Archivo Principal**:
Ejecuta el archivo principal para entrenar el modelo:
python masterclass_despliegue_practica.py
text

3. **Iniciar la API**:
En otro terminal, ejecuta:
uvicorn app_fastapi:app --reload
text

4. **Acceder a la Documentación**:
Visita `http://127.0.0.1:8000/docs` en tu navegador para acceder a la documentación interactiva de la API.

## Conclusión

Este proyecto combina técnicas avanzadas de análisis de sentimientos con una interfaz accesible a través de una API moderna. Gracias a herramientas como MLflow y FastAPI, se logra no solo un análisis efectivo sino también una implementación práctica que permite a otros interactuar con los modelos entrenados fácilmente.

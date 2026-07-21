# Detector de Chollos en Alojamientos de Andalucía

Sistema de aprendizaje automático que detecta alojamientos con precios anómalamente bajos ("chollos") en Booking.com para las provincias de Andalucía. Combina scraping web, preprocesamiento de datos y un modelo de regresión que predice el precio justo de cada alojamiento, con una interfaz web en Streamlit para el usuario final.

---

## Tecnologías

| Área | Herramientas |
|---|---|
| Lenguaje | Python 3.12 |
| Scraping | Selenium, Playwright, BeautifulSoup4, cloudscraper |
| Datos | pandas, numpy |
| Machine Learning | scikit-learn, XGBoost |
| Visualización | seaborn, matplotlib, plotly |
| Interfaz web | Streamlit |
| Gestión de entorno | uv |
| Análisis estadístico | R 4.x, corrplot, arrow |

---

## Estructura del proyecto

```
TFG/
├── TFG_Chollos/
│   ├── Scraping/
│   │   ├── Generador_urls_generales.py
│   │   ├── Scrp_estancias_provincias.py         # Scraping de listados (Selenium)
│   │   ├── Scrp_caracteristicas_estancias.py    # Scraping de fichas (Playwright)
│   │   ├── obtener_coordenadas_centros.py        # Geocodificación de localidades (Nominatim)
│   │   └── patch_room_size.py                    # Extracción del tamaño de habitación (m²) desde los CSVs ya scrapeados
│   ├── Cleaning/
│   │   ├── preprocessing.py   # Limpieza y estructuración del dataset
│   │   ├── post_analisis.py   # Eliminación de outliers
│   │   └── encoding.py        # Codificación para ML
│   ├── Modelizacion/
│   │   ├── Modelizacion.py             # Entrenamiento, selección y evaluación de los modelos de regresión
│   │   ├── transformaciones.py         # Partición y escalado del dataset
│   │   ├── categorizacion.py           # Etiquetado de categorías de chollo por umbrales sobre precio_real/precio_predicho
│   │   └── podar_bosque_aleatorio.py   # Reduce el Random Forest ganador a 20 árboles para que quepa en el repo y en memoria en Streamlit Community Cloud
│   ├── Graficos/
│   │   ├── Grafico_Alojamientos_Andalucia.py  # Mapa coroplético de alojamientos
│   │   ├── Graficos.py                        # Distribución de categorías de chollo en el conjunto de test
│   │   ├── preview_datasets.py                # Imágenes de previsualización de los datasets RAW/PROCESSED
│   │   └── correlacion.R                      # Matriz de correlación (Pearson)
│   ├── App/                   # Aplicación Streamlit (uso local)
│   │   ├── main.py                # Página principal (mapa + gráficos)
│   │   ├── run.py                 # Lanzador de la app
│   │   ├── graficos_analisis.py
│   │   ├── _predictor.py          # Lógica de predicción
│   │   ├── _scraper_app.py        # Scraping en tiempo real
│   │   ├── _feature_engineering.py  # Preprocesado de datos para predicción
│   │   └── pages/
│   │       └── Busqueda.py    # Página de búsqueda de chollos
│   ├── App_Cloud/              # Misma app, adaptada a Streamlit Community Cloud
│   │   ├── main.py                # Usa el modelo podado, incluido en el repo (carga desde disco, sin red)
│   │   ├── run.py
│   │   ├── graficos_analisis.py
│   │   ├── _predictor.py
│   │   ├── _scraper_app.py
│   │   ├── _feature_engineering.py
│   │   └── pages/
│   │       └── Busqueda.py
│   └── utils.py
├── data/
│   ├── raw/                   # Datos en bruto del scraping
│   ├── processed/             # Datos limpios y codificados
│   ├── models/                # Modelos entrenados (.pkl)
│   └── resultados/            # Resultados de evaluación
├── packages.txt                # Dependencias del sistema para el despliegue (chromium, para Playwright)
└── pyproject.toml
```

> `App_Cloud/` existe porque el Random Forest completo (~1,4 GB en memoria) supera el límite del plan gratuito de Streamlit Community Cloud. La solución inicial fue alojar el modelo completo en Hugging Face Hub y descargarlo al arrancar, pero eso no resolvía el pico de memoria al deserializarlo. La solución final poda el bosque a 20 árboles (`podar_bosque_aleatorio.py`, ~373 MB en memoria, ~59 MB comprimido, pérdida de rendimiento mínima), lo que permite incluirlo directamente en el repositorio y cargarlo desde disco sin depender de red.

---

## Análisis de correlación (R)

El script `TFG_Chollos/Graficos/correlacion.R` genera la matriz de correlación lineal (Pearson) sobre el dataset codificado y guarda el resultado en `images/correlacion_lineal.png`.

### Prerrequisitos R

- [R 4.x](https://cran.r-project.org/)
- Paquetes (se instalan automáticamente la primera vez que se ejecuta el script):
  - `arrow` — lectura de ficheros `.parquet`
  - `corrplot` — visualización de matrices de correlación

### Ejecución

```r
# Desde RStudio: abrir TFG_Chollos/Graficos/correlacion.R y pulsar Source
# O desde terminal:
Rscript TFG_Chollos/Graficos/correlacion.R
```

---

## Instalación

### Prerrequisitos

- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/) instalado globalmente

### Pasos

**1. Crear entorno virtual e instalar dependencias**

```powershell
uv venv --python 3.12
uv sync
```

**2. Instalar el paquete en modo editable**

Permite importar los módulos propios del proyecto desde cualquier fichero sin reinstalar tras cada cambio.

```powershell
uv pip install -e .
```

**3. Instalar Chromium para Playwright**

```powershell
playwright install chromium
```

> El driver de Selenium (ChromeDriver) se gestiona automáticamente mediante `webdriver-manager` y no requiere instalación manual.

**4. Configurar variables de entorno**

Crea un fichero `.env` en la raíz del proyecto (`TFG/`) con la ruta absoluta al paquete:

```
BASE=Disco:/ruta/al/proyecto/TFG/TFG_Chollos
```

Ejemplo en Windows:

```
BASE=C:/Users/usuario/OneDrive/TFG/TFG_Chollos
```

---

## Uso

### Ejecutar la aplicación web

```powershell
uv run python TFG_Chollos/App/run.py
```

O directamente con Streamlit:

```powershell
streamlit run TFG_Chollos/App/main.py
```

> Los modelos ya están entrenados e incluidos en `data/models/`. No es necesario ejecutar el pipeline de datos para usar la app.

`TFG_Chollos/App_Cloud/` contiene la misma aplicación adaptada al plan gratuito de Streamlit Community Cloud (modelo podado descargado desde Hugging Face Hub); no está pensada para ejecutarse en local.

### Pipeline de datos completo (opcional — solo si se quiere reentrenar desde cero)

```powershell
# 1. Scraping de listados por provincia (Selenium)
uv run python TFG_Chollos/Scraping/Scrp_estancias_provincias.py

# 2. Scraping de fichas individuales (Playwright)
uv run python TFG_Chollos/Scraping/Scrp_caracteristicas_estancias.py

# 3. Geocodificación de localidades (ejecutar una vez antes de preprocessing)
uv run python TFG_Chollos/Scraping/obtener_coordenadas_centros.py

# 4. Limpieza y estructuración
uv run python TFG_Chollos/Cleaning/preprocessing.py

# 5. Eliminación de outliers
uv run python TFG_Chollos/Cleaning/post_analisis.py

# 6. Codificación para ML (genera encoders en data/models/)
uv run python TFG_Chollos/Cleaning/encoding.py

# 7. Entrenamiento de modelos
uv run python TFG_Chollos/Modelizacion/Modelizacion.py

# 8. (Opcional) Reducir el modelo ganador para desplegarlo en Streamlit Community Cloud
uv run python TFG_Chollos/Modelizacion/podar_bosque_aleatorio.py
```

---

## Metodología ML

### Datos

Los datos se obtienen mediante scraping de Booking.com para todas las provincias de Andalucía. Por cada alojamiento se extraen:
- Información general: título, tipo, estrellas, valoración, número de valoraciones
- Ubicación: coordenadas, distancia al centro de la localidad
- Servicios y amenities de la habitación más económica
- Calendario de disponibilidad con precio por día

### Preprocesamiento

- Limpieza y gestión de valores nulos
- Eliminación de outliers por criterio de Tukey
- Codificación: `LabelEncoder` (localidad, provincia) y `get_dummies` (tipo)
- Escalado: centrado, normalización (MinMaxScaler) y estandarización (StandardScaler)
- Ingeniería de variables: `es_finde`, `es_domingo`, `dias_restantes`, distancia al centro

### Partición y validación

Se aplica una estrategia combinada de **holdout 70/15/15** y **Cross-Validation K-Fold (K=3)**:

```
X_train (70%) → GridSearchCV con K-Fold cv=3 → ajuste de hiperparámetros
X_val   (15%) → comparación y selección del modelo final
X_test  (15%) → evaluación final sin sesgo (se usa una única vez)
```

### Modelos entrenados

| Tipo | Modelos |
|---|---|
| Regresión (precio) | Regresión Lineal, Árbol de Decisión, Random Forest, XGBoost |

Los modelos de SVM, KNN y Redes Neuronales fueron descartados por tiempo de entrenamiento excesivo (>1h por fit). Los hiperparámetros se optimizan con `GridSearchCV`, seleccionando el mejor modelo por R² en validación.

### Resultados

Métricas sobre el conjunto de **validación**, usadas para comparar los 4 modelos y elegir el ganador (evaluarlos en test invalidaría la evaluación final, que debe usarse una única vez):

| Modelo | R² | MAE | RMSE |
|---|---|---|---|
| Regresión Lineal | 0.1438 | 63.33€ | 84.04€ |
| XGBoost | 0.5063 | 45.53€ | 63.82€ |
| Árbol de Decisión | 0.5930 | 38.22€ | 57.95€ |
| **Random Forest (ganador)** | **0.8670** | **19.42€** | **33.12€** |

Evaluado una única vez sobre el conjunto de **prueba** independiente, el Random Forest ganador alcanza R²=0.8676, MAE=19.43€, RMSE=33.05€ — sin indicios de sobreajuste entre validación y prueba.

### Categorización de chollos

No se entrena un clasificador aparte: una vez elegido el modelo de regresión ganador, cada alojamiento se etiqueta por umbrales sobre el ratio `precio_real / precio_predicho` (`Modelizacion/categorizacion.py`):

| Categoría | Condición del ratio |
|---|---|
| Hiper chollo | ≤ 0.75 |
| Super chollo | 0.75 – 0.85 |
| Chollo | 0.85 – 0.97 |
| Normal | 0.97 – 1.03 |
| Inflado | > 1.03 |

---

## Autor

Mario — Trabajo de Fin de Grado, Universidad de Sevilla

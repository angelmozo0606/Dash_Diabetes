# GlucoRisk — Predicción de diabetes

Dashboard interactivo en **Dash + Plotly** para explorar los indicadores de salud del BRFSS 2015 (CDC) y estimar el riesgo de diabetes o prediabetes con seis modelos de clasificación.

## Páginas

| Ruta | Contenido |
| --- | --- |
| `/` | Presentación, curva de prevalencia por edad, equipo y video |
| `/contexto` | Problema, fuente, diccionario de variables y métricas |
| `/exploratorio` | Univariado, bivariado (χ², Mann-Whitney, V de Cramér, Spearman) y pipeline ETL |
| `/modelos` | Tabla de métricas, curvas ROC, sobreajuste, matrices de confusión, importancia y odds ratios |
| `/prediccion` | Formulario de perfil, riesgo estimado, explicación local y comparación entre modelos |
| `/aplicaciones` | Casos de uso y límites del modelo |

## Modelos

Entrenados con 80/20 estratificado y validación cruzada de 5 folds (`train_models.py`):

| Modelo | ROC-AUC | Recall | F1 |
| --- | --- | --- | --- |
| XGBoost | 0.824 | 0.800 | 0.764 |
| Random Forest | 0.821 | 0.796 | 0.760 |
| Regresión Logística | 0.817 | 0.773 | 0.755 |
| Árbol de Decisión | 0.811 | 0.791 | 0.751 |
| KNN | 0.805 | 0.790 | 0.753 |
| Naive Bayes | 0.780 | 0.718 | 0.722 |

## Ejecución local

```bash
pip install -r requirements.txt
python train_models.py   # opcional: regenera artifacts/
python app.py
```

Abre <http://localhost:8050>. Si la carpeta `artifacts/` no existe o no es compatible con tu versión de scikit-learn, la app reentrena los modelos al iniciar (≈1–2 min).

## Despliegue en Google Cloud Run

```bash
gcloud auth login
gcloud config set project TU_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

gcloud run deploy dash-diabetes \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --port 8080
```

## Estructura

```
dash-diabetes/
├── app.py                 # App principal
├── train_models.py        # Entrenamiento y métricas
├── artifacts/             # Modelos (.joblib) y métricas (.json)
├── data/                  # CSV BRFSS 2015 balanceado 50/50
├── assets/style.css       # Estilos (Dash los carga automáticamente)
├── notebooks/             # EDA univariado y bivariado
├── requirements.txt
├── Dockerfile
└── Procfile
```

## Personalizar

En la parte superior de `app.py`:

- `TEAM`: nombres y enlaces de GitHub/LinkedIn del equipo.
- `VIDEO_URL`: enlace de vista previa del video (por ejemplo `https://drive.google.com/file/d/<ID>/preview`).

## Datos

[Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) — `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`. Resultado con fines académicos; no es una herramienta de diagnóstico.

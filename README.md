# VISUALENCE CV · Consigue Ventas
Prototipo de análisis y predicción del rendimiento de contenido digital (ML + Deep Learning + visión + Java).

## Estructura
```
visualence_pro/
├── README.md · requirements.txt · requirements-dl.txt · run_all.py · app.py
├── 01_limpiar.py            CSV crudo → limpio (duplicados, formatos, rangos, fechas)
├── 02_dividir.py            split TEMPORAL train/val/test + umbrales con train + AUDITORÍA anti-fuga
├── 03_eda_estadistica.py    EDA, ANOVA/Kruskal/chi²/t-test/Mann-Whitney, correlación, NLP (solo train)
├── 04_ml_clasico.py         6 modelos + GridSearch (TimeSeriesSplit) + permutación + curva de aprendizaje
├── 05_regresion.py          regresión lineal, Ridge, Random Forest (tasa de interacción)
├── 06_segmentacion.py       K-Means (silhouette) + validación en test
├── 07_deep_learning.py      misma red en Keras (TensorFlow) y PyTorch
├── 08_reporte.py            diagramas, matrices de confusión, ROC, curvas e informe técnico
├── 10_features_imagenes.py  utilidad: características de imágenes reales con OpenCV
├── vlc/                     config.py · features.py · utils.py · vision.py
├── data/raw | processed | splits
├── models/                  modelo_final.pkl · regresion.pkl · segmentador.pkl · umbrales.json · (DL)
├── reports/                 metricas.json · informe_tecnico.md · figures/ · preds/
├── java/                    Recursividad.java · ArbolDecision.java
└── utils_extra/generar_dataset.py   (genera un CSV crudo sintético)
```

## Instalación (VS Code → Terminal, dentro de la carpeta)
Python **3.10 – 3.12** recomendado (TensorFlow aún no soporta versiones más nuevas).
```
pip install -r requirements.txt
pip install -r requirements-dl.txt      # opcional: Keras y PyTorch (pesado)
```

## Ejecución
```
python run_all.py          # corre los pasos 01 a 08
streamlit run app.py       # abre la app en http://localhost:8501
```
Pasos sueltos: `python 01_limpiar.py`, `python 02_dividir.py`, … en orden numérico.
Java (JDK 11+): `cd java` y luego `java Recursividad.java` y `java ArbolDecision.java ../data/splits`.

## Cómo se evita la fuga de datos (data leakage)
| Riesgo | Control |
|---|---|
| Usar resultados (alcance, clics, leads…) para predecir | `FEATURES_PRE` vs `RESULTADOS` en `vlc/config.py` + `verificar_sin_fuga()` falla si se cuelan |
| Mezclar pasado y futuro | División **cronológica** 70/15/15 |
| Etiquetas con información del test | Umbrales alto/medio/bajo = terciles de **train** |
| Preprocesado que "ve" test | Imputación, escalado, one-hot y TF-IDF dentro de `Pipeline`, ajustados solo con train |
| Elegir modelo mirando test | Tuning con `TimeSeriesSplit` en train; selección con **val**; test se evalúa una vez |
| Mismo post en train y test | Eliminación de duplicados + auditoría de ids |
| Desfase entrenamiento/app | `features_texto()` es la misma función en ambos |
| Fuga oculta | Prueba de cordura: con etiquetas barajadas el F1 debe ser ≈ 0.33 |

## Datos reales
1. Exporta publicaciones de la agencia con las mismas columnas que `data/raw/publicaciones_crudo.csv`.
2. Para las imágenes: `python 10_features_imagenes.py carpeta_imagenes` (archivos `<id_post>.jpg`) y une `features_imagen.csv` por `id_post`.
3. Reemplaza el CSV en `data/raw/` y ejecuta `python run_all.py`.

## Limitaciones
- El dataset incluido es **sintético**: valida el flujo, no sirve para conclusiones de negocio.
- `pct_texto_en_img` calculado con OpenCV es una **aproximación**, no un OCR.
- Con pocos datos, las diferencias entre modelos caen dentro del ruido (ver IC bootstrap en el informe).

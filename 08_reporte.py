"""PASO 8 · Figuras (matrices de confusión, ROC, curvas, diagramas) e informe técnico, generados desde los resultados guardados."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from vlc.config import *
from vlc.utils import leer_metricas, guardar_fig, slug

sns.set_theme(style="whitegrid")
M = leer_metricas()
ML, DL = M["ml"]["resultados"], M.get("dl", {})
DL_OK = {k: v for k, v in DL.items() if isinstance(v, dict) and "test" in v}
mejor_ml = M["ml"]["mejor"]


# ------------------------------------------------------------------ diagramas
def caja(ax, x, y, w, h, texto, color="#dbe9f6", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=color, ec="#33506b", lw=1.2))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs)

def flecha(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color="#33506b", lw=1.2))

def diagrama_pipeline():
    fig, ax = plt.subplots(figsize=(14, 7)); ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
    caja(ax, 0.2, 5.6, 2.4, 1.0, "CSV crudo\n(publicaciones)", "#f6e8c8")
    caja(ax, 3.4, 5.6, 3.0, 1.0, "01 Limpieza\nduplicados · formatos · rangos\n(sin estadísticos globales)")
    caja(ax, 7.2, 5.6, 3.4, 1.0, "02 División TEMPORAL 70/15/15\numbrales alto/medio/bajo\ncon TRAIN + auditoría anti-fuga", "#d9f0d9")
    flecha(ax, 2.6, 6.1, 3.4, 6.1); flecha(ax, 6.4, 6.1, 7.2, 6.1)
    caja(ax, 0.6, 3.9, 2.6, 0.9, "TRAIN (pasado)\najusta todo", "#cfe8cf"); caja(ax, 4.2, 3.9, 2.6, 0.9, "VAL\nselecciona modelo", "#fbe9b7")
    caja(ax, 7.8, 3.9, 2.6, 0.9, "Mejor modelo\n(según VAL)", "#fbe9b7")
    caja(ax, 11.2, 3.9, 2.6, 0.9, "TEST (futuro)\n1 sola evaluación", "#f4c7c3")
    for x in (1.9, 5.5, 12.5): flecha(ax, 8.9, 5.6, x, 4.8)
    flecha(ax, 6.8, 4.35, 7.8, 4.35); flecha(ax, 10.4, 4.35, 11.2, 4.35)
    caja(ax, 0.2, 2.2, 3.4, 1.1, "Preprocesado en Pipeline\nimputar → escalar → one-hot → TF-IDF\n(fit SOLO con train)", "#e6dcf3")
    flecha(ax, 1.9, 3.9, 1.9, 3.3)
    caja(ax, 4.2, 1.1, 2.0, 1.0, "ML clásico\n6 modelos + GridSearch\n(TimeSeriesSplit)", "#dbe9f6", 8)
    caja(ax, 6.5, 1.1, 1.8, 1.0, "Regresión\nlineal/Ridge/RF", "#dbe9f6", 8)
    caja(ax, 8.6, 1.1, 2.2, 1.0, "Deep Learning\nKeras · PyTorch", "#dbe9f6", 8)
    caja(ax, 11.1, 1.1, 1.8, 1.0, "K-Means\nsegmentos", "#dbe9f6", 8)
    flecha(ax, 3.6, 2.7, 5.2, 2.1); flecha(ax, 3.6, 2.7, 7.4, 2.1); flecha(ax, 3.6, 2.7, 9.7, 2.1); flecha(ax, 3.6, 2.7, 12.0, 2.1)
    caja(ax, 11.2, 2.5, 2.6, 0.9, "08 Reporte + App\nStreamlit", "#f6e8c8"); flecha(ax, 12.5, 3.9, 12.5, 3.4)
    caja(ax, 4.2, 0.0, 6.6, 0.7, "Prueba de cordura: con etiquetas barajadas el F1 debe ≈ azar (0.33)", "#f4c7c3", 8)
    ax.set_title("VISUALENCE CV · Flujo del proyecto y controles contra fuga de datos", fontsize=12)
    guardar_fig(fig, "diagrama_pipeline")

def diagrama_split():
    s = M["split"]; fig, ax = plt.subplots(figsize=(11, 2.3)); ax.axis("off")
    tot = sum(s["filas"].values()); x = 0
    for k, c in [("train", "#7fc97f"), ("val", "#fdc086"), ("test", "#f4a6a0")]:
        w = s["filas"][k] / tot
        ax.add_patch(plt.Rectangle((x, 0.3), w, 0.5, fc=c, ec="k"))
        ax.text(x + w / 2, 0.55, f"{k.upper()}\n{s['filas'][k]} publicaciones", ha="center", va="center", fontsize=9)
        ax.text(x + w / 2, 0.15, f"{s['rango_fechas'][k][0]} → {s['rango_fechas'][k][1]}", ha="center", fontsize=8)
        x += w
    ax.annotate("", xy=(1, 0.95), xytext=(0, 0.95), arrowprops=dict(arrowstyle="->")); ax.text(0.5, 1.0, "tiempo", ha="center", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.15); ax.set_title("División temporal (el modelo nunca ve el futuro)")
    guardar_fig(fig, "diagrama_split")

def diagrama_red():
    fig, ax = plt.subplots(figsize=(12, 2.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 2.6); ax.axis("off")
    capas = [("Entrada\n(variables\npre-publicación)", "#f6e8c8"), ("Dense 128\nReLU", "#dbe9f6"), ("Dropout\n0.3", "#eeeeee"),
             ("Dense 64\nReLU", "#dbe9f6"), ("Dropout\n0.3", "#eeeeee"), ("Softmax 3\nbajo·medio·alto", "#d9f0d9")]
    for i, (t, c) in enumerate(capas):
        caja(ax, 0.2 + i * 1.95, 0.7, 1.55, 1.2, t, c, 8.5)
        if i: flecha(ax, 0.2 + i * 1.95 - 0.4, 1.3, 0.2 + i * 1.95, 1.3)
    ax.set_title("Arquitectura de la red (idéntica en Keras y PyTorch)"); guardar_fig(fig, "diagrama_red")


# ------------------------------------------------------------------ resultados
def cargar_preds(split="test"):
    out = {}
    for f in sorted(PREDS.glob(f"*__{split}.csv")):
        if f.name.startswith(("ml_", "dl_")):
            d = pd.read_csv(f); nombre = d["modelo"].iloc[0]
            out[nombre[3:] + (" (DL)" if nombre.startswith("dl_") else "")] = d
    return out

def matriz_ax(ax, d, titulo):
    m = confusion_matrix(d.y_true, d.y_pred, labels=NIVELES); pct = m / m.sum(1, keepdims=True).clip(1)
    txt = np.array([[f"{m[i, j]}\n({pct[i, j]:.0%})" for j in range(3)] for i in range(3)])
    sns.heatmap(pct, annot=txt, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax, xticklabels=NIVELES, yticklabels=NIVELES)
    ax.set_title(titulo, fontsize=9); ax.set_xlabel("Predicho"); ax.set_ylabel("Real")

def figs_resultados():
    P = {k: v for k, v in cargar_preds().items() if "base" not in k.lower()}
    n = len(P); cols = 3; filas = int(np.ceil(n / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(4.4 * cols, 4.4 * filas), constrained_layout=True); axs = np.atleast_1d(axs).ravel()
    for ax, (k, d) in zip(axs, P.items()):
        matriz_ax(ax, d, f"{k} · F1={f1_score(d.y_true, d.y_pred, average='macro'):.2f}")
    for ax in axs[n:]: ax.axis("off")
    fig.suptitle("Matrices de confusión en TEST (fila = real; % por fila)"); guardar_fig(fig, "matrices_confusion_test")

    fig, ax = plt.subplots(figsize=(5.5, 4.6)); matriz_ax(ax, cargar_preds()[mejor_ml], f"Mejor ML (según VAL): {mejor_ml}")
    guardar_fig(fig, "matriz_mejor_modelo")

    filas_c = [("Línea base", ML["Línea base"])] + [(k, v) for k, v in ML.items() if k != "Línea base"] + \
              [(f"{k} (DL)", v) for k, v in DL_OK.items()]
    df = pd.DataFrame([{"Modelo": k, "Validación": v["val"]["f1_macro"], "Test": v["test"]["f1_macro"]} for k, v in filas_c]).melt("Modelo", var_name="Conjunto", value_name="F1 macro")
    fig, ax = plt.subplots(figsize=(10, 4.5)); sns.barplot(data=df, x="Modelo", y="F1 macro", hue="Conjunto", ax=ax)
    ax.axhline(1 / 3, color="r", ls="--", lw=1); ax.text(-0.4, 0.34, "azar ≈ 0.33", color="r", fontsize=8)
    plt.xticks(rotation=25, ha="right"); ax.set_title("Comparación de modelos (F1 macro)"); guardar_fig(fig, "comparacion_modelos")

    # ROC one-vs-rest del mejor ML (y mejor DL si existe)
    comp = {mejor_ml: cargar_preds()[mejor_ml]}
    if DL_OK:
        mdl = max(DL_OK, key=lambda k: DL_OK[k]["val"]["f1_macro"]); comp[f"{mdl} (DL)"] = cargar_preds()[f"{mdl} (DL)"]
    fig, axs = plt.subplots(1, len(comp), figsize=(5.2 * len(comp), 4.4)); axs = np.atleast_1d(axs)
    for ax, (k, d) in zip(axs, comp.items()):
        for n_ in NIVELES:
            fpr, tpr, _ = roc_curve(d.y_true == n_, d[f"p_{n_}"]); ax.plot(fpr, tpr, label=f"{n_} (AUC={auc(fpr, tpr):.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_title(f"ROC (uno vs resto) · {k}"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    guardar_fig(fig, "roc_test")

    imp = pd.Series(M["ml"]["importancia_permutacion"]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.5)); imp.plot.barh(ax=ax, color="steelblue")
    ax.set_title("Importancia por permutación (caída de F1 en VAL)"); guardar_fig(fig, "importancia_variables")

    cu = M["ml"]["curva_aprendizaje"]; fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cu["tam"], cu["train"], "o-", label="train"); ax.plot(cu["tam"], cu["cv"], "o-", label="validación cruzada temporal")
    ax.set_xlabel("Tamaño de entrenamiento"); ax.set_ylabel("F1 macro"); ax.set_title(f"Curva de aprendizaje · {mejor_ml}"); ax.legend()
    guardar_fig(fig, "curva_aprendizaje")

    if DL_OK:
        fig, axs = plt.subplots(len(DL_OK), 2, figsize=(10, 3.6 * len(DL_OK))); axs = np.atleast_2d(axs)
        for i, (k, v) in enumerate(DL_OK.items()):
            h = v["historial"]
            axs[i, 0].plot(h["loss"], label="train"); axs[i, 0].plot(h["val_loss"], label="val"); axs[i, 0].set_title(f"{k} · pérdida"); axs[i, 0].legend()
            axs[i, 1].plot(h["acc"], label="train"); axs[i, 1].plot(h["val_acc"], label="val"); axs[i, 1].set_title(f"{k} · exactitud"); axs[i, 1].legend()
        guardar_fig(fig, "curvas_dl")

def figs_regresion():
    R = M["regresion"]; d = pd.read_csv(PREDS / f"reg_{slug(R['mejor'])}__test.csv"); res = d.y_true - d.y_pred
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].scatter(d.y_true, d.y_pred, alpha=.6); lim = [d.y_true.min(), d.y_true.max()]; axs[0].plot(lim, lim, "r--")
    axs[0].set_title(f"Real vs predicho · {R['mejor']}"); axs[0].set_xlabel("Tasa real"); axs[0].set_ylabel("Tasa predicha")
    axs[1].hist(res, bins=20); axs[1].set_title("Residuos"); stats.probplot(res, plot=axs[2]); axs[2].set_title("Q-Q de residuos")
    guardar_fig(fig, "regresion_diagnostico")
    c = pd.Series(R["coeficientes_lineal"]).sort_values(); fig, ax = plt.subplots(figsize=(6.5, 4.5))
    c.plot.barh(ax=ax, color=["#d9534f" if v < 0 else "#5cb85c" for v in c]); ax.set_title("Regresión lineal · coeficientes principales (estandarizados)")
    guardar_fig(fig, "regresion_coeficientes")

def bootstrap_f1(d, B=1000):
    rng = np.random.default_rng(SEED); n = len(d); v = []
    for _ in range(B):
        i = rng.integers(0, n, n); v.append(f1_score(d.y_true.values[i], d.y_pred.values[i], average="macro"))
    return np.percentile(v, [2.5, 97.5])


# ------------------------------------------------------------------ informe
def tabla(df):
    cab = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return cab + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in df.values)

def informe():
    S, Q, A = M["split"], M["calidad_datos"], M["auditoria_fuga"]; est = M["estadistica"]; R = M["regresion"]; sg = M["segmentacion"]
    f = lambda x: f"{x:.3f}"
    tml = pd.DataFrame([{"Modelo": k, "CV F1 (train)": f(v["cv_f1_train"]) if "cv_f1_train" in v else "—", "F1 val": f(v["val"]["f1_macro"]),
                         "F1 test": f(v["test"]["f1_macro"]), "Accuracy test": f(v["test"]["accuracy"]), "AUC test": f(v["test"]["auc_ovr"])}
                        for k, v in ML.items() for k in [k]]).assign(Modelo=list(ML))
    if DL_OK:
        tdl = pd.DataFrame([{"Modelo": k, "Épocas": v["epocas"], "F1 val": f(v["val"]["f1_macro"]), "F1 test": f(v["test"]["f1_macro"]),
                             "Accuracy test": f(v["test"]["accuracy"]), "AUC test": f(v["test"]["auc_ovr"])} for k, v in DL_OK.items()])
        sec_dl = (f"Arquitectura: `{DL['arquitectura']}`. Hiperparámetros: {DL['hiperparametros']}.\n\n{tabla(tdl)}\n\n"
                  "![Curvas](figures/curvas_dl.png)\n")
    else:
        sec_dl = ("*Los modelos de Deep Learning no se ejecutaron en este equipo (falta TensorFlow/PyTorch). "
                  "Instala `requirements-dl.txt`, ejecuta `python 07_deep_learning.py` y vuelve a correr `python 08_reporte.py`.*\n")
    treg = pd.DataFrame([{"Modelo": k, "R² val": f(v["val"]["R2"]), "R² test": f(v["test"]["R2"]), "MAE test": f"{v['test']['MAE']:.4f}",
                          "RMSE test": f"{v['test']['RMSE']:.4f}"} for k, v in R["resultados"].items()])
    ic = bootstrap_f1(cargar_preds()[mejor_ml]); b = ML["Línea base"]["test"]["f1_macro"]
    tests = "\n".join(f"- {k}: estadístico = {v['estadistico']:.2f}, p = {v['p']:.2e}" for k, v in est.items() if k != "spearman_con_tasa")
    aud = "\n".join(f"- {'✅' if v else '❌'} {k}" for k, v in A.items())
    pers = pd.DataFrame(sg["perfil_test"])[["segmento", "publicaciones", "tasa_prom"]]
    md = f"""# Informe técnico · VISUALENCE CV
*Proyecto de prácticas preprofesionales · Consigue Ventas*

## 1. Objetivo
Clasificar el rendimiento esperado de una publicación (**bajo / medio / alto**) y estimar su tasa de interacción **antes de publicarla**,
usando solo información disponible en ese momento (formato, horario, texto e imagen).

## 2. Datos y limpieza
- Filas iniciales: **{Q['filas_iniciales']}** → finales: **{Q['filas_finales']}**. Duplicados eliminados: {Q['duplicados_exactos']}; filas sin copy: {Q['filas_sin_copy']}.
- Nulos restantes (se imputan **dentro del pipeline con estadísticos de train**): {Q['nulos_restantes']}.
- ⚠️ El dataset es **sintético**: sirve para validar el flujo técnico; las conclusiones de negocio requieren datos reales de la agencia.

![Pipeline](figures/diagrama_pipeline.png)

## 3. Prevención de fuga de datos (data leakage)
1. **Solo variables pre-publicación** como entrada; alcance, interacciones, guardados, clics y leads están bloqueados por código (`verificar_sin_fuga`).
2. **División temporal** {S['filas']} (train/val/test): se entrena con el pasado y se evalúa con el futuro.
3. **Umbrales de las clases** calculados solo con train: bajo ≤ {S['umbrales']['bajo_medio']:.4f} < medio ≤ {S['umbrales']['medio_alto']:.4f} < alto.
4. **Preprocesado dentro de un Pipeline**: imputación, escalado, one-hot y TF-IDF se ajustan solo con train.
5. **Ajuste de hiperparámetros** con `TimeSeriesSplit` únicamente sobre train; el mejor modelo se elige con **validación**; **test se usa una sola vez**.
6. **Prueba de cordura**: con etiquetas barajadas el F1 en validación fue **{M['ml']['sanity_etiquetas_barajadas_f1_val']:.3f}** (azar ≈ 0.33), señal de que no hay fuga.

Auditoría automática:
{aud}

![Split](figures/diagrama_split.png)

## 4. Análisis exploratorio y estadístico (solo train)
{tests}

![Formato](figures/eda_formato.png) ![Hora](figures/eda_hora.png)
![Correlación](figures/eda_correlacion.png)
![Imagen](figures/eda_imagen.png)
![Palabras](figures/eda_palabras.png)

## 5. Machine Learning (clasificación)
{tabla(tml)}

Mejor modelo según validación: **{mejor_ml}** · F1 macro en test = **{ML[mejor_ml]['test']['f1_macro']:.3f}**
(IC 95 % bootstrap: {ic[0]:.3f} – {ic[1]:.3f}; línea base = {b:.3f}). Con ~{S['filas']['test']} publicaciones de test, las diferencias
pequeñas entre modelos **no son concluyentes**.

![Comparación](figures/comparacion_modelos.png)
![Matrices](figures/matrices_confusion_test.png)
![ROC](figures/roc_test.png)
![Importancia](figures/importancia_variables.png) ![Aprendizaje](figures/curva_aprendizaje.png)

## 6. Deep Learning (Keras vs PyTorch)
![Red](figures/diagrama_red.png)

{sec_dl}
## 7. Regresión de la tasa de interacción
{tabla(treg)}

Mejor según validación: **{R['mejor']}**. Normalidad de residuos (Shapiro-Wilk, test): p = {R['shapiro_residuos_test']['p']:.3f}.

![Diagnóstico](figures/regresion_diagnostico.png) ![Coeficientes](figures/regresion_coeficientes.png)

## 8. Segmentación (K-Means)
k óptimo por silhouette = **{sg['k']}** (silhouette = {sg['silhouette'][str(sg['k'])]:.3f}, separación débil). Validación en test:

{tabla(pers)}

![Segmentos](figures/seg_clusters.png)

## 9. Visión computacional
OpenCV extrae brillo, contraste, saturación, presencia de rostro (Haar) y una **estimación aproximada** del % de texto en la imagen (`vlc/vision.py`).

## 10. Limitaciones y mejoras
- Dataset sintético y pequeño; reentrenar con 100–300+ publicaciones reales (`10_features_imagenes.py` calcula las variables de imagen).
- Con pocos datos, los modelos simples (logística, árbol, bosque) suelen igualar o superar a las redes neuronales.
- Mejora futura: *embeddings* de imagen con una red preentrenada, seguimiento del modelo en producción y reentrenamiento periódico.
"""
    (REPORTS / "informe_tecnico.md").write_text(md, encoding="utf-8")

diagrama_pipeline(); diagrama_split(); diagrama_red(); figs_resultados(); figs_regresion(); informe()
print("Reporte listo → reports/figures/*.png y reports/informe_tecnico.md")

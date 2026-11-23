"""PASO 3 · EDA, pruebas estadísticas (SciPy) y NLP (NLTK). Se analiza SOLO el conjunto train (no se "espía" val/test)."""
import re
from collections import Counter
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from vlc.config import *
from vlc.utils import cargar_split, guardar_fig, actualizar_metricas

sns.set_theme(style="whitegrid")
df = cargar_split("train")

# ---------- Gráficos exploratorios ----------
fig, ax = plt.subplots(figsize=(5, 3.5))
df["nivel_rendimiento"].value_counts().reindex(NIVELES).plot.bar(ax=ax, color=["#d9534f", "#f0ad4e", "#5cb85c"])
ax.set_title("Distribución de niveles (train)"); ax.set_xlabel(""); plt.xticks(rotation=0); guardar_fig(fig, "eda_niveles")

fig, ax = plt.subplots(figsize=(6, 3.5)); sns.barplot(data=df, x="formato", y="tasa_interaccion", order=FORMATOS, ax=ax)
ax.set_title("Tasa de interacción por formato"); guardar_fig(fig, "eda_formato")

fig, ax = plt.subplots(figsize=(6, 3.5)); sns.lineplot(data=df, x="hora", y="tasa_interaccion", ax=ax)
ax.set_title("Tasa de interacción por hora"); guardar_fig(fig, "eda_hora")

piv = df.pivot_table(index="dia_semana", columns="formato", values="tasa_interaccion").reindex(DIAS)
fig, ax = plt.subplots(figsize=(7, 4)); sns.heatmap(piv, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
ax.set_title("Día × formato (tasa de interacción)"); guardar_fig(fig, "eda_heatmap_dia_formato")

cols_num = NUM + ["tasa_interaccion"]
corr = df[cols_num].corr(method="spearman")
fig, ax = plt.subplots(figsize=(9, 7)); sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
ax.set_title("Correlación de Spearman (variables pre-publicación vs resultado)"); guardar_fig(fig, "eda_correlacion")

fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
sns.boxplot(data=df, x="nivel_rendimiento", y="brillo_img", order=NIVELES, ax=axs[0]); axs[0].set_title("Brillo de la imagen")
sns.boxplot(data=df, x="nivel_rendimiento", y="pct_texto_en_img", order=NIVELES, ax=axs[1]); axs[1].set_title("% texto en imagen")
sns.barplot(data=df, x="tiene_rostro", y="tasa_interaccion", ax=axs[2]); axs[2].set_title("Con rostro (1) vs sin rostro (0)")
guardar_fig(fig, "eda_imagen")

# ---------- Estadística inferencial ----------
res = {}
g = [x["tasa_interaccion"].values for _, x in df.groupby("formato")]
f, p = stats.f_oneway(*g); res["ANOVA tasa ~ formato"] = {"estadistico": float(f), "p": float(p)}
h, p = stats.kruskal(*g); res["Kruskal-Wallis tasa ~ formato"] = {"estadistico": float(h), "p": float(p)}
chi, p, _, _ = stats.chi2_contingency(pd.crosstab(df["formato"], df["nivel_rendimiento"]))
res["Chi² formato vs nivel"] = {"estadistico": float(chi), "p": float(p)}
a, b = df.loc[df.tiene_cta == 1, "tasa_interaccion"], df.loc[df.tiene_cta == 0, "tasa_interaccion"]
t, p = stats.ttest_ind(a, b, equal_var=False); res["t-test CTA vs sin CTA"] = {"estadistico": float(t), "p": float(p)}
u, p = stats.mannwhitneyu(a, b); res["Mann-Whitney CTA vs sin CTA"] = {"estadistico": float(u), "p": float(p)}
rho = {c: float(stats.spearmanr(df[c], df["tasa_interaccion"], nan_policy="omit")[0]) for c in NUM}
res["spearman_con_tasa"] = dict(sorted(rho.items(), key=lambda kv: -abs(kv[1])))
actualizar_metricas("estadistica", res)

# ---------- NLP: palabras por nivel ----------
try:
    from nltk.corpus import stopwords
    SW = set(stopwords.words("spanish"))
except Exception:
    SW = set()
SW |= set("de la el en y a los las un una para con tu tus su que es por al lo se del como más mas o te nos este esta "
          "solo ya hoy ojo mira atención atencion".split())
def top(sub, n=10):
    c = Counter()
    for t in sub["copy"].dropna():
        c.update(w for w in re.findall(r"[a-záéíóúñ]{3,}", t.lower()) if w not in SW)
    return pd.DataFrame(c.most_common(n), columns=["palabra", "frecuencia"])
alto, bajo = top(df[df.nivel_rendimiento == "alto"]), top(df[df.nivel_rendimiento == "bajo"])
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(data=alto, y="palabra", x="frecuencia", color="seagreen", ax=axs[0]); axs[0].set_title("Palabras · rendimiento ALTO")
sns.barplot(data=bajo, y="palabra", x="frecuencia", color="indianred", ax=axs[1]); axs[1].set_title("Palabras · rendimiento BAJO")
guardar_fig(fig, "eda_palabras")
actualizar_metricas("nlp", {"top_alto": alto.values.tolist(), "top_bajo": bajo.values.tolist()})

for k, v in res.items():
    if k != "spearman_con_tasa": print(f"{k}: estadístico={v['estadistico']:.2f}  p={v['p']:.2e}")
print("EDA lista → reports/figures/eda_*.png")

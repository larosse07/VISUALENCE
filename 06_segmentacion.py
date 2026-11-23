"""PASO 6 · Segmentación de publicaciones por similitud (K-Means). Se ajusta con train; test valida si los segmentos generalizan."""
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from vlc.config import *
from vlc.features import construir_preprocesador
from vlc.utils import *

sns.set_theme(style="whitegrid")
tr, te = cargar_split("train"), cargar_split("test")
Xtr, _ = xy(tr); Xte, _ = xy(te)
prep = construir_preprocesador(max_terms=100).fit(Xtr)
Ztr, Zte = prep.transform(Xtr), prep.transform(Xte)

sil = {}
for k in range(2, 7):
    sil[k] = float(silhouette_score(Ztr, KMeans(k, n_init=10, random_state=SEED).fit_predict(Ztr)))
k_best = max(sil, key=sil.get)
km = KMeans(k_best, n_init=10, random_state=SEED).fit(Ztr)
seg = Pipeline([("prep", prep), ("km", km)]); joblib.dump(seg, MODELS / "segmentador.pkl")

tr = tr.assign(segmento=km.labels_); te = te.assign(segmento=km.predict(Zte))
def perfil(d):
    return d.groupby("segmento").agg(publicaciones=("id_post", "count"), tasa_prom=("tasa_interaccion", "mean"),
                                    len_copy=("len_copy", "mean"), pct_cta=("tiene_cta", "mean"),
                                    pct_rostro=("tiene_rostro", "mean"),
                                    formato_top=("formato", lambda s: s.mode().iat[0] if len(s.mode()) else "NA")).round(3)
p_tr, p_te = perfil(tr), perfil(te)
svd = TruncatedSVD(2, random_state=SEED).fit(Ztr); A, B = svd.transform(Ztr), svd.transform(Zte)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].plot(list(sil), list(sil.values()), "o-"); axs[0].axvline(k_best, color="r", ls="--"); axs[0].set_title("Silhouette por k"); axs[0].set_xlabel("k")
axs[1].scatter(A[:, 0], A[:, 1], c=tr.segmento, cmap="viridis", alpha=.6); axs[1].set_title(f"Segmentos en train (k={k_best})")
axs[2].scatter(B[:, 0], B[:, 1], c=te.segmento, cmap="viridis", alpha=.7); axs[2].set_title("Mismos segmentos en test")
guardar_fig(fig, "seg_clusters")

actualizar_metricas("segmentacion", {"k": int(k_best), "silhouette": sil, "perfil_train": p_tr.reset_index().to_dict("records"),
                                     "perfil_test": p_te.reset_index().to_dict("records")})
print("k óptimo:", k_best, sil); print(p_tr); print("\nValidación en test:\n", p_te[["publicaciones", "tasa_prom"]])

"""PASO 2 · División temporal train/val/test + etiquetas con umbrales calculados SOLO con train + auditoría anti-fuga."""
import json
import numpy as np, pandas as pd
from vlc.config import *
from vlc.utils import actualizar_metricas

df = pd.read_csv(DATA_PROC / "publicaciones_limpio.csv", parse_dates=["fecha"])
df = df.sort_values(["fecha", "id_post"]).reset_index(drop=True)

# División CRONOLÓGICA 70/15/15: el modelo se entrena con el pasado y se evalúa con el futuro
n = len(df); i1, i2 = int(n * 0.70), int(n * 0.85)
train, val, test = df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()

# Umbrales de alto/medio/bajo: terciles de TRAIN (nunca de todo el dataset)
q1, q2 = train["tasa_interaccion"].quantile([1 / 3, 2 / 3])
def etiquetar(d):
    d["nivel_rendimiento"] = pd.cut(d["tasa_interaccion"], [-np.inf, q1, q2, np.inf], labels=NIVELES).astype(str)
    return d
train, val, test = etiquetar(train), etiquetar(val), etiquetar(test)
for nombre, d in [("train", train), ("val", val), ("test", test)]:
    d.to_csv(DATA_SPLITS / f"{nombre}.csv", index=False)
(MODELS / "umbrales.json").write_text(json.dumps({"bajo_medio": float(q1), "medio_alto": float(q2)}, indent=2))

# ---- Auditoría anti-fuga ----
q1_chk, q2_chk = train["tasa_interaccion"].quantile([1 / 3, 2 / 3])
auditoria = {
    "Variables de entrada no incluyen resultados (alcance, clics, leads, etc.)": not (set(FEATURES_PRE) & set(RESULTADOS)),
    "Ningún id_post aparece en más de un conjunto": len(set(train.id_post) & set(val.id_post)) == 0
        and len(set(train.id_post) & set(test.id_post)) == 0 and len(set(val.id_post) & set(test.id_post)) == 0,
    "Orden temporal: train ≤ val ≤ test": bool(train.fecha.max() <= val.fecha.min() and val.fecha.max() <= test.fecha.min()),
    "Umbrales alto/medio/bajo calculados solo con train": bool(np.isclose(q1, q1_chk) and np.isclose(q2, q2_chk)),
    "Sin ids duplicados en el dataset": bool(df.id_post.is_unique),
}
assert all(auditoria.values()), f"Auditoría fallida: {auditoria}"
actualizar_metricas("auditoria_fuga", {k: bool(v) for k, v in auditoria.items()})
actualizar_metricas("split", {
    "filas": {"train": len(train), "val": len(val), "test": len(test)},
    "rango_fechas": {k: [str(d.fecha.min().date()), str(d.fecha.max().date())] for k, d in
                     [("train", train), ("val", val), ("test", test)]},
    "umbrales": {"bajo_medio": float(q1), "medio_alto": float(q2)},
    "distribucion": {k: d["nivel_rendimiento"].value_counts().reindex(NIVELES).fillna(0).astype(int).to_dict()
                     for k, d in [("train", train), ("val", val), ("test", test)]},
})

# ---- CSV discretizados para los ejercicios en Java (reglas fijas, sin estadísticos del dataset) ----
def java_df(d):
    return pd.DataFrame({
        "formato": d.formato.fillna("NA"), "tipo": d.tipo_contenido.fillna("NA"),
        "franja": pd.cut(d.hora, [-1, 11, 17, 24], labels=["manana", "tarde", "noche"]).astype(object).fillna("NA"),
        "cta": d.tiene_cta, "pregunta": d.tiene_pregunta, "rostro": d.tiene_rostro,
        "texto_img": pd.cut(d.pct_texto_en_img, [-1, 20, 40, 101], labels=["bajo", "medio", "alto"]).astype(object).fillna("NA"),
        "largo": pd.cut(d.len_copy, [-1, 80, 160, 10000], labels=["corto", "medio", "largo"]).astype(object).fillna("NA"),
        "nivel": d.nivel_rendimiento})
for nombre, d in [("train", train), ("val", val), ("test", test)]:
    java_df(d).to_csv(DATA_SPLITS / f"java_{nombre}.csv", index=False)

print(f"train={len(train)} val={len(val)} test={len(test)} | umbrales: {q1:.4f} / {q2:.4f}")
for k, v in auditoria.items(): print(("✔ " if v else "✘ ") + k)

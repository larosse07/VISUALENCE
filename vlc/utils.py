"""Utilidades: verificación anti-fuga, métricas, guardado de predicciones y figuras."""
import json, re, unicodedata
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from .config import *


def slug(texto):
    """Nombre de archivo seguro (sin tildes, espacios ni símbolos)."""
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_").lower()


def verificar_sin_fuga(columnas):
    """Falla si alguna columna de resultados (post-publicación) se cuela como variable de entrada."""
    malas = set(RESULTADOS) & set(columnas)
    if malas:
        raise ValueError(f"FUGA DE DATOS: columnas post-publicación usadas como entrada: {sorted(malas)}")


def cargar_split(nombre):
    return pd.read_csv(DATA_SPLITS / f"{nombre}.csv", parse_dates=["fecha"])


def xy(df):
    verificar_sin_fuga(FEATURES_PRE)
    return df[FEATURES_PRE].copy(), df["nivel_rendimiento"].astype(str)


def proba_ordenada(modelo, X):
    """Probabilidades en el orden fijo bajo/medio/alto, sin importar el orden interno de classes_."""
    p = modelo.predict_proba(X)
    clases = list(modelo.classes_)
    return p[:, [clases.index(c) for c in NIVELES]]


def metricas_clasificacion(y_true, y_pred, proba=None):
    m = dict(accuracy=accuracy_score(y_true, y_pred),
             f1_macro=f1_score(y_true, y_pred, average="macro", labels=NIVELES, zero_division=0),
             precision_macro=precision_score(y_true, y_pred, average="macro", labels=NIVELES, zero_division=0),
             recall_macro=recall_score(y_true, y_pred, average="macro", labels=NIVELES, zero_division=0))
    if proba is not None:
        try:
            m["auc_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr", labels=NIVELES)
        except ValueError:
            m["auc_ovr"] = float("nan")
    return {k: float(v) for k, v in m.items()}


def guardar_preds(nombre, split, y_true, y_pred, proba):
    d = pd.DataFrame({"modelo": nombre, "y_true": list(y_true), "y_pred": list(y_pred)})
    for i, n in enumerate(NIVELES):
        d[f"p_{n}"] = proba[:, i]
    d.to_csv(PREDS / f"{slug(nombre)}__{split}.csv", index=False)


def actualizar_metricas(clave, valor):
    ruta = REPORTS / "metricas.json"
    datos = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    datos[clave] = valor
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def leer_metricas():
    ruta = REPORTS / "metricas.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def guardar_fig(fig, nombre):
    fig.savefig(FIGS / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

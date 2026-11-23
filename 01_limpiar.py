"""PASO 1 · Limpieza del CSV crudo. NO usa estadísticas globales (la imputación ocurre luego, solo con train)."""
import unicodedata
import numpy as np, pandas as pd
from vlc.config import *
from vlc.features import features_texto
from vlc.utils import actualizar_metricas

ruta = DATA_RAW / "publicaciones_crudo.csv"
if not ruta.exists():
    raise SystemExit(f"No encuentro {ruta}. Copia ahí tu CSV crudo.")

df = pd.read_csv(ruta)
log = {"filas_iniciales": int(len(df))}

# 1) Duplicados (exactos y por id): evitan que la misma publicación caiga en train y test
n = len(df); df = df.drop_duplicates(); log["duplicados_exactos"] = int(n - len(df))
n = len(df); df = df.drop_duplicates(subset="id_post", keep="first"); log["duplicados_por_id"] = int(n - len(df))

# 2) Normalizar categorías (espacios, mayúsculas, tildes) y validar valores permitidos
def norm(s):
    if pd.isna(s): return np.nan
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().strip().lower().replace(" ", "_")
    return s or np.nan
for col, validos in [("formato", FORMATOS), ("tipo_contenido", TIPOS)]:
    df[col] = df[col].map(norm)
    log[f"{col}_invalidos"] = int((~df[col].isin(validos) & df[col].notna()).sum())
    df.loc[~df[col].isin(validos), col] = np.nan

# 3) Fechas con dos formatos mezclados
def parsear(s):
    if pd.isna(s): return pd.NaT
    s = str(s).strip()
    return pd.to_datetime(s, format="%d/%m/%Y" if "/" in s else "%Y-%m-%d", errors="coerce")
df["fecha"] = df["fecha"].map(parsear)
n = len(df); df = df[df["fecha"].notna()].copy(); log["filas_sin_fecha_valida"] = int(n - len(df))
df["dia_semana"] = df["fecha"].dt.weekday.map(dict(enumerate(DIAS)))

# 4) Rangos válidos (fuera de rango -> NaN; se imputan luego con estadísticos de train)
rangos = {"hora": (0, 23), "brillo_img": (0, 255), "contraste_img": (0, 255), "saturacion_img": (0, 255),
          "pct_texto_en_img": (0, 100)}
for c, (lo, hi) in rangos.items():
    df[c] = pd.to_numeric(df[c], errors="coerce")
    log[f"{c}_fuera_de_rango"] = int(((df[c] < lo) | (df[c] > hi)).sum())
    df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan

# 5) Texto: sin copy no hay publicación analizable. Se RECALCULAN las variables derivadas del texto
n = len(df); df = df[df["copy"].notna() & (df["copy"].astype(str).str.strip() != "")].copy()
log["filas_sin_copy"] = int(n - len(df))
feats = df["copy"].apply(features_texto).apply(pd.Series)
for c in feats.columns: df[c] = feats[c].values

# 6) Variable objetivo numérica válida; se elimina la etiqueta global del crudo (se recalcula luego SOLO con train)
df["tasa_interaccion"] = pd.to_numeric(df["tasa_interaccion"], errors="coerce")
n = len(df); df = df[df["tasa_interaccion"].between(0, 1)].copy(); log["filas_tasa_invalida"] = int(n - len(df))
df = df.drop(columns=["nivel_rendimiento"], errors="ignore")

df = df.sort_values(["fecha", "id_post"]).reset_index(drop=True)
log["filas_finales"] = int(len(df))
log["nulos_restantes"] = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
df.to_csv(DATA_PROC / "publicaciones_limpio.csv", index=False, encoding="utf-8")
actualizar_metricas("calidad_datos", log)
print("Limpieza lista:", log)

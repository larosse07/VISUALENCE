"""Ingeniería de características. El MISMO código se usa en entrenamiento y en la app (sin desfase train/serve)."""
import re
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import CAT, NUM, TXT

_CTA = ["escríbenos", "escribenos", "reserva", "comenta", "haz clic", "guarda", "compártelo",
        "compartelo", "whatsapp", "link", "enlace", "cotiza", "agenda"]
_URG = ["solo por hoy", "últimos", "ultimos", "tiempo limitado", "esta semana", "ahora mismo"]


def features_texto(txt: str) -> dict:
    t = str(txt).lower()
    return dict(
        len_copy=len(str(txt)),
        num_hashtags=len(re.findall(r"#\w+", str(txt))),
        tiene_cta=int(any(k in t for k in _CTA)),
        tiene_urgencia=int(any(k in t for k in _URG)),
        tiene_pregunta=int("?" in t),
        tiene_emoji=int(any(ord(c) > 10000 for c in str(txt))),
    )


def construir_preprocesador(con_texto: bool = True, max_terms: int = 150) -> ColumnTransformer:
    """Imputación + escalado + one-hot + TF-IDF. Se ajusta SOLO con train (siempre dentro de un Pipeline)."""
    partes = [
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
    ]
    if con_texto:
        partes.append(("txt", TfidfVectorizer(max_features=max_terms), TXT))
    return ColumnTransformer(partes, sparse_threshold=0)

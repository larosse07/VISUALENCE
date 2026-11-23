"""PASO 4 · Machine Learning clásico (clasificación alto/medio/bajo).
Anti-fuga: preprocesado dentro de Pipeline · tuning con TimeSeriesSplit SOLO sobre train · selección con val · test se evalúa una vez."""
import json, datetime, platform
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, learning_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from vlc.config import *
from vlc.features import construir_preprocesador
from vlc.utils import *

tr, va, te = cargar_split("train"), cargar_split("val"), cargar_split("test")
Xtr, ytr = xy(tr); Xva, yva = xy(va); Xte, yte = xy(te)
cv = TimeSeriesSplit(n_splits=5)

modelos = {
    "Regresión logística": (LogisticRegression(max_iter=2000), {"clf__C": [0.1, 1, 10]}),
    "KNN": (KNeighborsClassifier(), {"clf__n_neighbors": [5, 15, 31]}),
    "SVM (RBF)": (SVC(probability=True, random_state=SEED), {"clf__C": [1, 10]}),
    "Árbol de decisión": (DecisionTreeClassifier(random_state=SEED), {"clf__max_depth": [3, 5, 8]}),
    "Random Forest": (RandomForestClassifier(random_state=SEED), {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 8]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED),
                          {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3]}),
}

resultados, ajustados = {}, {}

def evaluar(nombre, pipe, params=None):
    r = {"params": params or {}}
    for split, X, y in [("val", Xva, yva), ("test", Xte, yte)]:
        pred, proba = pipe.predict(X), proba_ordenada(pipe, X)
        r[split] = metricas_clasificacion(y, pred, proba)
        guardar_preds(f"ml_{nombre}", split, y, pred, proba)
    resultados[nombre] = r

# Línea base: siempre la clase más frecuente de train
base = Pipeline([("prep", construir_preprocesador()), ("clf", DummyClassifier(strategy="most_frequent"))]).fit(Xtr, ytr)
evaluar("Línea base", base)

for nombre, (est, grid) in modelos.items():
    pipe = Pipeline([("prep", construir_preprocesador()), ("clf", est)])
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=1).fit(Xtr, ytr)   # solo train
    ajustados[nombre] = gs.best_estimator_
    evaluar(nombre, gs.best_estimator_, {k: (v if v is None or isinstance(v, str) else float(v)) for k, v in gs.best_params_.items()})
    resultados[nombre]["cv_f1_train"] = float(gs.best_score_)
    print(f"{nombre:22s} CV={gs.best_score_:.3f}  val F1={resultados[nombre]['val']['f1_macro']:.3f}  test F1={resultados[nombre]['test']['f1_macro']:.3f}")

# Selección del mejor modelo usando VALIDACIÓN (nunca test)
mejor = max(ajustados, key=lambda k: resultados[k]["val"]["f1_macro"])
best = ajustados[mejor]
print(f"\nMejor modelo según validación: {mejor}")

# Prueba de cordura anti-fuga: con etiquetas barajadas el modelo debe rendir como el azar (~0.33)
ybar = np.random.default_rng(SEED).permutation(ytr.values)
f1_bar = metricas_clasificacion(yva, clone(best).fit(Xtr, ybar).predict(Xva))["f1_macro"]

# Importancia por permutación (sobre val, con las columnas originales)
pi = permutation_importance(best, Xva, yva, scoring="f1_macro", n_repeats=20, random_state=SEED)
imp = pd.Series(pi.importances_mean, index=Xva.columns).sort_values(ascending=False)

# Curva de aprendizaje (solo con train)
tam, a_tr, a_cv = learning_curve(clone(best), Xtr, ytr, cv=TimeSeriesSplit(4), scoring="f1_macro",
                                 train_sizes=np.linspace(0.3, 1.0, 5))

# Modelo final de despliegue: mejor configuración reentrenada con train+val (test sigue intacto)
final = clone(best).fit(pd.concat([Xtr, Xva]), pd.concat([ytr, yva]))
joblib.dump(final, MODELS / "modelo_final.pkl")
(MODELS / "metadata.json").write_text(json.dumps({
    "modelo": mejor, "entrenado": datetime.datetime.now().isoformat(timespec="seconds"),
    "sklearn": sklearn.__version__, "python": platform.python_version(),
    "features": FEATURES_PRE, "nota": "Reentrenado con train+val; test no se usó."}, indent=2, ensure_ascii=False))

actualizar_metricas("ml", {"resultados": resultados, "mejor": mejor, "sanity_etiquetas_barajadas_f1_val": float(f1_bar),
                           "importancia_permutacion": {k: float(v) for k, v in imp.head(12).items()},
                           "curva_aprendizaje": {"tam": tam.tolist(), "train": a_tr.mean(1).tolist(), "cv": a_cv.mean(1).tolist()}})
print(f"Prueba de cordura (etiquetas barajadas) F1 val = {f1_bar:.3f}  (esperado ≈ 0.33)")

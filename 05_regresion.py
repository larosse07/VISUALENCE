"""PASO 5 · Regresión lineal (y variantes) para predecir la tasa de interacción numérica. Mismo split temporal."""
import numpy as np, pandas as pd, joblib
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from vlc.config import *
from vlc.features import construir_preprocesador
from vlc.utils import *

tr, va, te = cargar_split("train"), cargar_split("val"), cargar_split("test")
verificar_sin_fuga(FEATURES_PRE)
Xtr, Xva, Xte = tr[FEATURES_PRE], va[FEATURES_PRE], te[FEATURES_PRE]
ytr, yva, yte = tr["tasa_interaccion"], va["tasa_interaccion"], te["tasa_interaccion"]

def met(y, p):
    return {"R2": float(r2_score(y, p)), "MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p)))}

modelos = {
    "Línea base (media)": DummyRegressor(strategy="mean"),
    "Regresión lineal": LinearRegression(),
    "Ridge (α por CV)": RidgeCV(alphas=np.logspace(-2, 3, 20)),
    "Random Forest (reg.)": RandomForestRegressor(300, random_state=SEED),
}
res, aj = {}, {}
for n, est in modelos.items():
    pipe = Pipeline([("prep", construir_preprocesador(con_texto=False)), ("reg", est)]).fit(Xtr, ytr)
    aj[n] = pipe
    res[n] = {"val": met(yva, pipe.predict(Xva)), "test": met(yte, pipe.predict(Xte))}
    pd.DataFrame({"y_true": yte.values, "y_pred": pipe.predict(Xte)}).to_csv(PREDS / f"reg_{slug(n)}__test.csv", index=False)
    print(f"{n:22s} val R²={res[n]['val']['R2']:.3f}  test R²={res[n]['test']['R2']:.3f}  test MAE={res[n]['test']['MAE']:.4f}")

mejor = max([k for k in aj if "base" not in k], key=lambda k: res[k]["val"]["R2"])   # se elige con validación
lin = aj["Regresión lineal"]
coef = pd.Series(lin.named_steps["reg"].coef_, index=lin.named_steps["prep"].get_feature_names_out())
coef = coef.reindex(coef.abs().sort_values(ascending=False).index)
resid = yte.values - aj[mejor].predict(Xte)
sw = stats.shapiro(resid)

final = clone(aj[mejor]).fit(pd.concat([Xtr, Xva]), pd.concat([ytr, yva]))
joblib.dump(final, MODELS / "regresion.pkl")
actualizar_metricas("regresion", {"resultados": res, "mejor": mejor,
                                  "coeficientes_lineal": {k: float(v) for k, v in coef.head(12).items()},
                                  "shapiro_residuos_test": {"estadistico": float(sw.statistic), "p": float(sw.pvalue)}})
print("Mejor (según validación):", mejor)

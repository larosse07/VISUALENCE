import json
import numpy as np, pandas as pd, joblib
import streamlit as st
import matplotlib.pyplot as plt, seaborn as sns
from vlc.config import *
from vlc.features import features_texto
from vlc.utils import leer_metricas, proba_ordenada, verificar_sin_fuga

try:
    import cv2
    from vlc.vision import features_imagen, leer_imagen_bytes
    HAY_CV2 = True
except ImportError:
    HAY_CV2 = False

st.set_page_config(page_title="VISUALENCE CV", layout="wide")
st.title("VISUALENCE CV · Consigue Ventas")
st.caption("Análisis y predicción del rendimiento de contenido digital · prototipo de prácticas preprofesionales")
sns.set_theme(style="whitegrid")

M = leer_metricas()
if not M or not (MODELS / "modelo_final.pkl").exists():
    st.error("Faltan resultados. Ejecuta primero: python run_all.py  (o los scripts 01 a 08 en orden).")
    st.stop()

@st.cache_resource
def cargar():
    return (joblib.load(MODELS / "modelo_final.pkl"), joblib.load(MODELS / "regresion.pkl"),
            joblib.load(MODELS / "segmentador.pkl"), json.loads((MODELS / "umbrales.json").read_text()))
modelo, reg, seg, umbrales = cargar()

def figura(nombre, titulo=None):
    ruta = FIGS / f"{nombre}.png"
    if ruta.exists(): st.image(str(ruta), caption=titulo, use_container_width=True)
    else: st.info(f"Falta la figura {nombre}.png (ejecuta 08_reporte.py).")

tabs = st.tabs(["📊 Datos", "🤖 ML y Deep Learning", "📈 Regresión", "🧩 Segmentos", "🖼️ Imagen", "🛡️ Anti-fuga", "🔮 Predecir"])

# ---------------------------------------------------------------- Datos
with tabs[0]:
    up = st.file_uploader("Sube tu CSV o Excel de publicaciones (opcional)", type=["csv", "xlsx"])
    if up is None:
        df = pd.read_csv(DATA_PROC / "publicaciones_limpio.csv"); st.info("Mostrando el dataset de ejemplo (sintético, ya limpio).")
    else:
        df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
    req = {"formato", "hora", "tipo_contenido", "dia_semana", "tasa_interaccion"}
    if not req.issubset(df.columns):
        st.error(f"Al archivo le faltan columnas: {sorted(req - set(df.columns))}")
    else:
        c = st.columns(4)
        c[0].metric("Publicaciones", len(df)); c[1].metric("Tasa de interacción promedio", f"{df['tasa_interaccion'].mean():.2%}")
        if "alcance" in df: c[2].metric("Alcance promedio", f"{df['alcance'].mean():,.0f}")
        if "leads" in df: c[3].metric("Leads totales", f"{int(df['leads'].sum()):,}")
        g = st.columns(2)
        with g[0]:
            fig, ax = plt.subplots(); sns.barplot(data=df, x="formato", y="tasa_interaccion", ax=ax); ax.set_title("Por formato"); st.pyplot(fig)
        with g[1]:
            fig, ax = plt.subplots(); sns.lineplot(data=df, x="hora", y="tasa_interaccion", ax=ax); ax.set_title("Por hora"); st.pyplot(fig)
        g = st.columns(2)
        with g[0]:
            fig, ax = plt.subplots(); sns.barplot(data=df, x="tipo_contenido", y="tasa_interaccion", ax=ax); plt.xticks(rotation=30); ax.set_title("Por tipo"); st.pyplot(fig)
        with g[1]:
            piv = df.pivot_table(index="dia_semana", columns="formato", values="tasa_interaccion").reindex(DIAS)
            fig, ax = plt.subplots(); sns.heatmap(piv, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax); ax.set_title("Día × formato"); st.pyplot(fig)

# ---------------------------------------------------------------- ML y DL
with tabs[1]:
    ML, DL = M["ml"]["resultados"], M.get("dl", {})
    filas = [{"Modelo": k, "F1 val": v["val"]["f1_macro"], "F1 test": v["test"]["f1_macro"], "Accuracy test": v["test"]["accuracy"],
              "AUC test": v["test"].get("auc_ovr", np.nan)} for k, v in ML.items()]
    filas += [{"Modelo": f"{k} (DL)", "F1 val": v["val"]["f1_macro"], "F1 test": v["test"]["f1_macro"], "Accuracy test": v["test"]["accuracy"],
               "AUC test": v["test"].get("auc_ovr", np.nan)} for k, v in DL.items() if isinstance(v, dict) and "test" in v]
    st.dataframe(pd.DataFrame(filas).round(3), use_container_width=True)
    st.caption(f"Mejor modelo clásico según VALIDACIÓN: **{M['ml']['mejor']}**. Azar con 3 clases ≈ 0.33. "
               "Con pocas publicaciones de test, diferencias pequeñas no son concluyentes.")
    if not any(isinstance(v, dict) and "test" in v for v in DL.values()):
        st.warning("Deep Learning aún no ejecutado: instala requirements-dl.txt y corre 07_deep_learning.py y 08_reporte.py.")
    figura("comparacion_modelos"); figura("matrices_confusion_test"); figura("roc_test")
    a, b = st.columns(2)
    with a: figura("importancia_variables")
    with b: figura("curva_aprendizaje")
    figura("diagrama_red"); figura("curvas_dl")

# ---------------------------------------------------------------- Regresión
with tabs[2]:
    R = M["regresion"]
    st.dataframe(pd.DataFrame([{"Modelo": k, "R² val": v["val"]["R2"], "R² test": v["test"]["R2"], "MAE test": v["test"]["MAE"],
                                "RMSE test": v["test"]["RMSE"]} for k, v in R["resultados"].items()]).round(4), use_container_width=True)
    st.caption(f"Mejor según validación: **{R['mejor']}**. La regresión lineal es la más interpretable (coeficientes).")
    figura("regresion_diagnostico"); figura("regresion_coeficientes")

# ---------------------------------------------------------------- Segmentos
with tabs[3]:
    S = M["segmentacion"]; st.write(f"k óptimo por silhouette: **{S['k']}** (separación débil: {S['silhouette'][str(S['k'])]:.3f})")
    figura("seg_clusters")
    st.markdown("**Perfil en train**"); st.dataframe(pd.DataFrame(S["perfil_train"]), use_container_width=True)
    st.markdown("**Validación en test**"); st.dataframe(pd.DataFrame(S["perfil_test"]), use_container_width=True)

# ---------------------------------------------------------------- Imagen
with tabs[4]:
    st.subheader("Análisis visual de una pieza gráfica (OpenCV)")
    figura("eda_imagen")
    f = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"], key="img_tab")
    if f is not None and HAY_CV2:
        img = leer_imagen_bytes(f.getvalue()); fi = features_imagen(img)
        c1, c2 = st.columns(2)
        with c1: st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=320)
        with c2:
            alto = pd.read_csv(DATA_SPLITS / "train.csv").query("nivel_rendimiento == 'alto'")
            comp = pd.DataFrame({"Tu imagen": [fi["brillo_img"], fi["contraste_img"], fi["saturacion_img"], fi["pct_texto_en_img"]],
                                 "Promedio piezas ALTO": [alto.brillo_img.mean(), alto.contraste_img.mean(), alto.saturacion_img.mean(), alto.pct_texto_en_img.mean()]},
                                index=["Brillo", "Contraste", "Saturación", "% texto (aprox.)"]).round(1)
            st.dataframe(comp); st.write("Rostro detectado:", "sí" if fi["tiene_rostro"] else "no")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            for i, col in enumerate(["blue", "green", "red"]): ax.plot(cv2.calcHist([img], [i], None, [256], [0, 256]), color=col)
            ax.set_title("Histograma de color"); st.pyplot(fig)
    elif f is not None:
        st.warning("Instala OpenCV: pip install opencv-python")

# ---------------------------------------------------------------- Anti-fuga
with tabs[5]:
    st.subheader("Controles contra fuga de datos")
    for k, v in M["auditoria_fuga"].items(): st.write(("✅ " if v else "❌ ") + k)
    s = M["split"]
    st.write(f"**División temporal:** {s['filas']} · fechas: {s['rango_fechas']}")
    st.write(f"**Umbrales de clase (calculados solo con train):** bajo ≤ {s['umbrales']['bajo_medio']:.2%} < medio ≤ {s['umbrales']['medio_alto']:.2%} < alto")
    st.write(f"**Prueba de cordura** (etiquetas barajadas) → F1 val = {M['ml']['sanity_etiquetas_barajadas_f1_val']:.3f} (azar ≈ 0.33)")
    st.write("**Variables prohibidas como entrada:**", ", ".join(RESULTADOS))
    figura("diagrama_pipeline"); figura("diagrama_split")

# ---------------------------------------------------------------- Predecir
with tabs[6]:
    a, b = st.columns(2)
    with a:
        copy = st.text_area("Copy de la publicación", "🔥 Descubre 3 tips para vender más. ¿Ya los aplicas? Reserva tu asesoría gratis. #ventas")
        formato = st.selectbox("Formato", FORMATOS); tipo = st.selectbox("Tipo de contenido", TIPOS)
        dia = st.selectbox("Día", DIAS, index=3); hora = st.slider("Hora de publicación", 0, 23, 19)
    with b:
        img_f = st.file_uploader("Imagen de la pieza (opcional)", type=["jpg", "jpeg", "png"], key="img_pred")
        if img_f is not None and HAY_CV2:
            img = leer_imagen_bytes(img_f.getvalue()); fi = features_imagen(img)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=240)
            st.caption(f"Brillo {fi['brillo_img']:.0f} · Contraste {fi['contraste_img']:.0f} · Saturación {fi['saturacion_img']:.0f} · "
                       f"Texto ≈ {fi['pct_texto_en_img']:.0f}% · Rostro: {'sí' if fi['tiene_rostro'] else 'no'}")
        else:
            fi = dict(brillo_img=st.slider("Brillo", 0, 255, 125), contraste_img=st.slider("Contraste", 0, 100, 55),
                      saturacion_img=st.slider("Saturación", 0, 255, 110), pct_texto_en_img=st.slider("% de texto en la imagen", 0, 80, 20),
                      tiene_rostro=int(st.checkbox("¿Tiene rostro?")))
    if st.button("Predecir rendimiento", type="primary"):
        fila = dict(dia_semana=dia, formato=formato, tipo_contenido=tipo, hora=hora, copy=copy, **features_texto(copy), **fi)
        X = pd.DataFrame([fila])[FEATURES_PRE]; verificar_sin_fuga(X.columns)
        proba = proba_ordenada(modelo, X)[0]; clase = NIVELES[int(np.argmax(proba))]
        tasa = float(reg.predict(X)[0]); segmento = int(seg.predict(X)[0])
        st.subheader(f"Rendimiento estimado: {clase.upper()}")
        m = st.columns(3)
        m[0].metric("Confianza", f"{proba.max():.0%}"); m[1].metric("Tasa de interacción estimada", f"{tasa:.2%}"); m[2].metric("Segmento", segmento)
        st.caption(f"Alto = tasa ≥ {umbrales['medio_alto']:.2%} · Medio = entre {umbrales['bajo_medio']:.2%} y {umbrales['medio_alto']:.2%} · "
                   f"Bajo = menos de {umbrales['bajo_medio']:.2%}. Estimación orientativa (modelo entrenado con datos sintéticos).")
        st.bar_chart(pd.Series(proba, index=NIVELES))
        with st.expander("Variables detectadas"):
            st.json({k: (round(float(v), 1) if isinstance(v, (float, np.floating)) else v) for k, v in fila.items() if k != "copy"})

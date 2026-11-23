import os; os.makedirs("data/raw", exist_ok=True)  # ejecutar desde la raíz del proyecto
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
N = 800

formatos = ["reel", "carrusel", "imagen", "historia", "video_largo"]
tipos = ["promocional", "educativo", "testimonio", "entretenimiento", "producto"]
dias = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

fechas = pd.date_range("2025-01-01", periods=N, freq="10h")
df = pd.DataFrame({"id_post": range(1, N + 1)})
df["fecha"] = fechas.date
df["dia_semana"] = [dias[d.weekday()] for d in fechas]
df["hora"] = rng.choice(range(6, 24), N, p=np.array([1,1,2,2,3,3,3,4,4,5,6,7,8,8,7,5,3,2]) / np.array([1,1,2,2,3,3,3,4,4,5,6,7,8,8,7,5,3,2]).sum())
df["formato"] = rng.choice(formatos, N, p=[0.30, 0.25, 0.25, 0.10, 0.10])
df["tipo_contenido"] = rng.choice(tipos, N)

# ---- Copies con plantillas ----
aperturas = ["Descubre", "Conoce", "Te contamos", "Mira esto:", "Atención emprendedores:", "Hoy en Consigue Ventas", "Ojo:", "Aprende"]
temas = {
    "promocional": ["nuestra promo del mes", "un descuento exclusivo", "la oferta de lanzamiento", "un pack especial para tu negocio"],
    "educativo": ["3 tips para vender más", "cómo mejorar tu embudo de ventas", "errores comunes en redes sociales", "cómo fidelizar clientes"],
    "testimonio": ["la historia de un cliente que duplicó sus ventas", "lo que dicen quienes ya confían en nosotros", "un caso real de éxito"],
    "entretenimiento": ["lo que todo vendedor ha vivido", "el día a día en una agencia", "un reto con nuestro equipo"],
    "producto": ["nuestro nuevo servicio de ads", "la herramienta que automatiza tu seguimiento", "el paquete de contenido mensual"],
}
ctas = ["Escríbenos por WhatsApp", "Reserva tu asesoría gratis", "Comenta QUIERO y te escribimos", "Guarda este post", "Haz clic en el enlace de la bio", "Compártelo con un amigo"]
urg = ["Solo por hoy", "Últimos cupos", "Por tiempo limitado", "Esta semana"]
emojis = ["🔥", "🚀", "✅", "💡", "📈", "👉"]
hashtags_pool = ["#marketingdigital", "#ventas", "#emprendedores", "#peru", "#redessociales", "#negocios", "#lima", "#publicidad"]

copies, tiene_cta, tiene_urg, tiene_preg, tiene_emoji, n_hash = [], [], [], [], [], []
for t in df["tipo_contenido"]:
    txt = f"{rng.choice(aperturas)} {rng.choice(temas[t])}."
    cta = rng.random() < 0.55
    u = rng.random() < 0.25
    q = rng.random() < 0.30
    e = rng.random() < 0.60
    if q: txt += " ¿Ya lo estás aplicando en tu negocio?"
    if u: txt += f" {rng.choice(urg)}."
    if cta: txt += f" {rng.choice(ctas)}."
    if e: txt = f"{rng.choice(emojis)} " + txt
    h = int(rng.integers(0, 8))
    if h: txt += " " + " ".join(rng.choice(hashtags_pool, h, replace=False))
    copies.append(txt); tiene_cta.append(int(cta)); tiene_urg.append(int(u))
    tiene_preg.append(int(q)); tiene_emoji.append(int(e)); n_hash.append(h)

df["copy"] = copies
df["len_copy"] = df["copy"].str.len()
df["tiene_cta"] = tiene_cta
df["tiene_urgencia"] = tiene_urg
df["tiene_pregunta"] = tiene_preg
df["tiene_emoji"] = tiene_emoji
df["num_hashtags"] = n_hash

# ---- Características de imagen (para visión computacional) ----
df["brillo_img"] = np.clip(rng.normal(125, 35, N), 20, 240).round(1)
df["contraste_img"] = np.clip(rng.normal(55, 15, N), 10, 100).round(1)
df["saturacion_img"] = np.clip(rng.normal(110, 40, N), 10, 255).round(1)
df["pct_texto_en_img"] = np.clip(rng.normal(25, 15, N), 0, 80).round(1)
df["tiene_rostro"] = rng.choice([0, 1], N, p=[0.55, 0.45])

# ---- Puntaje latente: el rendimiento SÍ depende de las variables ----
score = (
    df["formato"].map({"reel": 1.6, "carrusel": 1.0, "video_largo": 0.5, "imagen": 0.0, "historia": -0.4})
    + df["tipo_contenido"].map({"educativo": 0.8, "testimonio": 0.9, "entretenimiento": 0.6, "producto": 0.0, "promocional": -0.2})
    + df["hora"].between(18, 21) * 1.0
    + df["hora"].between(12, 14) * 0.5
    + df["dia_semana"].isin(["martes", "jueves", "domingo"]) * 0.4
    + df["tiene_cta"] * 0.7
    + df["tiene_pregunta"] * 0.5
    + df["tiene_urgencia"] * 0.3
    + df["tiene_rostro"] * 0.6
    + ((df["brillo_img"] - 125).abs() < 30) * 0.4
    + (df["pct_texto_en_img"] > 45) * -0.6
    + (df["len_copy"] > 220) * -0.4
    + (df["num_hashtags"] > 6) * -0.3
    + rng.normal(0, 0.9, N)
)

# ---- Resultados (solo se conocen DESPUÉS de publicar) ----
base_alcance = rng.lognormal(7.2, 0.35, N)
df["alcance"] = (base_alcance * np.exp(0.28 * score)).astype(int)
tasa = np.clip(0.025 + 0.012 * score + rng.normal(0, 0.006, N), 0.004, 0.2)
df["interacciones"] = (df["alcance"] * tasa).astype(int)
df["guardados"] = (df["interacciones"] * np.clip(0.10 + 0.03 * df["tiene_cta"] + rng.normal(0, 0.03, N), 0.01, 0.4)).astype(int)
df["clics"] = (df["alcance"] * np.clip(0.008 + 0.004 * score + rng.normal(0, 0.003, N), 0.001, 0.08)).astype(int)
df["leads"] = rng.binomial(df["clics"].clip(lower=1), np.clip(0.04 + 0.01 * score, 0.005, 0.2))
df["tasa_interaccion"] = (df["interacciones"] / df["alcance"]).round(4)
df["nivel_rendimiento"] = pd.qcut(df["tasa_interaccion"].rank(method="first"), 3, labels=["bajo", "medio", "alto"])

orden = ["id_post","fecha","dia_semana","hora","formato","tipo_contenido","copy","len_copy","num_hashtags",
         "tiene_cta","tiene_urgencia","tiene_pregunta","tiene_emoji","brillo_img","contraste_img",
         "saturacion_img","pct_texto_en_img","tiene_rostro","alcance","interacciones","guardados","clics",
         "leads","tasa_interaccion","nivel_rendimiento"]
limpio = df[orden].copy()
limpio.to_csv("data/raw/_referencia_con_etiqueta_global.csv", index=False, encoding="utf-8-sig")

# ---- Versión CRUDA (con suciedad intencional para practicar limpieza) ----
crudo = limpio.copy()
crudo["formato"] = crudo["formato"].astype(object)
idx = rng.choice(N, 60, replace=False); crudo.loc[idx, "formato"] = crudo.loc[idx, "formato"].str.upper()
idx = rng.choice(N, 40, replace=False); crudo.loc[idx, "formato"] = " " + crudo.loc[idx, "formato"] + " "
for col, k in [("hora", 25), ("brillo_img", 30), ("alcance", 20), ("tipo_contenido", 20), ("copy", 15)]:
    crudo[col] = crudo[col].astype(object)
    crudo.loc[rng.choice(N, k, replace=False), col] = np.nan
crudo["fecha"] = crudo["fecha"].astype(str)
idx = rng.choice(N, 80, replace=False)
crudo.loc[idx, "fecha"] = pd.to_datetime(crudo.loc[idx, "fecha"]).dt.strftime("%d/%m/%Y")
crudo = pd.concat([crudo, crudo.sample(12, random_state=1)], ignore_index=True)  # duplicados
crudo["nivel_rendimiento"] = crudo["nivel_rendimiento"].astype(str)
crudo.to_csv("data/raw/publicaciones_crudo.csv", index=False, encoding="utf-8-sig")
print(limpio.shape, crudo.shape)
print(limpio["nivel_rendimiento"].value_counts())

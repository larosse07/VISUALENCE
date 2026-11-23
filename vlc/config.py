"""Configuración central. Aquí se define QUÉ variables puede usar un modelo (anti fuga de datos)."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROC = RAIZ / "data" / "processed"
DATA_SPLITS = RAIZ / "data" / "splits"
MODELS = RAIZ / "models"
REPORTS = RAIZ / "reports"
FIGS = REPORTS / "figures"
PREDS = REPORTS / "preds"
for _d in (DATA_RAW, DATA_PROC, DATA_SPLITS, MODELS, FIGS, PREDS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
NIVELES = ["bajo", "medio", "alto"]

# --- Variables PRE-publicación: lo único que un modelo puede ver ---
CAT = ["dia_semana", "formato", "tipo_contenido"]
NUM = ["hora", "len_copy", "num_hashtags", "tiene_cta", "tiene_urgencia", "tiene_pregunta",
       "tiene_emoji", "brillo_img", "contraste_img", "saturacion_img", "pct_texto_en_img", "tiene_rostro"]
TXT = "copy"
FEATURES_PRE = CAT + NUM + [TXT]

# --- Variables POST-publicación (resultados): PROHIBIDAS como entrada de cualquier modelo ---
RESULTADOS = ["alcance", "interacciones", "guardados", "clics", "leads", "tasa_interaccion", "nivel_rendimiento"]

FORMATOS = ["reel", "carrusel", "imagen", "historia", "video_largo"]
TIPOS = ["promocional", "educativo", "testimonio", "entretenimiento", "producto"]
DIAS = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

"""UTILIDAD · Extrae características de imágenes reales con OpenCV.
Uso:  python 10_features_imagenes.py ruta/a/carpeta_imagenes
Cada archivo debe llamarse <id_post>.jpg|png (ej. 17.jpg). Genera data/processed/features_imagen.csv
que se une al CSV crudo por id_post ANTES del paso 01 (reemplaza las columnas *_img y tiene_rostro)."""
import sys
from pathlib import Path
import pandas as pd
from vlc.config import DATA_PROC
from vlc.vision import features_imagen, leer_imagen_bytes

if len(sys.argv) < 2:
    raise SystemExit("Uso: python 10_features_imagenes.py ruta/a/carpeta_imagenes")
filas = []
for f in sorted(Path(sys.argv[1]).iterdir()):
    if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.stem.isdigit():
        img = leer_imagen_bytes(f.read_bytes())
        if img is not None:
            filas.append({"id_post": int(f.stem), **features_imagen(img)})
pd.DataFrame(filas).to_csv(DATA_PROC / "features_imagen.csv", index=False)
print(f"{len(filas)} imágenes procesadas → data/processed/features_imagen.csv")

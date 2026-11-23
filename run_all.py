"""Ejecuta todo el pipeline en orden:  python run_all.py"""
import subprocess, sys
PASOS = ["01_limpiar.py", "02_dividir.py", "03_eda_estadistica.py", "04_ml_clasico.py", "05_regresion.py",
         "06_segmentacion.py", "07_deep_learning.py", "08_reporte.py"]
for p in PASOS:
    print(f"\n{'=' * 8} {p} {'=' * 8}")
    if subprocess.run([sys.executable, p]).returncode != 0:
        raise SystemExit(f"Falló {p}. Corrige el error y vuelve a ejecutar.")
print("\nListo. Ahora ejecuta:  streamlit run app.py")

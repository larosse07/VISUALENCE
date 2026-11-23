"""Visión computacional básica (OpenCV): características de una pieza gráfica."""
import numpy as np
import cv2


def leer_imagen_bytes(contenido: bytes):
    return cv2.imdecode(np.frombuffer(contenido, np.uint8), cv2.IMREAD_COLOR)


def estimar_pct_texto(img) -> float:
    """APROXIMACIÓN del % de área ocupada por texto (bloques horizontales de bordes). No es un OCR."""
    h, w = img.shape[:2]
    esc = 640 / max(h, w)
    g = cv2.cvtColor(cv2.resize(img, None, fx=esc, fy=esc) if esc < 1 else img, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    cont, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    for c in cont:
        x, y, bw_, bh = cv2.boundingRect(c)
        if bw_ > 1.5 * bh and 8 <= bh <= 90 and bw_ >= 30:
            area += bw_ * bh
    return float(min(80.0, 100.0 * area / (g.shape[0] * g.shape[1])))


def features_imagen(img) -> dict:
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cascada = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    rostros = cascada.detectMultiScale(gris, 1.1, 5)
    return dict(brillo_img=float(gris.mean()), contraste_img=float(gris.std()),
                saturacion_img=float(hsv[:, :, 1].mean()), pct_texto_en_img=estimar_pct_texto(img),
                tiene_rostro=int(len(rostros) > 0))

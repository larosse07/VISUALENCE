"""PASO 7 · Deep Learning: la MISMA red neuronal en Keras (TensorFlow) y en PyTorch, con el mismo split y preprocesado.
Anti-fuga: el preprocesador se ajusta SOLO con train · early stopping con VAL · test se evalúa una sola vez al final."""
import os, random
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, pandas as pd, joblib
from vlc.config import *
from vlc.features import construir_preprocesador
from vlc.utils import *

tr, va, te = cargar_split("train"), cargar_split("val"), cargar_split("test")
Xtr_df, ytr = xy(tr); Xva_df, yva = xy(va); Xte_df, yte = xy(te)

prep = construir_preprocesador().fit(Xtr_df)                 # <- SOLO train
joblib.dump(prep, MODELS / "prep_dl.pkl")
Xtr, Xva, Xte = [prep.transform(d).astype("float32") for d in (Xtr_df, Xva_df, Xte_df)]
mapa = {n: i for i, n in enumerate(NIVELES)}
ytr_i, yva_i = ytr.map(mapa).values.astype("int64"), yva.map(mapa).values.astype("int64")
D = Xtr.shape[1]
MAX_EPOCAS, PACIENCIA, LOTE, LR = 200, 15, 32, 1e-3
random.seed(SEED); np.random.seed(SEED)
dl = {"arquitectura": f"Entrada({D}) → Dense128+ReLU → Dropout0.3 → Dense64+ReLU → Dropout0.3 → Softmax3",
      "hiperparametros": {"optimizador": "Adam", "lr": LR, "lote": LOTE, "max_epocas": MAX_EPOCAS, "paciencia": PACIENCIA}}


def registrar(nombre, proba_va, proba_te, hist, epocas):
    r = {"epocas": int(epocas), "historial": hist}
    for split, y, p in [("val", yva, proba_va), ("test", yte, proba_te)]:
        pred = np.array(NIVELES)[p.argmax(1)]
        r[split] = metricas_clasificacion(y, pred, p)
        guardar_preds(f"dl_{nombre}", split, y, pred, p)
    dl[nombre] = r
    print(f"{nombre}: épocas={epocas}  val F1={r['val']['f1_macro']:.3f}  test F1={r['test']['f1_macro']:.3f}")


def entrenar_keras():
    from tensorflow import keras
    keras.utils.set_random_seed(SEED)
    m = keras.Sequential([
        keras.layers.Input(shape=(D,)),
        keras.layers.Dense(128, activation="relu"), keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"), keras.layers.Dropout(0.3),
        keras.layers.Dense(3, activation="softmax")])
    m.compile(optimizer=keras.optimizers.Adam(LR), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA, restore_best_weights=True)
    h = m.fit(Xtr, ytr_i, validation_data=(Xva, yva_i), epochs=MAX_EPOCAS, batch_size=LOTE, callbacks=[es], verbose=0)
    hist = {k: [float(x) for x in v] for k, v in h.history.items()}
    hist = {"loss": hist["loss"], "val_loss": hist["val_loss"], "acc": hist["accuracy"], "val_acc": hist["val_accuracy"]}
    m.save(MODELS / "dl_keras.keras")
    registrar("Keras", m.predict(Xva, verbose=0), m.predict(Xte, verbose=0), hist, len(hist["loss"]))


def entrenar_torch():
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    torch.manual_seed(SEED)
    m = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(),
                      nn.Dropout(0.3), nn.Linear(64, 3))
    opt, crit = torch.optim.Adam(m.parameters(), lr=LR), nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr_i)
    Xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva_i)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=LOTE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    hist = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
    mejor, espera, estado = float("inf"), 0, None
    for ep in range(MAX_EPOCAS):
        m.train(); tot, ok, n = 0.0, 0, 0
        for xb, yb in loader:
            opt.zero_grad(); out = m(xb); loss = crit(out, yb); loss.backward(); opt.step()
            tot += loss.item() * len(xb); ok += (out.argmax(1) == yb).sum().item(); n += len(xb)
        m.eval()
        with torch.no_grad():
            ov = m(Xv); lv = crit(ov, yv).item(); av = (ov.argmax(1) == yv).float().mean().item()
        hist["loss"].append(tot / n); hist["acc"].append(ok / n); hist["val_loss"].append(lv); hist["val_acc"].append(av)
        if lv < mejor - 1e-4:
            mejor, espera, estado = lv, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            espera += 1
            if espera >= PACIENCIA: break
    m.load_state_dict(estado); m.eval()
    torch.save(m.state_dict(), MODELS / "dl_pytorch.pt")
    with torch.no_grad():
        pv = torch.softmax(m(Xv), 1).numpy(); pt = torch.softmax(m(torch.from_numpy(Xte)), 1).numpy()
    registrar("PyTorch", pv, pt, hist, len(hist["loss"]))


for nombre, fn in [("Keras", entrenar_keras), ("PyTorch", entrenar_torch)]:
    try:
        fn()
    except ImportError as e:
        print(f"[{nombre}] omitido: falta la librería ({e}). Instala: pip install -r requirements-dl.txt")
        dl[nombre] = {"disponible": False}
actualizar_metricas("dl", dl)

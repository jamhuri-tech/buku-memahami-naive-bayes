"""Bab 19: prior data uji SmSA berbeda dari data latih.

Prior data uji ditaksir tanpa label dengan algoritma EM Saerens,
Latinne & Decaestecker (2002): posterior dikoreksi dengan prior
taksiran, prior diperbarui menjadi rata-rata posterior terkoreksi, dan
seterusnya. Dibandingkan: tanpa koreksi, koreksi dengan prior EM, dan
koreksi dengan prior uji yang sebenarnya (hanya sebagai pembanding).
Terakhir, kalibrasi (Platt dan isotonik per kelas) yang ditaksir pada
data validasi, dengan model latih dibekukan (FrozenEstimator).
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, f1_score, log_loss

from bab19_evaluasi import KELAS, model_nb, tu, tv, yl, yu, yv


def koreksi(P, prior_lama, prior_baru):
    Q = P * (prior_baru / prior_lama)
    return Q / Q.sum(axis=1, keepdims=True)


def em_prior(P, prior_lama, langkah=100):
    prior = prior_lama.copy()
    for _ in range(langkah):
        prior = koreksi(P, prior_lama, prior).mean(axis=0)
    return prior


if __name__ == "__main__":
    nb = model_nb()
    assert list(nb.classes_) == KELAS
    P = nb.predict_proba(tu)
    lama = np.array([np.mean(yl == k) for k in KELAS])
    benar = np.array([np.mean(yu == k) for k in KELAS])
    em = em_prior(P, lama)
    print("prior (neg, net, pos)")
    for nama, pr in (("latih", lama), ("taksiran EM", em),
                     ("uji sebenarnya", benar)):
        print(f"  {nama:15s} " + "  ".join(f"{v:.3f}" for v in pr))
    print("koreksi          akurasi  F1 makro  log-loss")
    for nama, pr in (("tanpa", lama), ("prior EM", em),
                     ("prior sebenarnya", benar)):
        Q = koreksi(P, lama, pr)
        p = np.array(KELAS)[Q.argmax(axis=1)]
        print(f"{nama:16s}  {accuracy_score(yu, p):.4f}   "
              f"{f1_score(yu, p, average='macro'):.4f}    "
              f"{log_loss(yu, Q, labels=KELAS):.4f}")
    print("kalibrasi pada data validasi")
    for nama, metode in (("Platt", "sigmoid"), ("isotonik", "isotonic")):
        c = CalibratedClassifierCV(FrozenEstimator(nb), method=metode)
        c.fit(tv, yv)
        Q = c.predict_proba(tu)
        p = c.predict(tu)
        print(f"{nama:16s}  {accuracy_score(yu, p):.4f}   "
              f"{f1_score(yu, p, average='macro'):.4f}    "
              f"{log_loss(yu, Q, labels=KELAS):.4f}")

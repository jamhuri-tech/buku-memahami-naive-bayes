"""Bab 8: kategori yang tidak terlihat dan nilai yang hilang.

Bagian 1: kategori yang muncul di data uji tetapi tidak di data latih.
Bagian 2: 30% nilai fitur data uji disembunyikan secara acak. Naive
Bayes dapat melewatkan faktor fitur yang hilang (marginalisasi);
pembandingnya mengisi nilai hilang dengan kategori terbanyak (modus).
"""
import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB

from bab04_data import BENIH
from bab08_data import muat_kode

X, y, _ = muat_kode()
mc = X.max(axis=0) + 1


def log_gabungan(m, X, hilang):
    """log prior + jumlah log P(x_j | k) atas fitur yang teramati."""
    s = np.tile(m.class_log_prior_, (len(X), 1))
    for j, tabel in enumerate(m.feature_log_prob_):  # tabel: K x V_j
        ada = ~hilang[:, j]
        s[ada] += tabel[:, X[ada, j]].T
    return s


def posterior(s):
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    # ---- Bagian 1 ----
    Xl, Xu, yl, yu = train_test_split(X, y, train_size=100, stratify=y,
                                      random_state=BENIH)
    baru = (Xu > Xl.max(axis=0)).any(axis=1)
    print(f"latih 100 jamur; {baru.sum()} jamur uji memuat kategori baru")
    try:
        CategoricalNB(alpha=1).fit(Xl, yl).predict(Xu)
    except IndexError as e:
        print("tanpa min_categories:", type(e).__name__)
    m = CategoricalNB(alpha=1, min_categories=mc).fit(Xl, yl)
    print(f"dengan min_categories: akurasi {m.score(Xu, yu):.4f}")

    # ---- Bagian 2 ----
    rng = np.random.default_rng(BENIH)
    Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y,
                                      random_state=BENIH)
    m = CategoricalNB(alpha=0.01, min_categories=mc).fit(Xl, yl)
    hilang = rng.random(Xu.shape) < 0.3
    tak_hilang = np.zeros_like(hilang)
    modus = np.array([np.bincount(Xl[:, j]).argmax()
                      for j in range(X.shape[1])])
    Xisi = np.where(hilang, modus, Xu)
    P0 = posterior(log_gabungan(m, Xu, tak_hilang))
    print("marginalisasi tanpa nilai hilang = predict_proba:",
          np.allclose(P0, m.predict_proba(Xu)))
    print("data uji                    akurasi  log-loss")
    for nama, P in (
            ("lengkap", posterior(log_gabungan(m, Xu, tak_hilang))),
            ("30% hilang, dilewati", posterior(log_gabungan(m, Xu, hilang))),
            ("30% hilang, diisi modus", m.predict_proba(Xisi))):
        print(f"{nama:26s}  {np.mean(P.argmax(1) == yu):.4f}   "
              f"{log_loss(yu, P):.4f}")

"""Bab 7: memilih alpha pada teks bangkitan.

Kosakata 1000 kata dengan frekuensi berhukum Zipf. Setiap kelas
mempunyai peluang kata sendiri (frekuensi dasar dikali faktor acak),
dan setiap dokumen berisi 30 token. Fitur = kehadiran kata (0/1),
jadi model yang dipakai BernoulliNB. Kita ukur log-loss dan akurasi
data uji untuk berbagai alpha dan ukuran data latih, lalu melihat
alpha yang dipilih validasi silang lima lipatan.
"""
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB

from bab04_data import BENIH

V, PANJANG = 1000, 30
ALPHA = [1e-10, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def peluang_kata(rng):
    dasar = 1 / np.arange(1, V + 1) ** 1.1
    q = dasar * np.exp(0.35 * rng.standard_normal((2, V)))
    return q / q.sum(axis=1, keepdims=True)


def bangkit(n, q, rng):
    y = rng.integers(0, 2, n)
    X = np.zeros((n, V))
    for i in range(n):
        kata = rng.choice(V, PANJANG, p=q[y[i]])
        X[i, kata] = 1
    return X, y


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    q = peluang_kata(rng)
    Xu, yu = bangkit(20_000, q, rng)
    print("log-loss uji (akurasi) untuk setiap alpha")
    print("alpha      n=50           n=500          n=5000")
    hasil = {}
    data = {n: bangkit(n, q, rng) for n in (50, 500, 5000)}
    for a in ALPHA:
        baris = []
        for n, (X, y) in data.items():
            nb = BernoulliNB(alpha=a).fit(X, y)
            ll = log_loss(yu, nb.predict_proba(Xu))
            ak = accuracy_score(yu, nb.predict(Xu))
            baris.append(f"{ll:6.3f} ({ak:.3f})")
        teks = "0" if a < 1e-9 else f"{a:g}"
        print(f"{teks:5s}  " + "   ".join(baris))
    X, y = data[500]
    nb = BernoulliNB(alpha=10.0).fit(X, y)
    print(f"n = 500: kelas {np.bincount(y)}, alpha = 10 "
          f"menebak 0 untuk {np.sum(nb.predict(Xu) == 0)}")
    print("alpha pilihan validasi silang (neg_log_loss, 5 lipatan):")
    for n, (X, y) in data.items():
        cv = GridSearchCV(BernoulliNB(), {"alpha": ALPHA[1:]},
                          scoring="neg_log_loss", cv=5).fit(X, y)
        print(f"  n = {n:4d}: alpha = {cv.best_params_['alpha']:g}")

"""Bab 8: alpha dan ukuran data latih pada data jamur.

Untuk setiap ukuran n, 40 kali diambil n jamur latih secara acak dan
sisanya dipakai sebagai data uji. Dicatat rata-rata akurasi, log-loss,
dan bagian jamur uji yang peluang kelas sebenarnya < 1e-6 ("yakin
salah").
"""
import numpy as np
from sklearn.metrics import log_loss
from sklearn.naive_bayes import CategoricalNB

from bab04_data import BENIH
from bab08_data import muat_kode

X, y, _ = muat_kode()
mc = X.max(axis=0) + 1
rng = np.random.default_rng(BENIH)
ALPHA = [1e-10, 0.01, 0.1, 1.0, 10.0]


def percobaan(n, alpha, ulang=40):
    ak, ll, ys = [], [], []
    for _ in range(ulang):
        latih = rng.choice(len(y), n, replace=False)
        uji = np.setdiff1d(np.arange(len(y)), latih)
        m = CategoricalNB(alpha=alpha, min_categories=mc)
        P = m.fit(X[latih], y[latih]).predict_proba(X[uji])
        benar = P[np.arange(len(uji)), y[uji]]
        ak.append(np.mean(P.argmax(axis=1) == y[uji]))
        ll.append(log_loss(y[uji], P))
        ys.append(np.mean(benar < 1e-6))
    return np.mean(ak), np.mean(ll), np.mean(ys)


if __name__ == "__main__":
    print("   n  alpha   akurasi  log-loss  yakin salah")
    for n in (30, 100, 1000):
        for a in ALPHA:
            ak, ll, ys = percobaan(n, a)
            teks = "0" if a < 1e-9 else f"{a:g}"
            print(f"{n:4d}  {teks:5s}    {ak:.3f}    {ll:6.3f}     "
                  f"{ys:.3f}")

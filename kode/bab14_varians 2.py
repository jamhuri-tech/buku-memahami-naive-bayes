"""Bab 14: rata-rata dan varians bertahap.

Satu juta nilai dari N(1e8, 1), diolah dalam potongan 10.000. Varians
dihitung dengan empat cara: rumus jumlah kuadrat (E[x^2] - E[x]^2),
penggabungan Chan-Golub-LeVeque, GaussianNB.partial_fit, dan np.var
pada seluruh data.
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB

from bab04_data import BENIH


def gabung(a, b):
    """Menggabungkan (n, rata-rata, M2) dua potongan."""
    na, ma, Ma = a
    nb, mb, Mb = b
    n = na + nb
    d = mb - ma
    return n, ma + d * nb / n, Ma + Mb + d * d * na * nb / n


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    x = rng.normal(1e8, 1.0, 1_000_000)
    s1 = s2 = 0.0
    chan = (0, 0.0, 0.0)
    g = GaussianNB()
    for p in np.split(x, 100):
        s1 += p.sum()
        s2 += (p * p).sum()
        M2 = ((p - p.mean()) ** 2).sum()
        chan = gabung(chan, (len(p), p.mean(), M2))
        g.partial_fit(p[:, None], np.zeros(len(p)), classes=[0])
    n = len(x)
    print(f"jumlah kuadrat    : {s2 / n - (s1 / n) ** 2:12.6f}")
    print(f"Chan dkk.         : {chan[2] / chan[0]:12.6f}")
    print(f"GaussianNB        : {g.var_[0, 0] - g.epsilon_:12.6f}")
    print(f"np.var sekaligus  : {np.var(x):12.6f}")

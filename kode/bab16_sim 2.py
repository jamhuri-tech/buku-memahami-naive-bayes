"""Bab 16: kurva belajar naive Bayes lawan regresi logistik pada data
bangkitan yang galat Bayes-nya dapat dihitung.

Model A (asumsi naif benar): 30 fitur biner yang bebas bersyarat kelas.
Model B (asumsi dilanggar): 40 fitur biner yang bebas bersyarat kelas,
ditambah satu fitur lemah yang disalin enam kali (setiap salinan
terbalik dengan peluang 0,05), sehingga 46 fitur.
Untuk setiap ukuran data latih n, 40 ulangan; galat diukur pada 20.000
pengamatan uji. Regresi logistik memakai penalti L2 bawaan (C = 1).
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

from bab04_data import BENIH

UKURAN = [10, 20, 50, 100, 200, 500, 1000, 5000]


def model_a(rng):
    th = np.vstack([rng.uniform(0.2, 0.6, 30), rng.uniform(0.4, 0.8, 30)])

    def bangkit(n):
        y = rng.integers(0, 2, n)
        return (rng.random((n, 30)) < th[y]).astype(float), y
    return bangkit


def model_b(rng):
    th = np.vstack([rng.uniform(0.3, 0.6, 40), rng.uniform(0.4, 0.7, 40)])

    def bangkit(n):
        y = rng.integers(0, 2, n)
        x = rng.random((n, 40)) < th[y]
        z = rng.random(n) < np.where(y == 1, 0.65, 0.35)
        Z = np.repeat(z[:, None], 6, axis=1)      # enam salinan
        Z ^= rng.random(Z.shape) < 0.05           # sedikit bising
        return np.hstack([x, Z]).astype(float), y
    return bangkit


def kurva(bangkit, rng, ulang=40):
    Xu, yu = bangkit(20_000)
    hasil = []
    for n in UKURAN:
        g_nb, g_lr = [], []
        for _ in range(ulang):
            X, y = bangkit(n)
            if len(np.unique(y)) < 2:
                continue
            g_nb.append(np.mean(BernoulliNB().fit(X, y).predict(Xu) != yu))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                lr = LogisticRegression(max_iter=2000).fit(X, y)
            g_lr.append(np.mean(lr.predict(Xu) != yu))
        hasil.append((n, np.mean(g_nb), np.mean(g_lr)))
    return hasil


if __name__ == "__main__":
    for nama, pembuat in (("A: asumsi naif benar", model_a),
                          ("B: satu fitur disalin enam kali", model_b)):
        rng = np.random.default_rng(BENIH)
        print(nama)
        print("     n   galat NB   galat LR")
        for n, a, b in kurva(pembuat(rng), rng):
            print(f"{n:6d}    {a:.4f}     {b:.4f}")

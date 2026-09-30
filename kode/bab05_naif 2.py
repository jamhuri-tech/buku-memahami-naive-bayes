"""Bab 5: taksiran naif lawan tabel pola utuh pada sepuluh pesan.

Kemungkinan kelas naif: P(x | k) = hasil kali P(x_j | k), dengan
P(x_j = 1 | k) ditaksir sebagai bagian pesan kelas k yang memuat kata
j. Tanpa penghalusan (Bab 7).
"""
from collections import Counter
from itertools import product

import numpy as np

from bab04_data import KATA, PESAN_X, PESAN_Y


def latih_naif(X, y):
    prior = np.array([np.mean(y == k) for k in (0, 1)])
    theta = np.array([X[y == k].mean(axis=0) for k in (0, 1)])
    return prior, theta


def posterior_naif(x, prior, theta):
    # P(x_j | k) = theta jika x_j = 1, dan 1 - theta jika x_j = 0
    kemungkinan = np.prod(np.where(x == 1, theta, 1 - theta), axis=1)
    skor = prior * kemungkinan
    return skor / skor.sum()


if __name__ == "__main__":
    prior, theta = latih_naif(PESAN_X, PESAN_Y)
    print("prior (biasa, penipuan):", prior)
    print("P(kata | kelas)      "
          + "  ".join(f"{k:>8s}" for k in KATA))
    for k, nama in ((1, "penipuan"), (0, "biasa")):
        print(f"  {nama:8s}           "
              + "  ".join(f"{t:8.3f}" for t in theta[k]))

    hit = {k: Counter(tuple(b) for b in PESAN_X[PESAN_Y == k])
           for k in (0, 1)}
    print()
    print("pola  tabel utuh   naif")
    for x in product((1, 0), repeat=4):
        a, b = hit[1][x] / 4, hit[0][x] / 6
        pb = 0.4 * a + 0.6 * b
        utuh = f"{0.4 * a / pb:6.3f}" if pb > 0 else "   0/0"
        naif = posterior_naif(np.array(x), prior, theta)[1]
        print("".join(map(str, x)), f"  {utuh}     {naif:6.3f}")

    # Pembanding: BernoulliNB scikit-learn dengan penghalusan yang
    # sangat kecil (Bab 7 membahas alpha).
    from sklearn.naive_bayes import BernoulliNB

    nb = BernoulliNB(alpha=1e-10).fit(PESAN_X, PESAN_Y)
    p = nb.predict_proba([[1, 1, 0, 0], [1, 0, 1, 0]])[:, 1]
    print()
    print(f"BernoulliNB: pola 1100 -> {p[0]:.3f}, "
          f"pola 1010 -> {p[1]:.3f}")

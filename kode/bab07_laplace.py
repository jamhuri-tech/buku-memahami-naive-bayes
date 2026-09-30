"""Bab 7: penghalusan Laplace pada sepuluh pesan.

theta_kj = (N_kj + alpha) / (N_k + 2 alpha) untuk fitur biner.
Prior tidak dihaluskan (sama dengan scikit-learn).
"""
from itertools import product

import numpy as np
from sklearn.naive_bayes import BernoulliNB

from bab04_data import KATA, PESAN_X, PESAN_Y
from bab05_naif import posterior_naif


def latih_halus(X, y, alpha):
    prior = np.array([np.mean(y == k) for k in (0, 1)])
    Nk = np.array([np.sum(y == k) for k in (0, 1)])
    Nkj = np.array([X[y == k].sum(axis=0) for k in (0, 1)])
    theta = (Nkj + alpha) / (Nk[:, None] + 2 * alpha)
    return prior, theta


if __name__ == "__main__":
    prior, theta = latih_halus(PESAN_X, PESAN_Y, alpha=1.0)
    print("alpha = 1          " + "  ".join(f"{k:>8s}" for k in KATA))
    for k, nama in ((1, "penipuan"), (0, "biasa")):
        print(f"  {nama:8s}         "
              + "  ".join(f"{t:8.3f}" for t in theta[k]))
    _, theta0 = latih_halus(PESAN_X, PESAN_Y, alpha=0.0)
    print()
    print("pola  alpha=0  alpha=1")
    for x in product((1, 0), repeat=4):
        x = np.array(x)
        a0 = posterior_naif(x, prior, theta0)[1]
        a1 = posterior_naif(x, prior, theta)[1]
        print("".join(map(str, x)), f"  {a0:6.3f}   {a1:6.3f}")
    nb = BernoulliNB(alpha=1.0).fit(PESAN_X, PESAN_Y)
    q = nb.predict_proba([[1, 0, 0, 1], [1, 1, 0, 0]])[:, 1]
    print()
    print(f"BernoulliNB(alpha=1): 1001 -> {q[0]:.4f}, "
          f"1100 -> {q[1]:.4f}")

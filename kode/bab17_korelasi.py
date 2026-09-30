"""Bab 17: dua fitur Gaussian yang berkorelasi di dalam kelas.

Kelas 0 ~ N(mu0, S0), kelas 1 ~ N(mu1, S1), prior sama, varians setiap
fitur 1. Naive Bayes Gaussian (dilatih pada 20.000 data) dibandingkan
dengan pengklasifikasi Bayes optimal yang memakai sebaran sebenarnya,
pada 200.000 data uji.
"""
import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.naive_bayes import GaussianNB

from bab04_data import BENIH


def kov(rho):
    return np.array([[1.0, rho], [rho, 1.0]])


def percobaan(mu1, rho0, rho1, rng):
    mu0 = np.zeros(2)
    S = [kov(rho0), kov(rho1)]
    mu = [mu0, np.asarray(mu1, float)]

    def bangkit(n):
        y = rng.integers(0, 2, n)
        X = np.empty((n, 2))
        for k in (0, 1):
            m = y == k
            X[m] = rng.multivariate_normal(mu[k], S[k], m.sum())
        return X, y
    Xl, yl = bangkit(20_000)
    Xu, yu = bangkit(200_000)
    q_nb = GaussianNB().fit(Xl, yl).predict_proba(Xu)[:, 1]
    f = [mvn(mu[k], S[k]).logpdf(Xu) for k in (0, 1)]
    q_b = 1 / (1 + np.exp(f[0] - f[1]))

    def ll(q):
        q = np.clip(q, 1e-15, 1 - 1e-15)
        return -np.mean(yu * np.log(q) + (1 - yu) * np.log(1 - q))
    return (np.mean((q_nb > 0.5) == yu), np.mean((q_b > 0.5) == yu),
            np.mean((q_nb > 0.5) == (q_b > 0.5)), ll(q_nb), ll(q_b))


KASUS = [((1, 1), 0.0, 0.0), ((1, 1), 0.5, 0.5), ((1, 1), 0.9, 0.9),
         ((1, 0.3), 0.9, 0.9), ((1, 1), 0.9, -0.9)]

if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    print(" " * 21 + "akurasi" + " " * 15 + "log-loss")
    print("mu1      r0    r1    NB     opt.    sama   NB     opt.")
    for mu1, r0, r1 in KASUS:
        a, b, c, d, e = percobaan(mu1, r0, r1, rng)
        m = f"({mu1[0]:g}, {mu1[1]:g})"
        print(f"{m:8s} {r0:+.1f}  {r1:+.1f}  {a:.3f}  {b:.3f}   {c:.3f}"
              f"  {d:.3f}  {e:.3f}")

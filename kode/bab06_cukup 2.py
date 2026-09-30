"""Bab 6: statistik cukup naive Bayes Bernoulli.

Pelatihan hanya memerlukan dua hitungan: banyaknya pengamatan setiap
kelas dan banyaknya kehadiran setiap fitur di setiap kelas. Keduanya
diperoleh dengan satu perkalian matriks one-hot, dan hitungan dari
dua bagian data cukup dijumlahkan.
"""
import numpy as np
from sklearn.naive_bayes import BernoulliNB

from bab04_data import BENIH


def hitungan(X, y, K):
    Y = np.eye(K)[y]             # one-hot, n x K
    return Y.sum(axis=0), Y.T @ X  # N_k dan N_kj


def taksiran(Nk, Nkj):
    prior = Nk / Nk.sum()
    theta = Nkj / Nk[:, None]
    return prior, theta


rng = np.random.default_rng(BENIH)
n, p, K = 100_000, 50, 3
theta_benar = rng.uniform(0.02, 0.6, size=(K, p))
y = rng.integers(0, K, n)
X = (rng.random((n, p)) < theta_benar[y]).astype(float)

Nk, Nkj = hitungan(X, y, K)
prior, theta = taksiran(Nk, Nkj)
nb = BernoulliNB(alpha=1e-10).fit(X, y)
print("N_k            :", Nk.astype(int))
print("sama dengan class_count_  :",
      np.array_equal(Nk, nb.class_count_))
print("sama dengan feature_count_:",
      np.array_equal(Nkj, nb.feature_count_))
print(f"maks |theta - theta_benar| = "
      f"{np.abs(theta - theta_benar).max():.4f}")

# dua bagian data, hitungannya dijumlahkan
Na, Nja = hitungan(X[:40_000], y[:40_000], K)
Nb, Njb = hitungan(X[40_000:], y[40_000:], K)
p2, t2 = taksiran(Na + Nb, Nja + Njb)
print("gabungan dua bagian = data utuh:",
      np.array_equal(t2, theta) and np.array_equal(p2, prior))
# rata-rata dua taksiran TIDAK sama dengan taksiran data utuh
ta, tb = taksiran(Na, Nja)[1], taksiran(Nb, Njb)[1]
print(f"maks |rata-rata dua taksiran - theta| = "
      f"{np.abs((ta + tb) / 2 - theta).max():.5f}")

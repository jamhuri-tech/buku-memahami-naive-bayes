"""Bab 17: dua keadaan ketika naive Bayes gagal total.

(1) XOR biner: kelas = apakah dua koin sama (Bab 5).
(2) Dua kelas Gaussian dengan rata-rata dan varians sama, hanya arah
    korelasinya yang berbeda (+0,8 lawan -0,8).
Naive Bayes dibandingkan dengan model yang dapat menangkap
ketergantungan: naive Bayes kategorik pada pasangan fitur (1) dan QDA (2).
"""
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB

from bab04_data import BENIH

rng = np.random.default_rng(BENIH)

# (1) XOR biner
def xor(n):
    X = rng.integers(0, 2, (n, 2))
    return X, (X[:, 0] == X[:, 1]).astype(int)

Xl, yl = xor(10_000)
Xu, yu = xor(100_000)
nb = BernoulliNB().fit(Xl, yl)
pasangan = lambda X: (2 * X[:, 0] + X[:, 1])[:, None]
cat = CategoricalNB().fit(pasangan(Xl), yl)
print("XOR biner")
print(f"  naive Bayes, dua fitur      : akurasi {nb.score(Xu, yu):.4f}")
print(f"  posterior NB berkisar       : {nb.predict_proba(Xu)[:, 1].min():.3f}"
      f" - {nb.predict_proba(Xu)[:, 1].max():.3f}")
print(f"  satu fitur pasangan (4 nilai): akurasi "
      f"{cat.score(pasangan(Xu), yu):.4f}")


# (2) hanya korelasi yang berbeda
def korelasi(n):
    y = rng.integers(0, 2, n)
    X = np.empty((n, 2))
    for k, r in ((0, 0.8), (1, -0.8)):
        m = y == k
        X[m] = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], m.sum())
    return X, y

Xl, yl = korelasi(10_000)
Xu, yu = korelasi(100_000)
print("korelasi +0,8 lawan -0,8")
print(f"  naive Bayes Gaussian        : akurasi "
      f"{GaussianNB().fit(Xl, yl).score(Xu, yu):.4f}")
print(f"  QDA (kovarians penuh)       : akurasi "
      f"{QuadraticDiscriminantAnalysis().fit(Xl, yl).score(Xu, yu):.4f}")

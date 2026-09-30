"""Bab 7: rumus penghalusan di tiga kelas scikit-learn, dan alpha = 0.

Setiap taksiran scikit-learn dibandingkan dengan rumus
(hitungan + alpha) / (total + alpha * banyaknya nilai).
"""
import warnings

import numpy as np
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, MultinomialNB

from bab04_data import PESAN_X, PESAN_Y

a = 1.0
y = PESAN_Y

# Bernoulli: dua nilai (hadir / tidak) per kata
nb = BernoulliNB(alpha=a).fit(PESAN_X, y)
Nk = np.bincount(y)
rumus = (nb.feature_count_ + a) / (Nk[:, None] + 2 * a)
print("Bernoulli  : cocok",
      np.allclose(np.exp(nb.feature_log_prob_), rumus))

# Multinomial: satu sebaran atas semua p kata per kelas
H = PESAN_X * np.array([2, 1, 1, 3])      # hitungan kata rekaan
nb = MultinomialNB(alpha=a).fit(H, y)
Nkj = nb.feature_count_
rumus = (Nkj + a) / (Nkj.sum(axis=1, keepdims=True) + a * H.shape[1])
print("Multinomial: cocok",
      np.allclose(np.exp(nb.feature_log_prob_), rumus))

# Kategorik: fitur 0 bernilai 0, 1, atau 2
C = np.column_stack([PESAN_X[:, 0] + PESAN_X[:, 1], PESAN_X[:, 3]])
nb = CategoricalNB(alpha=a).fit(C, y)
N0 = nb.category_count_[0]                 # K x 3 untuk fitur 0
rumus = (N0 + a) / (N0.sum(axis=1, keepdims=True) + a * 3)
print("Kategorik  : cocok",
      np.allclose(np.exp(nb.feature_log_prob_[0]), rumus))

# alpha = 0: dengan dan tanpa force_alpha
print()
for fa in (False, True):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        nb = BernoulliNB(alpha=0.0, force_alpha=fa).fit(PESAN_X, y)
        q = nb.predict_proba([[1, 0, 0, 1]])[0, 1]
    pesan = str(w[0].message)[:32] if w else "-"
    print(f"force_alpha={fa!s:5s}: P(pen | 1001) = {q:.3g}")
    print(f"   peringatan: {pesan}...")

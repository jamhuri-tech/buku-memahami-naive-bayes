"""Bab 13: kelas buatan sendiri dicocokkan dengan scikit-learn pada
empat data: sepuluh pesan (Bernoulli), jamur (kategorik), SMS spam
(multinomial, matriks jarang), dan kanker payudara (Gaussian).
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import (BernoulliNB, CategoricalNB, GaussianNB,
                                 MultinomialNB)

from bab04_data import PESAN_X, PESAN_Y
from bab08_data import muat_kode
from bab09_data import muat as muat_kanker
from bab10_data import muat as muat_sms
from bab13_nb import Bernoulli, Gaussian, Kategorik, Multinomial

Xj, yj, _ = muat_kode()
teks, ys = muat_sms()
Xs = CountVectorizer().fit_transform(teks)
Xk, yk = muat_kanker()
kasus = [
    ("Bernoulli, pesan", Bernoulli(1.0), BernoulliNB(alpha=1.0),
     PESAN_X, PESAN_Y),
    ("kategorik, jamur", Kategorik(1.0), CategoricalNB(alpha=1.0),
     Xj, yj),
    ("multinomial, SMS", Multinomial(1.0), MultinomialNB(alpha=1.0),
     Xs, ys),
    ("Gaussian, kanker", Gaussian(), GaussianNB(), Xk, yk),
]
print("keluarga           maks |selisih log-posterior|  tebakan sama")
for nama, kita, skl, X, y in kasus:
    a = kita.fit(X, y).predict_log_proba(X)
    b = skl.fit(X, y).predict_log_proba(X)
    sama = np.array_equal(kita.predict(X), skl.predict(X))
    print(f"{nama:18s}        {np.abs(a - b).max():9.2e}"
          f"             {sama}")

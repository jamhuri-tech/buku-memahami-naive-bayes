"""Bab 16: kurva belajar pada data nyata.

Untuk setiap ukuran n, n pengamatan latih diambil acak berstrata dan
sisanya menjadi data uji; 20 ulangan. SMS spam: naive Bayes multinomial
lawan regresi logistik pada hitungan kata (kosakata dari data latih).
Kanker payudara: naive Bayes Gaussian lawan regresi logistik pada fitur
yang dibakukan.
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bab04_data import BENIH
from bab09_data import muat as muat_kanker
from bab10_data import muat as muat_sms


def kurva(X, y, ukuran, nb, lr, ulang=20):
    hasil = []
    for n in ukuran:
        a, b = [], []
        for u in range(ulang):
            Xl, Xu, yl, yu = train_test_split(
                X, y, train_size=n, stratify=y, random_state=BENIH + u)
            a.append(np.mean(nb().fit(Xl, yl).predict(Xu) != yu))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                b.append(np.mean(lr().fit(Xl, yl).predict(Xu) != yu))
        hasil.append((n, np.mean(a), np.mean(b)))
    return hasil


teks, ys = muat_sms()
Xk, yk = muat_kanker()
KASUS = [
    ("SMS spam", np.array(teks, dtype=object), ys,
     [20, 50, 100, 200, 500, 1000, 4000],
     lambda: make_pipeline(CountVectorizer(), MultinomialNB()),
     lambda: make_pipeline(CountVectorizer(),
                           LogisticRegression(max_iter=2000))),
    ("kanker payudara", Xk, yk, [10, 20, 50, 100, 200, 400],
     GaussianNB,
     lambda: make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=2000))),
]

if __name__ == "__main__":
    for nama, X, y, ukuran, nb, lr in KASUS:
        print(nama)
        print("     n   galat NB   galat LR")
        for n, a, b in kurva(X, y, ukuran, nb, lr):
            print(f"{n:6d}    {a:.4f}     {b:.4f}")

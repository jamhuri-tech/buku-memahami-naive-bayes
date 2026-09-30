"""Pemeriksa Contoh Soal Bab 10 (pecahan eksak)."""
from fractions import Fraction as F
from math import factorial

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

# Enam dokumen rekaan; kolom: hadiah, transfer, rapat.
H = np.array([[2, 1, 0], [1, 2, 0], [1, 0, 0],      # penipuan
              [0, 1, 2], [1, 0, 1], [0, 1, 2]])     # biasa
y = np.array([1, 1, 1, 0, 0, 0])
q = np.array([[2, 0, 1]])                           # hadiah hadiah rapat

# ---- Contoh Soal 10.1: CountVectorizer ----
vek = CountVectorizer().fit(["Free entry in 2 a wkly comp"])
assert list(vek.get_feature_names_out()) == ["comp", "entry", "free",
                                             "in", "wkly"]

# ---- Contoh Soal 10.2: multinomial ----
assert H[y == 1].sum(axis=0).tolist() == [4, 3, 0]
assert H[y == 0].sum(axis=0).tolist() == [1, 2, 5]
t1 = [F(4 + 1, 7 + 3), F(3 + 1, 10), F(0 + 1, 10)]
t0 = [F(1 + 1, 8 + 3), F(2 + 1, 11), F(5 + 1, 11)]
assert t1 == [F(1, 2), F(2, 5), F(1, 10)]
assert t0 == [F(2, 11), F(3, 11), F(6, 11)]
s1 = t1[0] ** 2 * t1[2]
s0 = t0[0] ** 2 * t0[2]
assert s1 == F(1, 40) and s0 == F(24, 1331)
post = s1 / (s1 + s0)
assert post == F(1331, 2291) and round(float(post), 3) == 0.581
nb = MultinomialNB(alpha=1).fit(H, y)
assert abs(nb.predict_proba(q)[0, 1] - float(post)) < 1e-12
koef = factorial(3) // (factorial(2) * factorial(1))
assert koef == 3
assert s1 / s0 == F(1331, 960)

# ---- Contoh Soal 10.3: Bernoulli ----
B = (H > 0).astype(int)
assert B[y == 1].sum(axis=0).tolist() == [3, 2, 0]
assert B[y == 0].sum(axis=0).tolist() == [1, 2, 3]
b1 = [F(3 + 1, 5), F(2 + 1, 5), F(0 + 1, 5)]
b0 = [F(1 + 1, 5), F(2 + 1, 5), F(3 + 1, 5)]
s1 = b1[0] * (1 - b1[1]) * b1[2]
s0 = b0[0] * (1 - b0[1]) * b0[2]
assert s1 == F(8, 125) and s0 == F(16, 125)
assert s1 / (s1 + s0) == F(1, 3)
nb = BernoulliNB(alpha=1).fit(B, y)
assert abs(nb.predict_proba((q > 0).astype(int))[0, 1] - 1 / 3) < 1e-12

# ---- Contoh Soal 10.4: pesan diulang dua kali ----
r = F(1331, 960)
p2 = r ** 2 / (r ** 2 + 1)
assert r ** 2 == F(1771561, 921600)
assert p2 == F(1771561, 2693161)
assert round(float(p2), 3) == 0.658
nb = MultinomialNB(alpha=1).fit(H, y)
assert abs(nb.predict_proba(2 * q)[0, 1] - float(p2)) < 1e-12

print("Contoh Soal Bab 10: semua bilangan cocok")

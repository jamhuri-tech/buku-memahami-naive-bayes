"""Pemeriksa Contoh Soal Bab 14."""
import math
from fractions import Fraction as F

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from bab14_varians import gabung

# ---- Contoh Soal 14.1: hitungan dua potongan ----
a1 = [3, 1, 0]
a2 = [1, 2, 0]
tot = [u + v for u, v in zip(a1, a2)]
assert tot == [4, 3, 0]
t = [F(c + 1, 7 + 3) for c in tot]
assert t == [F(1, 2), F(2, 5), F(1, 10)]
t1 = [F(c + 1, 4 + 3) for c in a1]
t2 = [F(c + 1, 3 + 3) for c in a2]
assert t1 == [F(4, 7), F(2, 7), F(1, 7)]
assert t2 == [F(1, 3), F(1, 2), F(1, 6)]
assert (t1[0] + t2[0]) / 2 == F(19, 42)
assert round(19 / 42, 3) == 0.452

# ---- Contoh Soal 14.2: penggabungan Chan ----
A = (2, 6.0, 2.0)
B = (2, 10.0, 2.0)
n, m, M2 = gabung(A, B)
assert (n, m, M2) == (4, 8.0, 20.0) and M2 / n == 5.0
assert 4 ** 2 * 2 * 2 / 4 == 16

# ---- Contoh Soal 14.3: rumus jumlah kuadrat gagal ----
x = np.array([1e8 + 1, 1e8 - 1])
ex2 = np.mean(x * x)
assert ex2 == 1e16                      # 1e16 + 1 tidak terwakili
assert np.mean(x) ** 2 == 1e16
assert ex2 - np.mean(x) ** 2 == 0.0
assert np.var(x) == 1.0
assert np.spacing(1e16) == 2.0
n, m, M2 = gabung((1, 1e8 + 1, 0.0), (1, 1e8 - 1, 0.0))
assert M2 / n == 1.0

# ---- Contoh Soal 14.4: tabrakan ember ----
# hadiah dan rapat satu ember; transfer ember lain
pen = [4 + 0, 3]
bia = [1 + 5, 2]
tp = [F(c + 1, 7 + 2) for c in pen]
tb = [F(c + 1, 8 + 2) for c in bia]
assert tp == [F(5, 9), F(4, 9)] and tb == [F(7, 10), F(3, 10)]
lr = (tp[0] / tb[0]) ** 3
assert lr == F(50, 63) ** 3
assert round(float(lr), 3) == 0.5
assert round(float(lr / (1 + lr)), 3) == 0.333

# ---- Contoh Soal 14.5: peluang tabrakan ----
V = 8749
for b, harap in ((14, 0.414), (20, 0.008)):
    assert round(1 - math.exp(-(V - 1) / 2 ** b), 3) == harap
assert round(math.exp(-8748 / 16384), 4) == 0.5863
hv = HashingVectorizer(n_features=2 ** 14, alternate_sign=False,
                       norm=None)
assert hv.n_features == 16384

print("Contoh Soal Bab 14: semua bilangan cocok")

"""Pemeriksa Contoh Soal Bab 18."""
import math
from fractions import Fraction as F

import numpy as np

from bab18_ukuran import brier, ece, logloss

# ---- Contoh Soal 18.1: ECE dua kelompok ----
q = np.array([0.2] * 5 + [0.9] * 5)
y = np.array([1, 1, 0, 0, 0] + [1, 1, 1, 0, 0])
assert abs(ece(q, y) - 0.25) < 1e-12
assert F(5, 10) * abs(F(2, 10) - F(4, 10)) + \
    F(5, 10) * abs(F(9, 10) - F(6, 10)) == F(1, 4)

# ---- Contoh Soal 18.2: Brier dan log-loss ----
q = np.array([0.9, 0.6, 0.2, 0.99])
y = np.array([1, 0, 0, 0])
assert abs(brier(q, y) - 0.347525) < 1e-12
assert round(1.3901 / 4, 4) == 0.3475
suku = [-math.log(0.9), -math.log(0.4), -math.log(0.8), -math.log(0.01)]
assert [round(v, 4) for v in suku] == [0.1054, 0.9163, 0.2231, 4.6052]
assert abs(logloss(q, y) - sum(suku) / 4) < 1e-12
assert round(sum(suku) / 4, 4) == 1.4625
assert round(suku[3] / sum(suku), 3) == 0.787

# ---- Contoh Soal 18.3: Platt ----
a, b = 0.394, -0.936
for t, z, p in ((20, 6.944, 0.99904), (-5, -2.906, 0.0519)):
    assert abs(a * t + b - z) < 1e-12
    assert round(1 / (1 + math.exp(-z)), 5 if p > 0.5 else 4) == p
assert round((1 - 1 / (1 + math.exp(-20))) * 1e9, 2) == 2.06

# ---- Contoh Soal 18.4: prior bergeser ----
odds = F(3, 4) / F(1, 4)
latih = F(2, 10) / F(8, 10)
for pi, harap in ((F(1, 2), F(12, 13)), (F(2, 100), None)):
    baru = odds * (pi / (1 - pi)) / latih
    post = baru / (1 + baru)
    if harap is not None:
        assert baru == 12 and post == harap
    else:
        assert baru == F(12, 49)
        assert post == F(12, 61)
        assert round(float(post), 3) == 0.197
assert round(12 / 13, 3) == 0.923

print("Contoh Soal Bab 18: semua bilangan cocok")

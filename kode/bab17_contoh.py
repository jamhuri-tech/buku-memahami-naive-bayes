"""Pemeriksa Contoh Soal Bab 17."""
import math
from fractions import Fraction as F

import numpy as np
from scipy.stats import multivariate_normal as mvn

# ---- Contoh Soal 17.1: kapan fitur kembar membalik keputusan ----
odds_prior, lr = F(1, 4), 3
satu = odds_prior * lr
dua = odds_prior * lr ** 2
assert satu == F(3, 4) and satu / (1 + satu) == F(3, 7)
assert dua == F(9, 4) and dua / (1 + dua) == F(9, 13)
assert F(3, 7) < F(1, 2) < F(9, 13)
assert round(3 / 7, 3) == 0.429 and round(9 / 13, 3) == 0.692
# membalik tepat bila 1/LR^2 < odds prior < 1/LR (untuk LR > 1)
assert F(1, 9) < odds_prior < F(1, 3)

# ---- Contoh Soal 17.2: fitur kembar tidak membalik ----
pr = F(8, 20)
s1, s0 = pr * F(6, 8) ** 2, (1 - pr) * F(2, 12) ** 2
assert s1 / (s1 + s0) == F(27, 29)
s1, s0 = pr * F(2, 8) ** 2, (1 - pr) * F(10, 12) ** 2
assert s1 == F(1, 40) and s0 == F(5, 12)
assert s1 / (s1 + s0) == F(3, 53)
assert round(3 / 53, 3) == 0.057 and round(1 / 6, 3) == 0.167

# ---- Contoh Soal 17.3: hanya korelasi yang berbeda ----
x = np.array([1.0, 1.0])
Q0 = (1 - 2 * 0.8 + 1) / (1 - 0.64)
Q1 = (1 + 2 * 0.8 + 1) / (1 - 0.64)
assert round(Q0, 4) == 1.1111 and round(Q1, 4) == 10.0
lo = -Q0 / 2 + Q1 / 2
assert round(lo, 4) == 4.4444
l0 = mvn([0, 0], [[1, 0.8], [0.8, 1]]).logpdf(x)
l1 = mvn([0, 0], [[1, -0.8], [-0.8, 1]]).logpdf(x)
assert abs((l0 - l1) - lo) < 1e-12
assert round(1 / (1 + math.exp(-lo)), 4) == 0.9884

# ---- Contoh Soal 17.4: arah batas optimal lawan naive Bayes ----
def arah(dmu, rho):
    S = np.array([[1, rho], [rho, 1]])
    return np.linalg.solve(S, dmu)
w = arah(np.array([1.0, 1.0]), 0.9)
assert np.allclose(w, [1 / 1.9, 1 / 1.9])
w = arah(np.array([1.0, 0.3]), 0.9)
assert np.allclose(w, [0.73 / 0.19, -0.6 / 0.19])
assert round(0.73 / 0.19, 3) == 3.842 and round(-0.6 / 0.19, 3) == -3.158
cos = w @ np.array([1, 0.3]) / (np.linalg.norm(w) * math.hypot(1, 0.3))
assert round(math.degrees(math.acos(cos)), 1) == 56.1

print("Contoh Soal Bab 17: semua bilangan cocok")

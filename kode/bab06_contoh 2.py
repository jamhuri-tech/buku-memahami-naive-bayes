"""Pemeriksa Contoh Soal Bab 6 (pecahan eksak)."""
import math
from fractions import Fraction as F

import numpy as np

# ---- Contoh Soal 6.1: MLE Bernoulli, 3 dari 4 pesan ----
def L(t):
    return t ** 3 * (1 - t)
assert L(F(3, 4)) == F(27, 256)
assert L(F(1, 2)) == F(1, 16)
assert round(27 / 256, 4) == 0.1055
# turunan log-kemungkinan: 3/t - 1/(1 - t) = 0 di t = 3/4
t = F(3, 4)
assert 3 / t - 1 / (1 - t) == 0
# turunan kedua -3/t^2 - 1/(1-t)^2 < 0
assert -3 / t**2 - 1 / (1 - t)**2 == F(-64, 3)

# ---- Contoh Soal 6.2: prior tiga kelas ----
N = [3, 4, 5]
pi = [F(v, sum(N)) for v in N]
assert pi == [F(1, 4), F(1, 3), F(5, 12)]
lam = sum(N)                    # pengali Lagrange = n
assert lam == 12
assert all(v / p == lam for v, p in zip(N, pi))

# ---- Contoh Soal 6.3: Gaussian ----
g = [5, 7, 9, 11]
mu = F(sum(g), 4)
assert mu == 8
ss = sum((v - mu) ** 2 for v in g)
assert ss == 20
assert ss / 4 == 5 and ss / 3 == F(20, 3)
assert round(20 / 3, 3) == 6.667
# log-kemungkinan di (8, 5) lebih besar daripada di (8, 20/3)
def ll(m, s2):
    return sum(-0.5 * math.log(2 * math.pi * s2) - (v - m) ** 2 / (2 * s2)
               for v in g)
assert ll(8, 5) > ll(8, 20 / 3)
# var_smoothing scikit-learn: 1e-9 x varians terbesar
assert abs(1e-9 * 5 - 5e-9) < 1e-20

# ---- Contoh Soal 6.4: menggabungkan dua kelompok ----
gab = F(3 + 2, 4 + 6)
assert gab == F(1, 2)
rata = (F(3, 4) + F(2, 6)) / 2
assert rata == F(13, 24) and round(float(rata), 3) == 0.542
# rata-rata tertimbang dengan bobot N sama dengan gabungan
assert (4 * F(3, 4) + 6 * F(2, 6)) / 10 == gab

# ---- Contoh Soal 6.5: peluang hitungan nol ----
assert round(0.95 ** 20, 4) == 0.3585
assert round(0.95 ** 4, 4) == 0.8145
Nmin = math.ceil(math.log(0.01) / math.log(0.95))
assert Nmin == 90
assert 0.95 ** 90 < 0.01 < 0.95 ** 89
assert round(math.log(0.01) / math.log(0.95), 2) == 89.78
assert round(0.95 ** 90, 4) == 0.0099

print("Contoh Soal Bab 6: semua bilangan cocok")

"""Pemeriksa Contoh Soal Bab 9."""
import math
from fractions import Fraction as F

import numpy as np
from scipy.stats import norm

from bab09_data import FITUR, muat

# ---- Contoh Soal 9.1: banyaknya angka dalam pesan ----
p1 = math.exp(-0.4) / math.sqrt(10 * math.pi)    # N(6; 8, 5)
p0 = math.exp(-4.0) / math.sqrt(4 * math.pi)     # N(6; 2, 2)
assert abs(p1 - norm.pdf(6, 8, math.sqrt(5))) < 1e-15
assert abs(p0 - norm.pdf(6, 2, math.sqrt(2))) < 1e-15
assert round(p1, 4) == 0.1196 and round(p0, 5) == 0.00517
assert round(math.exp(-0.4), 4) == 0.6703 and round(math.exp(-4), 5) == 0.01832
assert round(math.sqrt(10 * math.pi), 3) == 5.605
assert round(math.sqrt(4 * math.pi), 3) == 3.545
lo = 0.5 * math.log(2 / 5) - 0.4 + 4.0
assert abs(lo - math.log(p1 / p0)) < 1e-12
assert round(lo, 3) == 3.142 and round(0.5 * math.log(0.4), 3) == -0.458
assert round(p1 / p0, 1) == 23.1
assert round(1 / (1 + math.exp(-lo)), 3) == 0.959

# ---- Contoh Soal 9.2: kepadatan bukan peluang ----
X, y = muat()
j = FITUR.index("titik cekung (terburuk)")
x = X[:, j]
mu0, s0 = x[y == 0].mean(), x[y == 0].std()
assert round(mu0, 4) == 0.0744 and round(s0, 4) == 0.0357
puncak = 1 / (s0 * math.sqrt(2 * math.pi))
assert round(puncak, 1) == 11.2
assert round(1 / (0.0357 * math.sqrt(2 * math.pi)), 2) == 11.17
# satuan diubah (x 1000): kepadatan dibagi 1000, rasio tetap
a = norm.pdf(0.14, mu0, s0) / norm.pdf(0.14, x[y == 1].mean(),
                                         x[y == 1].std())
b = norm.pdf(140, 1000 * mu0, 1000 * s0) / norm.pdf(
    140, 1000 * x[y == 1].mean(), 1000 * x[y == 1].std())
assert abs(a - b) < 1e-9 * a

# ---- Contoh Soal 9.3: rata-rata sama, varians beda ----
t = math.sqrt(8 * math.log(2) / 3)
assert round(8 * math.log(2) / 3, 3) == 1.848 and round(t, 3) == 1.36
lo = lambda z: math.log(norm.pdf(z, 0, 2) / norm.pdf(z, 0, 1))
assert abs(lo(t)) < 1e-12 and abs(lo(-t)) < 1e-12
assert abs(lo(0) + math.log(2)) < 1e-12
assert round(lo(3), 3) == round(-math.log(2) + 27 / 8, 3) == 2.682
assert round(1 / (1 + math.exp(-lo(3))), 3) == 0.936

# ---- Contoh Soal 9.4: varians sama, batas linear ----
w1, w2 = F(2 - 0, 1), F(1 - 0, 4)
b = -(F(2**2 - 0, 2 * 1) + F(1**2 - 0, 2 * 4))
assert (w1, w2, b) == (2, F(1, 4), F(-17, 8))
assert w1 * 1 + w2 * F(1, 2) + b == 0
assert w1 * 1 + w2 * 1 + b == F(1, 8)
assert round(1 / (1 + math.exp(-1 / 8)), 4) == 0.5312
def lo2(x1, x2):
    return (math.log(norm.pdf(x1, 2, 1) / norm.pdf(x1, 0, 1))
            + math.log(norm.pdf(x2, 1, 2) / norm.pdf(x2, 0, 2)))
assert abs(lo2(1, 1) - 1 / 8) < 1e-12
assert abs(lo2(0.3, 5.3) - (0.6 + 5.3 / 4 - 17 / 8)) < 1e-12

# ---- Contoh Soal 9.5: var_smoothing ----
eps, vmaks, vk = 1e-9, 1e5, 1e-6
tambah = eps * vmaks
assert abs(tambah - 1e-4) < 1e-18
vbaru = vk + tambah
assert abs(vbaru - 1.01e-4) < 1e-16
assert round(math.sqrt(vbaru) / math.sqrt(vk), 2) == 10.05
assert round(vbaru / vk) == 101

print("Contoh Soal Bab 9: semua bilangan cocok")

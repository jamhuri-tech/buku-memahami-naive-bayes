"""Pemeriksa Contoh Soal Bab 12."""
import math

import numpy as np
from scipy.special import logsumexp

# ---- Contoh Soal 12.1: kapan hasil kali menjadi nol ----
p, k_sub, k_nol = 1.0, None, None
for k in range(1, 400):
    p *= 0.01
    if k_sub is None and p < np.finfo(float).tiny:
        k_sub = k
    if p == 0.0:
        k_nol = k
        break
assert k_sub == 154 and k_nol == 162
assert 0.01 ** 161 > 0
assert round(200 * math.log(0.01), 1) == -921.0
assert round(200 * math.log10(0.01)) == -400

# ---- Contoh Soal 12.2: logsumexp dengan tangan ----
J = np.array([-1000.0, -1001.0, -1003.0])
assert np.all(np.exp(J) == 0)
e = np.exp(J - J.max())
assert round(math.exp(-1), 4) == 0.3679 and round(math.exp(-3), 4) == 0.0498
assert round(e.sum(), 4) == 1.4177
assert np.allclose(np.round(e / e.sum(), 3), [0.705, 0.259, 0.035])
lz = -1000 + math.log(e.sum())
assert round(math.log(e.sum()), 4) == 0.3490
assert abs(lz - logsumexp(J)) < 1e-12

# ---- Contoh Soal 12.3: posterior jenuh ----
p_ham = math.exp(-40) / (1 + math.exp(-40))
assert round(p_ham * 1e18, 2) == 4.25
assert 1 / (1 + math.exp(-40)) == 1.0
assert 1 - 1 / (1 + math.exp(-40)) == 0.0
t = 53 * math.log(2)
assert round(t, 2) == 36.74
assert round(2 ** -53 * 1e16, 2) == 1.11

# ---- Contoh Soal 12.4: log1p ----
a = math.log(1 - 1e-10)
b = math.log1p(-1e-10)
assert abs(b - (-1e-10 - 0.5e-20)) < 1e-24
assert round(abs(a - b) / abs(b) * 1e8, 1) == 8.3
assert math.log(1 - 1e-20) == 0.0 and math.log1p(-1e-20) == -1e-20
assert 1 - 1e-20 == 1.0

print("Contoh Soal Bab 12: semua bilangan cocok")

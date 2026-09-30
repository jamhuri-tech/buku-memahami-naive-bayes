"""Pemeriksa Contoh Soal Bab 5 (pecahan eksak)."""
import math
from fractions import Fraction as F

import numpy as np

from bab04_data import PESAN_X, PESAN_Y

# ---- Contoh Soal 5.1: demam dan batuk ----
pf = F(2, 10)
d = {True: F(8, 10), False: F(1, 10)}      # P(demam | flu?)
b = {True: F(5, 10), False: F(2, 10)}      # P(batuk | flu?)
P_d = pf * d[True] + (1 - pf) * d[False]
P_b = pf * b[True] + (1 - pf) * b[False]
P_db = pf * d[True] * b[True] + (1 - pf) * d[False] * b[False]
assert P_d == F(24, 100) and P_b == F(26, 100)
assert pf * d[True] * b[True] == F(8, 100)
assert (1 - pf) * d[False] * b[False] == F(16, 1000)
assert P_db == F(96, 1000)
assert P_d * P_b == F(624, 10000)
assert P_db / P_d == F(4, 10)
# peluang flu bila demam dan bila demam serta batuk
assert pf * d[True] / P_d == F(2, 3)
assert pf * d[True] * b[True] / P_db == F(5, 6)

# ---- Contoh Soal 5.2: dua koin, kelas = sama ----
# P(x1=1, x2=1) = 1/4 = P(x1=1) P(x2=1)
assert F(1, 4) == F(1, 2) * F(1, 2)
# di kelas y=1 (sama): pasangan (0,0) dan (1,1), masing-masing 1/2
assert F(1, 4) / F(1, 2) == F(1, 2)            # P(1,1 | y=1)
assert F(1, 2) * F(1, 2) == F(1, 4)            # hasil kali marginal
# naive Bayes: semua P(x_j | y) = 1/2, jadi posterior = prior = 1/2
assert F(1, 2) * F(1, 2) * F(1, 2) / (2 * F(1, 2) ** 3) == F(1, 2)

# ---- Contoh Soal 5.3: banyak parameter ----
assert 3 * (4 ** 10 - 1) == 3_145_725
assert 3 * 10 * (4 - 1) == 90
assert 2 * (2 ** 4 - 1) == 30 and 2 * 4 == 8

# ---- Contoh Soal 5.4: naif pada sepuluh pesan ----
t1 = [F(int(PESAN_X[PESAN_Y == 1][:, j].sum()), 4) for j in range(4)]
t0 = [F(int(PESAN_X[PESAN_Y == 0][:, j].sum()), 6) for j in range(4)]
assert t1 == [F(3, 4), F(3, 4), F(3, 4), F(0)]
assert t0 == [F(1, 6), F(2, 6), F(1, 6), F(4, 6)]


def skor(x, prior, t):
    s = prior
    for xj, tj in zip(x, t):
        s *= tj if xj else 1 - tj
    return s


s1 = skor([1, 1, 0, 0], F(2, 5), t1)
s0 = skor([1, 1, 0, 0], F(3, 5), t0)
assert s1 == F(9, 160)
assert s0 == F(1, 108)
post = s1 / (s1 + s0)
assert post == F(243, 283)
assert round(float(post), 3) == 0.859
# hasil kali kemungkinan kelas biasa: (1/6)(2/6)(5/6)(2/6) = 20/1296
assert F(1, 6) * F(2, 6) * F(5, 6) * F(2, 6) == F(20, 1296) == F(5, 324)
assert F(3, 4) * F(3, 4) * F(1, 4) * 1 == F(9, 64)
# pola 1001
s1 = skor([1, 0, 0, 1], F(2, 5), t1)
s0 = skor([1, 0, 0, 1], F(3, 5), t0)
assert s1 == 0 and s0 == F(1, 27)
assert F(1, 6) * F(4, 6) * F(5, 6) * F(4, 6) == F(80, 1296)

# ---- Contoh Soal 5.5: bentuk odds, kata yang digandakan ----
odds_prior = F(2, 10) / F(8, 10)
lr = F(6, 10) / F(5, 100)
assert odds_prior == F(1, 4) and lr == 12
assert odds_prior * lr == 3
assert F(3, 4) == 3 / (1 + F(3))
assert odds_prior * lr ** 2 == 36
assert F(36, 37) == 36 / (1 + F(36))
assert round(36 / 37, 3) == 0.973
assert round(math.log(12), 3) == 2.485

# ---- log-odds pesan 1100 (Subbab 5.3, keluaran bab05_bukti.py) ----
lo = math.log(F(2, 3)) + math.log(F(9, 2)) + math.log(F(9, 4)) \
    + math.log(F(3, 10)) + math.log(3)
assert abs(lo - math.log(F(243, 40))) < 1e-12
assert abs(1 / (1 + math.exp(-lo)) - 243 / 283) < 1e-12

print("Contoh Soal Bab 5: semua bilangan cocok")

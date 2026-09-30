"""Pemeriksa Contoh Soal Bab 2."""
import math
from fractions import Fraction as F

from bab04_data import PESAN_X, PESAN_Y

# ---- Contoh Soal 2.1: peluang bersyarat dari sepuluh pesan ----
had = PESAN_X[:, 0] == 1
pen = PESAN_Y == 1
assert F(int((had & pen).sum()), 10) == F(3, 10)
assert F(int(pen.sum()), 10) == F(2, 5)
assert F(3, 10) / F(2, 5) == F(3, 4)
assert F(int(had.sum()), 10) == F(2, 5)
assert F(3, 10) / F(2, 5) == F(3, 4)          # P(pen | hadiah) juga 3/4

# ---- Contoh Soal 2.2: tiga operator, peluang total dan Bayes ----
pr = [F(5, 10), F(3, 10), F(2, 10)]
sp = [F(2, 100), F(5, 100), F(10, 100)]
tot = sum(a * b for a, b in zip(pr, sp))
assert tot == F(45, 1000)
assert pr[2] * sp[2] / tot == F(4, 9)
assert pr[0] * sp[0] / tot == F(2, 9)
assert round(4 / 9, 3) == 0.444

# ---- Contoh Soal 2.3: multinomial ----
koef = math.factorial(4) // (math.factorial(2) * 1 * 1)
assert koef == 12
p = koef * F(1, 2) ** 2 * F(1, 3) * F(1, 6)
assert p == F(1, 6)

# ---- Contoh Soal 2.4: odds dan log-odds ----
assert F(3, 4) / F(1, 4) == 3
assert round(math.log(3), 4) == 1.0986
assert round(1 / (1 + math.exp(-2)), 4) == 0.8808
assert F(1, 5) / F(4, 5) == F(1, 4)
assert round(math.log(1 / 4), 4) == -1.3863

print("Contoh Soal Bab 2: semua bilangan cocok")

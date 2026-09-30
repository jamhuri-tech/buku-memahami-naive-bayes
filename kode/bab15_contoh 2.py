"""Pemeriksa Contoh Soal Bab 15."""
import math
from fractions import Fraction as F

import numpy as np
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from bab15_linear import linear_bernoulli, linear_multinomial

H = np.array([[2, 1, 0], [1, 2, 0], [1, 0, 0],
              [0, 1, 2], [1, 0, 1], [0, 1, 2]])
y = np.array([1, 1, 1, 0, 0, 0])

# ---- Contoh Soal 15.1: bobot multinomial ----
r = [F(1, 2) / F(2, 11), F(2, 5) / F(3, 11), F(1, 10) / F(6, 11)]
assert r == [F(11, 4), F(22, 15), F(11, 60)]
w = [math.log(v) for v in r]
assert [round(v, 4) for v in w] == [1.0116, 0.3830, -1.6964]
lo = 2 * w[0] + w[2]
assert round(lo, 4) == 0.3268
assert abs(lo - math.log(1331 / 960)) < 1e-12
wn, bn = linear_multinomial(MultinomialNB(alpha=1).fit(H, y))
assert np.allclose(wn, w) and abs(bn) < 1e-12

# ---- Contoh Soal 15.2: bobot Bernoulli ----
t1 = [F(4, 5), F(3, 5), F(1, 5)]
t0 = [F(2, 5), F(3, 5), F(4, 5)]
odds = [a * (1 - b) / (b * (1 - a)) for a, b in zip(t1, t0)]
assert odds == [6, 1, F(1, 16)]
c = sum(math.log((1 - a) / (1 - b)) for a, b in zip(t1, t0))
assert abs(c - math.log(F(4, 3))) < 1e-12
assert round(math.log(6), 4) == 1.7918 and round(math.log(16), 4) == 2.7726
assert round(math.log(4 / 3), 4) == 0.2877
lo = math.log(4 / 3) + math.log(6) + math.log(1 / 16)
assert abs(lo - math.log(0.5)) < 1e-12
wb, bb = linear_bernoulli(BernoulliNB(alpha=1).fit((H > 0) * 1.0, y))
assert np.allclose(wb, [math.log(6), 0, -math.log(16)])
assert abs(bb - math.log(4 / 3)) < 1e-12

# ---- Contoh Soal 15.3: Gaussian, varians berbeda dan sama ----
# sigma 1 dan 2, rata-rata 0: log-odds = -ln 2 + (3/8) x^2
assert F(1, 2) - F(1, 8) == F(3, 8)
# rata-rata 0 dan 2, varians sama 1: log-odds = 2x - 2
assert (2 ** 2 - 0) / 2 == 2
# varians sama 4, rata-rata 0 dan 1: bobot 1/4, intersep -1/8
assert F(1, 4) == F(1 - 0, 4) and F(1 ** 2, 2 * 4) == F(1, 8)

# ---- Contoh Soal 15.4: tiga kelas, batas antarkelas ----
b = np.array([0.0, math.log(1 / 2), math.log(1 / 4)])
W = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
x = np.array([1.0, 1.0])
a = b + W @ x
assert np.allclose(a, [0, 1 - math.log(2), 2 - math.log(4)])
assert round(1 - math.log(2), 4) == 0.3069
assert round(2 - math.log(4), 4) == 0.6137
assert int(np.argmax(a)) == 2
# batas kelas 2 dan 3: x1 - ln2 = 2 x2 - ln4  ->  x1 - 2 x2 = -ln 2
assert abs((a[1] - a[2]) - (1 - 2 - math.log(2) + math.log(4))) < 1e-12

print("Contoh Soal Bab 15: semua bilangan cocok")

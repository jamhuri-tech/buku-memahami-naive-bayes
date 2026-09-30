"""Pemeriksa Contoh Soal Bab 16."""
import math
from fractions import Fraction as F

import numpy as np
from sklearn.linear_model import LogisticRegression

# data: x=1 -> 6 penipuan, 2 biasa; x=0 -> 2 penipuan, 10 biasa
x = np.array([1] * 8 + [0] * 12)
y = np.array([1] * 6 + [0] * 2 + [1] * 2 + [0] * 10)

# ---- Contoh Soal 16.1: satu fitur, keduanya sama ----
prior = F(8, 20)
p1, p0 = F(6, 8), F(2, 12)
post = prior * p1 / (prior * p1 + (1 - prior) * p0)
assert post == F(3, 4)
assert F(6, 8) == F(3, 4)
lr = LogisticRegression(penalty=None).fit(x[:, None], y)
assert abs(lr.predict_proba([[1]])[0, 1] - 0.75) < 1e-4
assert abs(lr.predict_proba([[0]])[0, 1] - 2 / 12) < 1e-4

# ---- Contoh Soal 16.2: fitur kembar ----
s1 = prior * p1 ** 2
s0 = (1 - prior) * p0 ** 2
assert s1 == F(9, 40) and s0 == F(1, 60)
assert s1 / (s1 + s0) == F(27, 29)
assert round(27 / 29, 3) == 0.931
X2 = np.column_stack([x, x])
lr2 = LogisticRegression(penalty=None).fit(X2, y)
assert abs(lr2.predict_proba([[1, 1]])[0, 1] - 0.75) < 1e-4
assert abs(lr2.coef_[0, 0] - lr2.coef_[0, 1]) < 1e-6
assert abs(lr2.coef_.sum() - math.log(15)) < 1e-3    # ln(3) - ln(1/5)

# ---- Contoh Soal 16.3: batas Hoeffding ----
def perlu(p, eps=0.05, delta=0.05):
    return math.log(4 * p / delta) / (2 * eps ** 2)
assert round(math.log(8000), 3) == 8.987
assert round(math.log(800_000), 3) == 13.592
assert math.ceil(perlu(100)) == 1798
assert math.ceil(perlu(10_000)) == 2719
assert round(perlu(10_000) / perlu(100), 2) == 1.51

print("Contoh Soal Bab 16: semua bilangan cocok")

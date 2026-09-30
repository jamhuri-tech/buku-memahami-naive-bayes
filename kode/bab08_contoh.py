"""Pemeriksa Contoh Soal Bab 8 (pecahan eksak)."""
import math
from fractions import Fraction as F

import numpy as np
from sklearn.naive_bayes import BernoulliNB, CategoricalNB

# Delapan jamur rekaan: (bau, spora); bau: 0 busuk, 1 tanpa, 2 almon;
# spora: 0 putih, 1 hitam, 2 coklat. Kelas 1 = beracun.
X = np.array([[0, 0], [0, 1], [1, 0], [0, 0],
              [2, 1], [1, 2], [1, 1], [2, 2]])
y = np.array([1, 1, 1, 1, 0, 0, 0, 0])


def taksir(j, k, a=1, V=3):
    N = np.sum(y == k)
    return [F(int(np.sum((X[:, j] == v) & (y == k))) + a, int(N) + a * V)
            for v in range(V)]


# ---- Contoh Soal 8.1 ----
assert taksir(0, 1) == [F(4, 7), F(2, 7), F(1, 7)]
assert taksir(0, 0) == [F(1, 7), F(3, 7), F(3, 7)]
assert taksir(1, 1) == [F(4, 7), F(2, 7), F(1, 7)]
assert taksir(1, 0) == [F(1, 7), F(3, 7), F(3, 7)]
s1 = F(1, 2) * F(2, 7) * F(2, 7)
s0 = F(1, 2) * F(3, 7) * F(3, 7)
assert s1 == F(2, 49) and s0 == F(9, 98)
assert s1 / (s1 + s0) == F(4, 13)
assert round(4 / 13, 3) == 0.308
m = CategoricalNB(alpha=1).fit(X, y)
assert abs(m.predict_proba([[1, 1]])[0, 1] - 4 / 13) < 1e-12
# jamur (busuk, coklat)
s1, s0 = F(1, 2) * F(4, 7) * F(1, 7), F(1, 2) * F(1, 7) * F(3, 7)
assert s1 / (s1 + s0) == F(4, 7)
assert abs(m.predict_proba([[0, 2]])[0, 1] - 4 / 7) < 1e-12

# ---- Contoh Soal 8.2: bobot bukti bau pada data jamur ----
t1 = F(2160 + 1, 3916 + 9)
t0 = F(0 + 1, 4208 + 9)
assert t1 == F(2161, 3925) and t0 == F(1, 4217)
assert round(float(t1), 5) == 0.55057 and round(float(t0), 6) == 0.000237
assert round(float(t1 / t0), 1) == 2321.8
assert round(math.log(t1 / t0), 2) == 7.75
t1 = F(120 + 1, 3925)
t0 = F(3408 + 1, 4217)
assert round(float(t1), 4) == 0.0308 and round(float(t0), 4) == 0.8084
assert round(float(t1 / t0), 4) == 0.0381
assert round(math.log(t1 / t0), 2) == -3.27
assert round(math.exp(7.75), -1) == 2320

# ---- Contoh Soal 8.3: nilai hilang ----
s1, s0 = F(1, 2) * F(4, 7), F(1, 2) * F(1, 7)
assert s1 / (s1 + s0) == F(4, 5)
# pengisian: busuk (3 kali) atau tanpa bau (3 kali), seri
assert sum(X[:, 0] == 0) == 3 and sum(X[:, 1 - 1] == 1) == 3
s1, s0 = F(1, 2) * F(4, 7) * F(4, 7), F(1, 2) * F(1, 7) * F(1, 7)
assert s1 / (s1 + s0) == F(16, 17)
s1, s0 = F(1, 2) * F(2, 7) * F(4, 7), F(1, 2) * F(3, 7) * F(1, 7)
assert s1 / (s1 + s0) == F(8, 11)
# menjumlahkan atas semua nilai bau sama dengan melewatkan faktornya
for k in (0, 1):
    assert sum(taksir(0, k)) == 1

# ---- Contoh Soal 8.4: one-hot + Bernoulli ----
def bern(n1, N=4, a=1):
    return F(n1 + a, N + 2 * a)
# racun: busuk 3, tanpa 1, almon 0; putih 3, hitam 1, coklat 0
r = (1 - bern(3)) * bern(1) * (1 - bern(0))
assert r == F(2, 6) * F(2, 6) * F(5, 6) == F(20, 216)
# makan: busuk 0, tanpa 2, almon 2; putih 0, hitam 2, coklat 2
mk = (1 - bern(0)) * bern(2) * (1 - bern(2))
assert mk == F(5, 6) * F(3, 6) * F(3, 6) == F(45, 216)
p = F(1, 2) * r * r / (F(1, 2) * r * r + F(1, 2) * mk * mk)
assert p == F(400, 2425) == F(16, 97)
assert round(16 / 97, 3) == 0.165
oh = np.hstack([np.eye(3)[X[:, 0]], np.eye(3)[X[:, 1]]])
q = BernoulliNB(alpha=1).fit(oh, y).predict_proba([[0, 1, 0, 0, 1, 0]])
assert abs(q[0, 1] - 16 / 97) < 1e-12

# ---- Contoh Soal 8.5: kategori baru, min_categories = 4 ----
assert F(0 + 1, 4 + 4) == F(1, 8)
assert F(1, 8) / F(1, 8) == 1
m4 = CategoricalNB(alpha=1, min_categories=[4, 3]).fit(X, y)
assert np.allclose(np.exp(m4.feature_log_prob_[0][:, 3]), [1 / 8, 1 / 8])
# kategori baru tidak mengubah posterior (kelas sama besar)
assert abs(m4.predict_proba([[3, 1]])[0, 1] - 2 / 5) < 1e-12
s1, s0 = F(1, 2) * F(1, 8) * F(2, 7), F(1, 2) * F(1, 8) * F(3, 7)
assert s1 / (s1 + s0) == F(2, 5)

print("Contoh Soal Bab 8: semua bilangan cocok")
